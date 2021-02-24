"""Trainer class.

The basic idea is the class that executes the training loops. It manages a few things:

1. One or more explorer nodes. The explorers use the shared policy to explor the sim
   environment and report back their histories.

2. A copy of the policy. The policy is assumed to contain the necessary prediction methods.
   After a couple epochs, the new policy weights are sent to the explorer nodes that _this_
   trainer node controls.
"""
import copy
import pathlib
import time
from typing import Dict, List

import numpy
import torch
from torch import nn
from torch import distributed
from torch.distributed import rpc
from torch.nn import parallel
from torch.utils import tensorboard
from torch.utils import data

from crowd_search import dataset
from crowd_search import distributed_utils
from crowd_search import explorer
from crowd_search import policy
from crowd_search import shared_storage
from crowd_search.visualization import viz


class Trainer:
    def __init__(
        self,
        cfg: Dict,
        device: torch.device,
        num_learners: int,
        explorer_nodes: List[int],
        rank: int,
        run_dir: pathlib.Path,
        batch_size: int,
        storage_node: int,
    ) -> None:
        self.cfg = cfg
        self.explorer_nodes = explorer_nodes
        self.epochs = 80
        self.device = device
        self.num_learners = num_learners
        self.batch_size = batch_size
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        # If main node, create a logger
        self.is_main = False
        if rank == 0:
            self.logger = tensorboard.SummaryWriter(run_dir)
            self.is_main = True

        # Create the policy
        self.policy = policy.CrowdSearchPolicy(
            cfg.get("models"),
            cfg.get("robot"),
            cfg.get("human"),
            cfg.get("incentives"),
            cfg.get("action-space"),
            device,
        )
        self.policy.to(self.device)
        if num_learners > 1:
            if torch.cuda.is_available():
                self.policy = parallel.DistributedDataParallel(
                    self.policy, device_ids=[rank], find_unused_parameters=True
                )
            else:
                self.policy = parallel.DistributedDataParallel(
                    self.policy, device_ids=None, find_unused_parameters=True
                )

        cfg.get("mu-zero").update({"action-space": list(range(41))}) 
        self.config = cfg.get("mu-zero")

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=1.0e-4, weight_decay=1.0e-4
        )
        self.training_step = 0
        self.dataset = dataset.Dataset(cfg.get("mu-zero"))
        self.explorer_references = []

        storage_info = rpc.get_worker_info(f"Storage:{storage_node}")
        self.storage_node = rpc.remote(storage_info, shared_storage.SharedStorage)

        for explorer_node in explorer_nodes:
            explorer_info = rpc.get_worker_info(f"Explorer:{explorer_node}")
            self.explorer_references.append(
                rpc.remote(
                    explorer_info, explorer.Explorer, args=(cfg, self.storage_node)
                )
            )
        self.global_step = 0

        # Update explorers with initial policy weights
        self._send_policy()
        self._get_history()

    def _send_policy(self):
        model = copy.deepcopy(distributed_utils.unwrap_ddp(self.policy))
        model.to("cpu")
        self.storage_node.rpc_sync().update_policy(model)

    def _get_history(self):
        new_history = []
        while not new_history:
            new_history = self.storage_node.rpc_sync().get_history()
            time.sleep(1.0)

        self.storage_node.rpc_sync().clear_history()
        self._combine_datasets(new_history)

        average_rewards = [
            numpy.mean(history.reward_history) for history in new_history
        ]
        max_rewards = [numpy.max(history.reward_history) for history in new_history]
        
        sim_time = numpy.mean([len(history.reward_history) for history in new_history])
        if self.is_main:
            self.logger.add_scalar(
                "explorer/avg_reward", numpy.mean(average_rewards), self.global_step
            )
            self.logger.add_scalar(
                "explorer/max_reward", numpy.sum(max_rewards), self.global_step
            )
            self.logger.add_scalar(
                "explorer/avg_sim_time", sim_time, self.global_step
            )

    def _combine_datasets(self, histories: List):
        """Called after new history is acquired from local storage node. We want to
        send all new history to each other trainer node."""
        if distributed.is_initialized():
            output = [None] * self.num_learners
            distributed.all_gather_object(output, histories)

            histories = []
            for history in output:
                histories.extend(history)

        self.dataset.transitions.extend(copy.deepcopy(histories))

        if self.is_main:
            for history in histories:
                if 1 in history.reward_history:
                    viz.plot_history(
                        history, self.run_dir / f"plots/{self.global_step}.gif"
                    )

                    break

        if len(self.dataset.transitions) > 2000:
            self.dataset.transitions = self.dataset.transitions[-2000:]

    def continous_train(self) -> None:
        """Target function to call after intialization.

        This function loops over the available training data and trains our models.
        Periodically, a trainer will look out to the explorer nodes for new data. At
        this point, our trainer will also send the explorer a copy of the latest model
        weights."""
        # Wait for the first round of explorations to come back from explorers.
        for epoch in range(self.epochs):
            # Update memory from explorer workers
            if self.is_main:
                print(f"Starting epoch {epoch}.")
            if self.num_learners > 1:
                sampler = data.distributed.DistributedSampler(
                    self.dataset, shuffle=True
                )
            else:
                sampler = data.RandomSampler(self.dataset)
            loader = data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=0,
                sampler=sampler,
                collate_fn=dataset.collate,
            )
            for mini_epoch in range(1000):
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(mini_epoch)
                for batch in loader:
                    (
                        robot_state_batch,
                        human_state_batch,
                        action_batch,
                        target_value,
                        target_reward,
                        target_policy,
                        gradient_scale_batch,
                        weight_batch,
                    ) = batch
                    target_value_scalar = numpy.array(target_value, dtype="float32")
                    priorities = numpy.zeros_like(target_value)

                    if self.config["PER"]:
                        weight_batch = weight_batch.to(self.device)

                    robot_state_batch = robot_state_batch.to(self.device)
                    human_state_batch = human_state_batch.to(self.device)
                    action_batch = action_batch.to(self.device)
                    target_value = target_value.to(self.device)
                    target_reward = target_reward.to(self.device)
                    target_policy = target_policy.to(self.device)
                    gradient_scale_batch = gradient_scale_batch.to(self.device)

                    # observation_batch: batch, channels, height, width
                    # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
                    # target_value: batch, num_unroll_steps+1
                    # target_reward: batch, num_unroll_steps+1
                    # target_policy: batch, num_unroll_steps+1, len(action_space)
                    # gradient_scale_batch: batch, num_unroll_steps+1

                    target_value = policy.scalar_to_support(
                        target_value, self.config["support-size"]
                    )
                    target_reward = policy.scalar_to_support(
                        target_reward, self.config["support-size"]
                    )
                    # target_value: batch, num_unroll_steps+1, 2*support_size+1
                    # target_reward: batch, num_unroll_steps+1, 2*support_size+1

                    ## Generate predictions
                    (
                        value,
                        reward,
                        policy_logits,
                        hidden_state,
                    ) = distributed_utils.unwrap_ddp(self.policy).initial_inference(
                        robot_state_batch, human_state_batch
                    )
                    predictions = [(value, reward, policy_logits)]
                    for i in range(1, action_batch.shape[1]):
                        (
                            value,
                            reward,
                            policy_logits,
                            hidden_state,
                        ) = distributed_utils.unwrap_ddp(
                            self.policy
                        ).recurrent_inference(
                            hidden_state, action_batch[:, i].unsqueeze(-1)
                        )

                        # Scale the gradient at the start of the dynamics function
                        # (See paper appendix Training)
                        hidden_state.register_hook(lambda grad: grad * 0.5)
                        predictions.append((value, reward, policy_logits))
                    # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 |
                    # 2*support_size+1 | 9 (according to the 2nd dim)

                    ## Compute losses
                    value_loss, reward_loss, policy_loss = (0, 0, 0)
                    value, reward, policy_logits = predictions[0]

                    # Ignore reward loss for the first batch step
                    current_value_loss, _, current_policy_loss = self.loss_function(
                        value.squeeze(-1),
                        reward.squeeze(-1),
                        policy_logits,
                        target_value[:, 0],
                        target_reward[:, 0],
                        target_policy[:, 0],
                    )
                    value_loss += current_value_loss
                    policy_loss += current_policy_loss
                    # Compute priorities for the prioritized replay
                    # (See paper appendix Training)
                    pred_value_scalar = (
                        policy.support_to_scalar(value, self.config["support-size"])
                        .detach()
                        .cpu()
                        .numpy()
                        .squeeze()
                    )
                    priorities[:, 0] = (
                        numpy.abs(pred_value_scalar - target_value_scalar[:, 0])
                        ** self.config["PER-alpha"]
                    )

                    for i in range(1, len(predictions)):
                        value, reward, policy_logits = predictions[i]
                        (
                            current_value_loss,
                            current_reward_loss,
                            current_policy_loss,
                        ) = self.loss_function(
                            value.squeeze(-1),
                            reward.squeeze(-1),
                            policy_logits,
                            target_value[:, i],
                            target_reward[:, i],
                            target_policy[:, i],
                        )

                        # Scale gradient by the number of unroll steps
                        # (See paper appendix Training)
                        current_value_loss.register_hook(
                            lambda grad: grad / gradient_scale_batch[:, i]
                        )
                        current_reward_loss.register_hook(
                            lambda grad: grad / gradient_scale_batch[:, i]
                        )
                        current_policy_loss.register_hook(
                            lambda grad: grad / gradient_scale_batch[:, i]
                        )

                        value_loss += current_value_loss
                        reward_loss += current_reward_loss
                        policy_loss += current_policy_loss

                        # Compute priorities for the prioritized replay
                        # (See paper appendix Training)
                        pred_value_scalar = (
                            policy.support_to_scalar(value, self.config["support-size"])
                            .detach()
                            .cpu()
                            .numpy()
                            .squeeze()
                        )
                        priorities[:, i] = (
                            numpy.abs(pred_value_scalar - target_value_scalar[:, i])
                            ** self.config["PER-alpha"]
                        )

                    # Scale the value loss, paper recommends by 0.25
                    # (See paper appendix Reanalyze)
                    loss = (
                        value_loss * self.config["value-loss-weight"]
                        + reward_loss
                        + policy_loss
                    )
                    #if self.config["PER"]:
                    #    # Correct PER bias by using importance-sampling (IS) weights
                    #    loss *= weight_batch.squeeze(-1)
                    # Mean over batch dimension (pseudocode do a sum)
                    loss = loss.mean()

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.training_step += 1

                    if self.is_main:
                        self.logger.add_scalar(
                            "loss/total_loss", loss.item(), self.global_step
                        )
                        self.logger.add_scalar(
                            "loss/value_loss",
                            value_loss.mean().item(),
                            self.global_step,
                        )
                        self.logger.add_scalar(
                            "loss/reward_loss",
                            reward_loss.mean().item(),
                            self.global_step,
                        )
                        self.logger.add_scalar(
                            "loss/policy_loss",
                            policy_loss.mean().item(),
                            self.global_step,
                        )

                        log_str = f"global_step={self.global_step}. "
                        log_str += f"total_loss={loss.item():.5f}. "
                        log_str += f"value_loss={value_loss.mean().item():.5f}. "
                        log_str += f"reward_loss={reward_loss.mean().item():.5f}. "
                        log_str += f"policy_loss={policy_loss.mean().item():.5f}."
                        print(log_str)

                    self.global_step += 1

            self._get_history()
            self._send_policy()

    @staticmethod
    def loss_function(
        value, reward, policy_logits, target_value, target_reward, target_policy,
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(
            1
        )
        return value_loss, reward_loss, policy_loss
