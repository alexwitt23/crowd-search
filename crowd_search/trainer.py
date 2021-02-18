"""Trainer class. 

The basic idea is the class that executes the training loops. It manages a few things:

1. One or more explorer nodes. The explorers use the shared policy to explor the sim
   environment and report back their histories.

2. A copy of the policy. The policy is assumed to contain the necessary prediction methods.
   After a couple epochs, the new policy weights are sent to the explorer nodes that _this_
   trainer node controls.
"""

import pathlib
from typing import Dict, List

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
from crowd_search import policy2


class Trainer:
    def __init__(
        self,
        cfg: Dict,
        models_cfg: Dict,
        device: torch.device,
        num_learners: int,
        explorer_nodes: List[int],
        rank: int,
        run_dir: pathlib.Path,
        batch_size: int,
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
            global logger
            logger = tensorboard.SummaryWriter(run_dir)
            self.is_main = True

        # Create the policy
        self.policy = policy2.CrowdSearchPolicy(
            cfg.get("models"),
            cfg.get("robot"),
            cfg.get("human"),
            cfg.get("incentives"),
            cfg.get("action-space"),
            device,
        )
        # Wrap the models in the distributed data parallel class for most efficient
        # distributed training.
        # self.policy.gnn = parallel.DistributedDataParallel(
        #    self.policy.gnn, device_ids=[distributed.get_rank()],
        # )
        # self.policy.value_estimator = parallel.DistributedDataParallel(
        #    self.policy.value_estimator, device_ids=[distributed.get_rank()],
        # )
        # self.policy.state_estimator = parallel.DistributedDataParallel(
        #    self.policy.state_estimator, device_ids=[distributed.get_rank()],
        # )
        # self.policy.set_epsilon(0)

        # self.optimizer = torch.optim.AdamW(
        #    list(self.policy.gnn.parameters())
        #    + list(self.policy.value_estimator.parameters())
        #    + list(self.policy.state_estimator.parameters()),
        #    lr=1.0e-3,
        # )

        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.dataset = dataset.Dataset()
        self.explorer_references = []

        for explorer_node in explorer_nodes:
            explorer_info = rpc.get_worker_info(f"Explorer:{explorer_node}")
            self.explorer_references.append(
                rpc.remote(explorer_info, explorer.Explorer, args=(cfg,),)
            )

    def _collate_explorations(self):
        """Call out to all the explorer nodes and pass the policy so new
        observations can be made."""
        futures = []
        for explorer_rref in self.explorer_references:
            futures.append(explorer_rref.rpc_async().run_episode())

        explorer_histories = torch.futures.wait_all(futures)
        self.dataset.transitions.extend(explorer_histories)

        # Clear out the explorer history.
        futures = []
        for explorer_rref in self.explorer_references:
            futures.append(explorer_rref.rpc_async().clear_history())

        torch.futures.wait_all(futures)
        distributed.barrier()

    def combine_dataset(self, histories) -> None:
        """This function sends the new history data from the explorers to all training
        nodes so traing batching is efficient as possible. Since the trainer nodes are
        on their own distributed process group, we don't have to worry about specifying
        one here."""
        output = [None] * self.num_learners
        distributed.all_gather_object(output, histories)
        histories = []
        for history in output:
            histories.extend(history)

        return histories

    def broadcast_models(self) -> None:
        """Send the newly trained models out to the trainer nodes."""

        gnn = {
            name: weight.cpu()
            for name, weight in distributed_utils.unwrap_ddp(self.policy.gnn)
            .state_dict()
            .items()
        }
        state_est = {
            name: weight.cpu()
            for name, weight in distributed_utils.unwrap_ddp(
                self.policy.state_estimator
            )
            .state_dict()
            .items()
        }
        value_est = {
            name: weight.cpu()
            for name, weight in distributed_utils.unwrap_ddp(
                self.policy.value_estimator
            )
            .state_dict()
            .items()
        }
        futures = []
        for explorer_rref in self.explorer_references:
            futures.append(
                explorer_rref.rpc_async().update_models(
                    {
                        "gnn": gnn,
                        "state_estimator": state_est,
                        "value_estimator": value_est,
                    }
                )
            )

        torch.futures.wait_all(futures)

    def continous_train(self) -> None:
        """Target function to call after intialization.

        This function loops over the available training data and trains our models.
        Periodically, a trainer will look out to the explorer nodes for new data. At
        this point, our trainer will also send the explorer a copy of the latest model
        weights."""
        # self.broadcast_models()
        # Wait for the first round of explorations to come back from explorers.
        global_step = 0
        for epoch in range(self.epochs):
            # Update memory from explorer workers
            self._collate_explorations()
            print(f"Starting epoch {epoch}.")
            sampler = data.distributed.DistributedSampler(self.dataset, shuffle=True)
            loader = data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=0,
                sampler=sampler,
                collate_fn=dataset.collate
            )
            for mini_epoch in range(10):
                sampler.set_epoch(mini_epoch)
                for idx, batch in enumerate(loader):
                    (
                        observation_batch,
                        action_batch,
                        value_batch,
                        reward_batch,
                        policy_batch,
                        weight_batch,
                        gradient_scale_batch,
                    ) = batch
        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = numpy.array(target_value, dtype="float32")
        priorities = numpy.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        observation_batch = torch.tensor(observation_batch).float().to(device)
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # gradient_scale_batch: batch, num_unroll_steps+1

        target_value = policy2.scalar_to_support(target_value, self.config.support_size)
        target_reward = policy2.scalar_to_support(
            target_reward, self.config.support_size
        )
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1

        ## Generate predictions
        value, reward, policy_logits, hidden_state = self.policy.initial_inference(
            observation_batch
        )
        predictions = [(value, reward, policy_logits)]
        for i in range(1, action_batch.shape[1]):
            (
                value,
                reward,
                policy_logits,
                hidden_state,
            ) = self.policy.recurrent_inference(hidden_state, action_batch[:, i])
            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_logits))
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

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
        # Compute priorities for the prioritized replay (See paper appendix Training)
        pred_value_scalar = (
            models.support_to_scalar(value, self.config.support_size)
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )
        priorities[:, 0] = (
            numpy.abs(pred_value_scalar - target_value_scalar[:, 0])
            ** self.config.PER_alpha
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

            # Scale gradient by the number of unroll steps (See paper appendix Training)
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

            # Compute priorities for the prioritized replay (See paper appendix Training)
            pred_value_scalar = (
                models.support_to_scalar(value, self.config.support_size)
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            priorities[:, i] = (
                numpy.abs(pred_value_scalar - target_value_scalar[:, i])
                ** self.config.PER_alpha
            )

        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
        loss = value_loss * self.config.value_loss_weight + reward_loss + policy_loss
        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            loss *= weight_batch
        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
            priorities,
            # For log purpose
            loss.item(),
            value_loss.mean().item(),
            reward_loss.mean().item(),
            policy_loss.mean().item(),
        )

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
