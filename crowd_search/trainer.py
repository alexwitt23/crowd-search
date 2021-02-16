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
from torch.utils import tensorboard
from torch.utils import data
from torch.nn import parallel

from crowd_search import dataset
from crowd_search import distributed_utils
from crowd_search import explorer
from crowd_search import policy


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
        batch_size: int
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
        self.policy = policy.CrowdSearchPolicy(
            cfg.get("models"),
            cfg.get("robot"),
            cfg.get("human"),
            cfg.get("incentives"),
            cfg.get("action-space"),
            device,
        )
        # Wrap the models in the distributed data parallel class for most efficient
        # distributed training.
        self.policy.gnn = parallel.DistributedDataParallel(
            self.policy.gnn, device_ids=[distributed.get_rank()],
        )
        self.policy.value_estimator = parallel.DistributedDataParallel(
            self.policy.value_estimator, device_ids=[distributed.get_rank()],
        )
        self.policy.state_estimator = parallel.DistributedDataParallel(
            self.policy.state_estimator, device_ids=[distributed.get_rank()],
        )
        self.policy.set_epsilon(0)

        self.optimizer = torch.optim.AdamW(
            list(self.policy.gnn.parameters())
            + list(self.policy.value_estimator.parameters())
            + list(self.policy.state_estimator.parameters()),
            lr=1.0e-3,
        )

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
        histories = []
        for history in explorer_histories:
            histories.extend(history)

        histories = self.combine_dataset(histories)
        self.dataset.transitions.extend(histories)

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
        self.broadcast_models()
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
                collate_fn=dataset.collate_fn,
                num_workers=0,
                sampler=sampler,
                drop_last=True,
            )
            for _ in range(10):
                for idx, batch in enumerate(loader):
                    (
                        robot_states,
                        human_states,
                        values,
                        rewards,
                        next_robot_states,
                        next_human_states,
                    ) = batch

                    robot_states = robot_states.to(self.device)
                    human_states = human_states.to(self.device)
                    rewards = rewards.to(self.device)
                    next_robot_states = next_robot_states.to(self.device)
                    next_human_states = next_human_states.to(self.device)
                    robot_state, human_state = self.policy.gnn(
                        robot_states, human_states
                    )
                    outputs = self.policy.value_estimator(robot_state)

                    gamma_bar = pow(0.9, 0.25 * 1.0)

                    next_robot_states, next_human_state = self.policy.gnn(
                        next_robot_states, next_human_states
                    )
                    next_value = self.policy.value_estimator(next_robot_states)
                    target_values = rewards.unsqueeze(-1) + gamma_bar * next_value
                    value_loss = self.l2(outputs, target_values) * 1000

                    _, next_human_states_est = self.policy.state_estimator(
                        robot_states, human_state, None,
                    )

                    state_loss = self.l2(next_human_states_est, next_human_states)
                    total_loss =  value_loss + state_loss

                    total_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.is_main:
                        logger.add_scalar("state_loss", state_loss.item(), global_step)
                        logger.add_scalar("value_loss", value_loss.item(), global_step)
                        logger.add_scalar("total_loss", total_loss.item(), global_step)
                        logger.add_scalar(
                            "avg_reward", rewards.mean().item(), global_step
                        )
                        log_str = f"epoch={epoch}. total_loss={total_loss.item():.4f}."
                        log_str += f" value_loss={value_loss.item():.4f}. state_loss={state_loss.item():.4f}"
                        log_str += f" rewards={rewards.mean().item():.4f}"
                        print(log_str)

                    global_step += 1

            # After the epochs, send the new model weights out to the training nodes.
            self.broadcast_models()
