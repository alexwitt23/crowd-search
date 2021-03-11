"""Trainer class.

The basic idea is the class that executes the training loops. It manages a few things:

1. One or more explorer nodes. The explorers use the shared policy to explor the sim
   environment and report back their histories.

2. A copy of the policy. The policy is assumed to contain the necessary prediction methods.
   After a couple epochs, the new policy weights are sent to the explorer nodes that _this_
   trainer node controls.
"""
import random
import copy
import pathlib
import tempfile
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
from crowd_search import ppo
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
        cache_dir: pathlib.Path,
    ) -> None:
        self.cfg = cfg
        self.explorer_nodes = explorer_nodes
        self.epochs = 8000
        self.device = device
        self.num_learners = num_learners
        self.batch_size = batch_size
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.eps_clip = 0.2
        self.num_episodes = 0
        self.incentives = cfg.get("incentives")
        # If main node, create a logger
        self.is_main = False
        if rank == 0:
            self.logger = tensorboard.SummaryWriter(run_dir)
            self.is_main = True

        # Create the policy
        self.policy = ppo.PPO(
            cfg.get("models"),
            cfg.get("robot"),
            cfg.get("human"),
            cfg.get("incentives"),
            device,
        )
        self.policy_old = ppo.PPO(
            cfg.get("models"),
            cfg.get("robot"),
            cfg.get("human"),
            cfg.get("incentives"),
            device,
        )
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy.to(self.device)
        self.policy_old.to(self.device)

        if num_learners > 1:
            if torch.cuda.is_available():
                self.policy = parallel.DistributedDataParallel(
                    self.policy, device_ids=[rank], find_unused_parameters=True
                )
            else:
                self.policy = parallel.DistributedDataParallel(
                    self.policy, device_ids=None, find_unused_parameters=True
                )

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=1.0e-3, weight_decay=0.0, betas=(0.9, 0.999)
        )
        self.epoch = 0
        self.training_step = 0
        self.dataset = dataset.Dataset(cache_dir)
        self.cache_dir = cache_dir
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

        self.MseLoss = nn.MSELoss()
        # Update explorers with initial policy weights
        self._send_policy()

    def _send_policy(self):
        model = copy.deepcopy(distributed_utils.unwrap_ddp(self.policy_old))
        model.to("cpu")
        self.storage_node.rpc_sync().update_policy(model)

    def _get_history(self):

        new_histories = []
        while not new_histories:
            (
                new_histories,
                mean_times,
                success,
                timeout,
                collision,
            ) = self.storage_node.rpc_sync().get_history()
            time.sleep(2.0)

        self.storage_node.rpc_sync().clear_history()

        datas = []
        for history in new_histories:
            for data in history:
                datas.append(data["undiscounted_reward"])
        average_rewards = torch.mean(torch.stack(datas))

        if self.is_main:
            self.logger.add_scalar(
                "explorer/avg_reward", average_rewards, self.global_step
            )
            self.logger.add_scalar(
                "explorer/avg_sim_time", mean_times, self.global_step
            )
            self.logger.add_scalar(
                "explorer/success_frac", success, self.global_step,
            )
            self.logger.add_scalar(
                "explorer/collision_frac", collision, self.global_step,
            )
            self.logger.add_scalar(
                "explorer/timeout_frac", timeout, self.global_step,
            )
        self._combine_datasets(new_histories)

    def _combine_datasets(self, histories: List[List[Dict[str, torch.Tensor]]]):
        """Called after new history is acquired from local storage node. We want to
        send all new history to each other trainer node."""

        if self.is_main:
            reached_goal = []
            collision = []
            timeout = []

            for idx, history in enumerate(histories):
                if (
                    self.incentives.get("success")
                    == history[-1]["undiscounted_reward"].item()
                ):
                    reached_goal.append(idx)
                elif (
                    self.incentives.get("collision")
                    == history[-1]["undiscounted_reward"].item()
                ):
                    collision.append(idx)
                else:
                    timeout.append(idx)

            if reached_goal:
                idx = random.choice(reached_goal)
                viz.plot_history(
                    histories[idx], self.run_dir / f"plots/{self.global_step}_s.gif"
                )
            if timeout:
                idx = random.choice(timeout)
                viz.plot_history(
                    histories[idx], self.run_dir / f"plots/{self.global_step}_t.gif",
                )
            if collision:
                idx = random.choice(collision)
                viz.plot_history(
                    histories[idx], self.run_dir / f"plots/{self.global_step}_c.gif",
                )

        self.dataset.update(histories, self.epoch)
        if distributed.is_initialized():
            distributed.barrier()

    def get_items(self):
        self.dataset.prepare_for_epoch()
        if distributed.is_initialized():
            distributed.barrier()

    def continous_train(self) -> None:
        """Target function to call after intialization.

        This function loops over the available training data and trains our models.
        Periodically, a trainer will look out to the explorer nodes for new data. At
        this point, our trainer will also send the explorer a copy of the latest model
        weights."""
        # Wait for the first round of explorations to come back from explorers.
        for epoch in range(self.epochs):
            self.epoch = epoch
            self._get_history()
            # Update memory from explorer workers
            if self.is_main:
                print(f"Starting epoch {epoch}.")

            self.get_items()

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
            for mini_epoch in range(10):
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(mini_epoch)
                for batch in loader:
                    (
                        robot_state_batch,
                        human_state_batch,
                        action_batch,
                        target_reward,
                        logprobs_batch,
                    ) = batch
                    robot_state_batch = robot_state_batch.to(self.device).transpose(
                        1, 2
                    )
                    human_state_batch = human_state_batch.to(self.device).transpose(
                        1, 2
                    )
                    action_batch = action_batch.to(self.device)
                    target_reward = target_reward.to(self.device).squeeze(-1)
                    logprobs_batch = logprobs_batch.to(self.device).squeeze(-1)

                    logprobs, state_values, dist_entropy = distributed_utils.unwrap_ddp(
                        self.policy
                    ).evaluate(robot_state_batch, human_state_batch, action_batch)

                    # Finding the ratio (pi_theta / pi_theta__old):
                    ratios = torch.exp(logprobs - logprobs_batch.detach())

                    # Finding Surrogate Loss:
                    advantages = target_reward - state_values.detach()
                    surr1 = ratios * advantages
                    surr2 = (
                        torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                        * advantages
                    )
                    loss = (
                        -torch.min(surr1, surr2)
                        + 0.5 * self.MseLoss(state_values, target_reward)
                        - 0.01 * dist_entropy
                    )
                    loss = loss.mean()

                    if torch.isnan(robot_state_batch).any():
                        raise ValueError("robot_state_batch is nan")
                    if torch.isnan(human_state_batch).any():
                        raise ValueError("human_state_batch is nan")
                    if torch.isnan(action_batch).any():
                        raise ValueError("action_batch is nan")
                    if torch.isnan(target_reward).any():
                        raise ValueError("target_reward is nan")
                    if torch.isnan(logprobs_batch).any():
                        raise ValueError("logprobs_batch is nan")
                    if torch.isnan(logprobs).any():
                        raise ValueError("logprobs is nan")
                    if torch.isnan(state_values).any():
                        raise ValueError("state_values is nan")
                    if torch.isnan(dist_entropy).any():
                        raise ValueError("dist_entropy is nan")
                    if torch.isnan(loss) or torch.isinf(loss):
                        raise ValueError("Loss blew up.")

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.training_step += 1

                    if self.is_main:
                        self.logger.add_scalar(
                            "loss/total_loss", loss.item(), self.global_step
                        )

                        log_str = f"global_step={self.global_step}. "
                        log_str += f"total_loss={loss.item():.5f}. "
                        print(log_str)

                    self.global_step += 1

            self.policy_old.load_state_dict(
                distributed_utils.unwrap_ddp(self.policy).state_dict()
            )
            self._send_policy()
