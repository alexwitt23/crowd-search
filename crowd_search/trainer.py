"""Trainer class.

The basic idea is the class that executes the training loops. It manages a few things:

1. One or more explorer nodes. The explorers use the shared policy to explor the sim
   environment and report back their histories.

2. A copy of the policy. The policy is assumed to contain the necessary prediction methods.
   After a couple epochs, the new policy weights are sent to the explorer nodes that _this_
   trainer node controls.

NOTE: Ideally this trainer script supports any policy, right now it is tied to PPO.
TODO(alex): Fix the note above.
"""
import random
import copy
import pathlib
import time
from typing import Dict, List

import gym
import torch
from torch import nn
from torch import distributed
from torch.distributed import rpc
from torch.nn import parallel
from torch.utils import tensorboard
from torch.utils import data
import yaml

from crowd_search import dataset
from crowd_search import distributed_utils
from crowd_search import explorer
from crowd_search import shared_storage
from crowd_search.policies import policy_factory
from crowd_search.visualization import viz


class Trainer:
    """The trainer class."""
    def __init__(
        self,
        cfg: Dict,
        device: torch.device,
        num_learners: int,
        explorer_nodes: List[int],
        rank: int,
        run_dir: pathlib.Path,
        storage_node: int,
        cache_dir: pathlib.Path,
    ) -> None:
        """
        Args:
            cfg: The configuration dictionary read in from the supplied .yaml file.
            device: The device to use for this trainer.
            num_learners: How many other learner (a.k.a.) trainer nodes there are.
            explorer_nodes: A list of the node ranks of the explorer nodes that are
                tied to this explorer node.
            rank: The global rank of this trainer node.
            run_dir: Where to write results during training.
            storage_node: The node rank of the associated storage node.
            cache_dir: Where to save intermediate results to. This is for the dataset.
        """
        self.cfg = cfg
        self.explorer_nodes = explorer_nodes
        self.device = device
        self.num_learners = num_learners
        self.run_dir = run_dir
        self.num_episodes = 0
        self.incentives = cfg.get("incentives")
        train_cfg = cfg.get("training")
        self.batch_size = train_cfg.get("batch-size")
        self.epochs = train_cfg.get("num-epochs")
        # If main node, create a logger
        self.is_main = False
        if rank == 0:
            self.logger = tensorboard.SummaryWriter(run_dir)
            self.is_main = True
        self.rank = rank
        # Create the policy
        environment = gym.make(
            "CrowdSim-v1",
            env_cfg=cfg.get("sim-environment"),
            incentive_cfg=cfg.get("incentives"),
            motion_planner_cfg=cfg.get("human-motion-planner"),
            human_cfg=cfg.get("human"),
            robot_cfg=cfg.get("robot"),
        )
        kwargs = {"device": self.device, "action_space": environment.action_space}

        if self.is_main:
            print(f">> Using policy: {cfg.get('policy').get('type')}.")

        self.policy = policy_factory.make_policy(cfg.get("policy"), kwargs=kwargs)
        self.policy_old = policy_factory.make_policy(cfg.get("policy"), kwargs=kwargs)
        if self.is_main:
            print(f">> Model: {self.policy}")
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy.to(self.device)
        self.policy_old.to(self.device)
        self.visualize_results = train_cfg.get("visualize-results")

        if num_learners > 1:
            if torch.cuda.is_available():
                self.policy = parallel.DistributedDataParallel(
                    self.policy, device_ids=[rank], find_unused_parameters=True
                )
            else:
                self.policy = parallel.DistributedDataParallel(
                    self.policy, device_ids=None, find_unused_parameters=True
                )

        optimizer_cfg = train_cfg.get("optimizer")
        optimizer_kwargs = optimizer_cfg.get("kwargs")
        self.optimizer = torch.optim.Adam(self.policy.parameters(), **optimizer_kwargs)

        self.epoch = 0
        self.training_step = 0
        self.dataset = dataset.Dataset(
            cache_dir, policy_factory.get_policy(cfg.get("policy").get("type"))
        )
        self.cache_dir = cache_dir
        self.explorer_references = []

        storage_info = rpc.get_worker_info(f"Storage:{storage_node}")
        self.storage_node = rpc.remote(
            storage_info,
            shared_storage.SharedStorage,
            args=(
                train_cfg.get("max-storage-memory"),
                policy_factory.get_policy(cfg.get("policy").get("type")),
            ),
        )

        for explorer_node in explorer_nodes:
            explorer_info = rpc.get_worker_info(f"Explorer:{explorer_node}")
            self.explorer_references.append(
                rpc.remote(
                    explorer_info, explorer.Explorer, args=(cfg, self.storage_node)
                )
            )
        self.best_success = 0.0
        self.best_metrics = self.run_dir / "best-metrics.yaml"
        self.global_step = 0

        self.mse_loss = nn.MSELoss()
        # Update explorers with initial policy weights
        self._send_policy()
        self.continous_train()

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

            if success > self.best_success:
                self.best_success = success
                torch.save(
                    distributed_utils.unwrap_ddp(self.policy_old).state_dict(),
                    self.run_dir / "best_success.pt",
                )
                self.best_metrics.write_text(yaml.dump({"success-rate": success}))


        self._combine_datasets(new_histories)
        if distributed.is_initialized():
            distributed.barrier()

    def _combine_datasets(self, histories: List[List[Dict[str, torch.Tensor]]]):
        """Called after new history is acquired from local storage node. We want to
        send all new history to each other trainer node."""

        if self.is_main and self.visualize_results:
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

        self.dataset.update(histories, self.epoch, self.is_main)
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

            while self.storage_node.rpc_sync().get_num_episodes() < 10:
                time.sleep(5.0)

            self._get_history()
            self.storage_node.rpc_sync().set_epoch(epoch + 1)
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
                collate_fn=self.dataset.collate,
                drop_last=True
            )
            for mini_epoch in range(5):
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(mini_epoch)
                for batch in loader:
                    loss = self.policy.process_batch(batch)

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

        if self.is_main:
            print("Training complete.")
