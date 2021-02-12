import copy
import time
import random
from typing import Dict, List

import torch
from torch import nn
from torch import distributed
from torch.distributed import rpc
from torch.utils import data
from torch.nn import parallel

from crowd_search import models
from crowd_search import distributed_utils


import gym
import torch
import tqdm
from torch import distributed
from torch.distributed import rpc
from torch import multiprocessing

from crowd_search import explorer
from crowd_search import config
from crowd_search import policy
from crowd_search import distributed_utils
from third_party.crowd_sim.envs.utils import agent


class Dataset(data.Dataset):
    def __init__(self):
        self.transitions = []
        self.trainable_length = 0

    def __len__(self):
        return self.trainable_length

    def __getitem__(self, idx):
        return self.transitions[idx]


def _collate_fn(data_batch):
    robot_state = []
    humans_state = []
    value = []
    reward = []
    next_robot_state = []
    next_humans_state = []
    for data in data_batch:
        robot_state.append(data["robot_state"])
        humans_state.append(data["humans_state"])
        value.append(data["value"])
        reward.append(data["reward"])
        next_robot_state.append(data["next_robot_state"])
        next_humans_state.append(data["next_humans_state"])

    robot_state = torch.stack(robot_state).squeeze(1)
    humans_state = torch.stack(humans_state).squeeze(1)
    value = torch.stack(value)
    reward = torch.stack(reward)
    next_robot_state = torch.stack(next_robot_state).squeeze(1).transpose(-2, -1)
    next_humans_state = torch.stack(next_humans_state).squeeze(1).transpose(-2, -1)

    return (
        robot_state,
        humans_state,
        value,
        reward,
        next_robot_state,
        next_humans_state,
    )


class Trainer:
    def __init__(
        self,
        cfg: Dict,
        models_cfg: Dict,
        device: torch.device,
        num_explorers: int,
        num_learners: int,
        explorer_nodes: List[int],
    ) -> None:
        self.cfg = cfg
        self.num_explorers = num_explorers
        self.explorer_nodes = explorer_nodes
        self.epochs = 40
        self.device = device

        # Create models
        gnn_cfg = models_cfg.get("gnn")
        self.gnn = models.GNN(
            gnn_cfg.get("robot-state-dimension"),
            gnn_cfg.get("human-state-dimension"),
            attn_layer_channels=gnn_cfg.get("mlp-layer-channels"),
            num_attention_layers=gnn_cfg.get("gnn-layers"),
        )
        self.gnn.to(device=device)
        self.gnn.train()
        self.gnn = parallel.DistributedDataParallel(
            self.gnn, device_ids=[distributed.get_rank()],
        )

        value_net_cfg = models_cfg.get("value-net")
        self.value_estimator = models.ValueNet(
            value_net_cfg.get("channel-depth"), value_net_cfg.get("net-dims")
        )
        self.value_estimator.to(self.device)
        self.value_estimator.train()
        self.value_estimator = parallel.DistributedDataParallel(
            self.value_estimator, device_ids=[distributed.get_rank()],
        )

        state_net_cfg = models_cfg.get("state-net")
        self.state_estimator = models.StateNet(
            state_net_cfg.get("time-step"), state_net_cfg.get("channel-depth")
        )
        self.state_estimator.to(device)
        self.state_estimator.train()
        self.state_estimator = parallel.DistributedDataParallel(
            self.state_estimator, device_ids=[distributed.get_rank()],
        )

        self.optimizer = torch.optim.AdamW(
            list(self.gnn.parameters())
            + list(self.value_estimator.parameters())
            + list(self.state_estimator.parameters()),
            lr=1.0e-3,
        )

        self.l2 = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()

        self.dataset = Dataset()

        self.explorer_references = []
        self.episode_history = {}

        # Get remote references to the associated explorer nodes. Create each
        # explorer's necessary info.
        environment = gym.make(
            "CrowdSim-v0",
            env_cfg=cfg.get("sim-environment"),
            incentive_cfg=cfg.get("incentives"),
        )
        robot_policy = policy.CrowdSearchPolicy(
            cfg.get("models"),
            cfg.get("robot"),
            cfg.get("human"),
            cfg.get("incentives"),
            "cpu",
        )
        sim_robot = agent.Robot(config.BaseEnvConfig(), "robot")
        sim_robot.time_step = environment.time_step
        sim_robot.set_policy(robot_policy)
        environment.configure(config.BaseEnvConfig())
        environment.set_robot(sim_robot)

        for explorer_node in explorer_nodes:
            explorer_info = rpc.get_worker_info(f"Explorer:{explorer_node}")
            self.episode_history[explorer_info.id] = []
            self.explorer_references.append(
                rpc.remote(
                    explorer_info,
                    explorer.Explorer,
                    args=(
                        environment,
                        sim_robot,
                        robot_policy,
                    ),
                )
            )
        # Send the explorers off to continuously explore.
        for explorer_rref in self.explorer_references:
            explorer_rref.rpc_async().continuous_exploration()

    def _broadcast_models(self) -> None:
        """Send the model weights to the explorer nodes."""
        futures = []
        gnn_state = {
            name: tensor.cpu()
            for name, tensor in distributed_utils.unwrap_ddp(self.gnn)
            .state_dict()
            .items()
        }
        value_state = {
            name: tensor.cpu()
            for name, tensor in distributed_utils.unwrap_ddp(self.value_estimator)
            .state_dict()
            .items()
        }
        state_state = {
            name: tensor.cpu()
            for name, tensor in distributed_utils.unwrap_ddp(self.state_estimator)
            .state_dict()
            .items()
        }
        for explorer_rref in self.explorer_references:
            futures.append(
                explorer_rref.rpc_async().update_models(
                    {
                        "gnn": gnn_state,
                        "value_estimator": value_state,
                        "state_estimator": state_state,
                    }
                )
            )

        torch.futures.wait_all(futures)
        distributed.barrier()

    def _collate_explorations(self):
        """Call out to the explorer nodes and ask for their histories."""
        # See if any nodes have history.
        num_histories = [0 for _ in range(len(self.explorer_references))]

        while sum(num_histories) < 100:
            futures = [
                explorer_rref.rpc_async().get_history_len()
                for explorer_rref in self.explorer_references
            ]
            num_histories = torch.futures.wait_all(futures)
            time.sleep(5.0)
            print(num_histories)

        history_futures = []
        for num_history, explorer_rref in zip(futures, self.explorer_references):
            if num_history:
                history_futures.append(explorer_rref.rpc_async().get_history())

        histories = torch.futures.wait_all(history_futures)
        for history in histories:
            self.dataset.transitions.extend(history)
        self.dataset.trainable_length += 100
        print(len(self.dataset.transitions))

        clear_histories = []
        for num_history, explorer_rref in zip(futures, self.explorer_references):
            if num_history:
                clear_histories.append(explorer_rref.rpc_async().clear_history())

        torch.futures.wait_all(clear_histories)
        distributed.barrier()

    def continous_train(self) -> None:
        """Target function to call after intialization.

        This function loops over the available training data and trains our models.
        Periodically, a trainer will look out to the explorer nodes for new data. At
        this point, our trainer will also send the explorer a copy of the latest model
        weights."""

        # First, send out this trainer's copy of the weights to the explorer nodes.
        self._broadcast_models()

        # Wait for the first round of explorations to come back from explorers.

        for epoch in range(self.epochs):

            # Update memory from explorer workers
            self._collate_explorations()

            loader = data.DataLoader(
                self.dataset,
                batch_size=20,
                pin_memory=True,
                collate_fn=_collate_fn,
                shuffle=True,
            )

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
                robot_state, human_state = self.gnn(robot_states, human_states)
                outputs = self.value_estimator(robot_state)

                gamma_bar = pow(0.9, 0.25 * 1.0)

                next_robot_states, next_human_state = self.gnn(
                    next_robot_states, next_human_states
                )
                next_value = self.value_estimator(next_robot_states)
                target_values = rewards.unsqueeze(-1) + gamma_bar * next_value
                value_loss = self.l1(outputs, target_values)

                _, next_human_states_est = self.state_estimator(
                    robot_states, human_state, None,
                )

                state_loss = self.l2(next_human_states_est, next_human_states)
                total_loss = value_loss + state_loss

                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(
                    f"{self.device}. idx={idx}. epoch={epoch}. total_loss={total_loss.item():.4f}. rewards={rewards.mean().item():.4f}"
                )

            self._broadcast_models()
