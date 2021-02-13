import copy
import time
import random
from typing import Dict, List

import gym
import torch
from torch import nn
from torch import distributed
from torch.distributed import rpc
from torch.utils import data
from torch.nn import parallel

from crowd_search import agents
from crowd_search import config
from crowd_search import distributed_utils
from crowd_search import explorer
from crowd_search import models
from crowd_search import policy
from third_party.crowd_sim.envs.utils import agent


class Dataset(data.Dataset):
    def __init__(self):
        self.transitions = []
        self.trainable_length = 0

    def __len__(self):
        return len(self.transitions)

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
        explorer_nodes: List[int],
    ) -> None:
        self.cfg = cfg
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

        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.dataset = Dataset()
        self.explorer_references = []

        # Get remote references to the associated explorer nodes. Create each
        # explorer's necessary info.
        environment = gym.make(
            "CrowdSim-v1",
            env_cfg=cfg.get("sim-environment"),
            incentive_cfg=cfg.get("incentives"),
            motion_planner_cfg=cfg.get("human-motion-planner"),
            human_cfg=cfg.get("human"),
            robot_cfg=cfg.get("robot"),
        )

        self.robot_policy = policy.CrowdSearchPolicy(
            cfg.get("models"),
            cfg.get("robot"),
            cfg.get("human"),
            cfg.get("incentives"),
            "cpu",
        )
        self.policy_rref = rpc.RRef(self.robot_policy)
        sim_robot = agents.Robot(cfg.get("robot"))
        # sim_robot.set_policy(robot_policy)

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

        histories = torch.futures.wait_all(futures)
        
        for history in histories:
            self.dataset.transitions.extend(copy.deepcopy(history))

        distributed.barrier()


    def continous_train(self) -> None:
        """Target function to call after intialization.

        This function loops over the available training data and trains our models.
        Periodically, a trainer will look out to the explorer nodes for new data. At
        this point, our trainer will also send the explorer a copy of the latest model
        weights."""

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
                num_workers=0,
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
