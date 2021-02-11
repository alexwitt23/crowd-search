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

from crowd_search import explorer
from crowd_search import models
from crowd_search import distributed_utils

import asyncio
import copy
import time
from typing import List

import gym
import torch
import tqdm
from torch import distributed
from torch.distributed import rpc
from torch import multiprocessing

from crowd_search import replay_buffer
from crowd_search import config
from crowd_search import policy
from crowd_search import shared_storage
from crowd_search import distributed_utils
from third_party.crowd_sim.envs import crowd_sim
from third_party.crowd_sim.envs.utils import agent
from third_party.crowd_sim.envs.utils import info


class Explorer:
    def __init__(
        self,
        environment: crowd_sim.CrowdSim,
        robot: agent.Robot,
        robot_policy: policy.CrowdSearchPolicy,
        gamma: float,
        learner_idx: int,
    ) -> None:
        self.id = rpc.get_worker_info().id
        self.learner_idx = learner_idx
        self.environment = copy.deepcopy(environment)
        self.robot = copy.deepcopy(robot)
        self.gamma = gamma
        self.training_step = 10000
        self.target_policy = robot_policy

    def run_episode(self, agent_reference):
        """Run a single episode of the crowd search game.
        
        This function is passed a remote refference to an agent with
        an actual copy of the weights."""

        # Keep track of the various social aspects of the robot's navigation
        success = collision = timeout = discomfort = 0
        cumulative_rewards = []
        average_returns = []
        success_times = []
        collision_cases = []
        collision_times = []
        timeout_cases = []
        timeout_times = []

        # Reset the environment at the beginning of each episode.
        observation = self.environment.reset(phase)

        states, actions, rewards = [], [], []

        goal_reached = False
        while not goal_reached:

            action = self.robot.act(observation)
            observation, reward, goal_reached, socal_info = self.environment.step(
                action
            )
            states.append(self.environment.robot.policy.last_state)
            actions.append(action)
            rewards.append(reward)

            # Check if robot exhibited discomforting behavior.
            if isinstance(socal_info, info.Discomfort):
                discomfort += 1

        if isinstance(socal_info, info.ReachGoal):
            success += 1
            success_times.append(self.environment.global_time)
        elif isinstance(socal_info, info.Collision):
            collision += 1
            collision_times.append(self.environment.global_time)
        elif isinstance(socal_info, info.Timeout):
            timeout += 1
            timeout_times.append(self.environment.time_limit)

        return self._process_epoch(states, actions, rewards)

    def _process_epoch(self, states: List, actions: List, rewards: List) -> None:
        """Extract the transition states from the episode."""
        history = []
        for idx, state in enumerate(states[:-1]):
            reward = rewards[idx]
            # define the value of states in IL as cumulative discounted rewards, which is the same in RL
            state = self.target_policy.transform(state)
            next_state = self.target_policy.transform(states[idx + 1])
            next_state = states[idx + 1]
            if idx == len(states) - 1:
                # terminal state
                value = reward
            else:
                value = 0
            value = torch.Tensor([value])
            reward = torch.Tensor([rewards[idx]])

            history.append(
                {
                    "robot_state": state[0],
                    "humans_state": state[1],
                    "value": value,
                    "reward": reward,
                    "next_robot_state": next_state[0],
                    "next_humans_state": next_state[1],
                }
            )

        return history


class Dataset(data.Dataset):
    def __init__(self):
        self.transitions = []

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
        models_cfg: Dict,
        device: torch.device,
        num_explorers: int,
        num_learners: int,
        explorer_nodes: List[int],
    ):
        self.num_explorers = num_explorers
        self.explorer_nodes = explorer_nodes
        self.epochs = 40
        self.device = device
        self.agent_rref = rpc.RRef(self)

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

        for explorer_node in explorer_nodes:
            explorer_info = rpc.get_worker_info(f"Explorer:{explorer_node}")
            self.explorer_references.append(rpc.remote(explorer_info, Explorer))
            self.episode_history[explorer_info.id] = []

    def run_episode(self):
        futures = []
        for explorer_rref in self.explorer_references:

            futures.append(
                explorer_rref.rpc_sync().run_episode(explorer_rref, self.agent_rref)
            )

        # wait until all obervers have finished this episode
        for future in futures:
            future.wait()

    def continous_train(self) -> None:
        # Send the same model out to all explorer nodes
        self._broadcast_models()

        for epoch in range(self.epochs):

            # Update memory from explorer workers
            self._collate_explorations(self.process_group)

            loader = data.DataLoader(
                self.dataset,
                batch_size=20,
                pin_memory=True,
                collate_fn=_collate_fn,
                shuffle=True,
            )

            for batch in loader:
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
                    f"epoch={epoch}. total_loss={total_loss.item():.4f}. rewards={rewards.mean().item():.4f}"
                )

            # After epoch, send weights out to explorer nodes:
            self._broadcast_models()

    def process_batch(self, batch):
        (
            robot_states,
            human_states,
            rewards,
            next_robot_states,
            next_human_states,
        ) = batch

        robot_states = torch.stack(robot_states).squeeze(1)
        human_states = torch.stack(human_states).squeeze(1)
        rewards = torch.stack(rewards)
        next_robot_states = torch.stack(next_robot_states).squeeze(1)
        next_human_states = torch.stack(next_human_states).squeeze(1)

        # if use_cuda:
        #    robot_states = robot_states.cuda()
        #    human_states = human_states.cuda()
        #    rewards = rewards.cuda()
        #    next_robot_states = next_robot_states.cuda()
        #    next_human_states = next_human_states.cuda()

        robot_state, human_state = self.gnn(robot_states, human_states)
        outputs = self.value_estimator(robot_state)

        gamma_bar = pow(0.9, 0.25 * 1.0)

        next_robot_states, next_human_state = self.gnn(
            next_robot_states.transpose(1, 2), next_human_states.transpose(1, 2)
        )
        next_value = self.value_estimator(next_robot_states)
        target_values = rewards.unsqueeze(-1) + gamma_bar * next_value
        value_loss = self.l1(outputs, target_values)

        _, next_human_states_est = self.state_estimator(
            robot_states, human_state, None,
        )
        state_loss = self.l2(
            next_human_states_est.transpose(1, 2), next_human_states.squeeze(1)
        )
        total_loss = value_loss + state_loss

        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        print(f"loss={total_loss.item():.5f}, rewards={torch.mean(rewards).item():.5f}")
