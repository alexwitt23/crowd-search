"""An explorer class that navigates the gym environment defined in 
`third_party.crowd_sim`."""

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
        phase = "train"
        assert phase in ["train", "val", "test"]
        self.running_episode = True

        self.robot.policy.set_phase(phase)

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
