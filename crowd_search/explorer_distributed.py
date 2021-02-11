"""An explorer class that navigates the gym environment defined in 
`third_party.crowd_sim`."""

import copy
import time
from typing import List

import gym
import torch
import tqdm
from torch import distributed
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
        process_group: distributed.ProcessGroupGloo,
        learner_idx: int,
    ) -> None:
        self.learner_idx = learner_idx
        self.process_group = process_group
        self.environment = environment
        self.robot = copy.deepcopy(robot)
        self.gamma = gamma
        self.training_step = 10000
        self.target_policy = robot_policy

    def get_model_update(self):
        # TODO(alex): Make this a dictionary
        models = [None, None, None]
        models = distributed_utils.broadcast_models(
            models, self.process_group, self.learner_idx
        )
        [model.cpu() for model in models]
        self.target_policy.gnn = models[0]
        self.target_policy.value_estimator = models[1]
        self.target_policy.state_estimator = models[2]

    def continuous_play(self) -> None:
        """Run episodes continuously."""
        while True:
            self.get_model_update()
            # Run episode
            episode_history = self.run_episode(phase="train")
            distributed_utils.collate_explorations(
                episode_history, None, self.process_group, self.learner_idx
            )

    @torch.no_grad()
    def run_episode(self, phase: str):
        """Run a single episode of the crowd search game."""
        assert phase in ["train", "val", "test"]

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

            action = self.environment.robot.act(observation)
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
        """
        # Update the replay memory if the run with sucessful or not sucessful
        self._update_memory(states, actions, rewards)
        
        cumulative_rewards.append(
            sum(
                [
                    pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                    * reward
                    for t, reward in enumerate(rewards)
                ]
            )
        )
        returns = []
        for step in range(len(rewards)):
            step_return = sum(
                [
                    pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                    * reward
                    for t, reward in enumerate(rewards[step:])
                ]
            )
            returns.append(step_return)
        average_returns.append(sum(returns) / len(returns))

        success_rate = success / num_episodes
        collision_rate = collision / num_episodes
        timeout_rate = timeout / num_episodes
        assert success + collision + timeout == num_episodes
        avg_nav_time = (
            sum(success_times) / len(success_times)
            if success_times
            else self.environment.time_limit
        )
        """
        return self._process_epoch(states, actions, rewards)
        """
        print(
            {
                "success_rate": success_rate,
                "discomfort_rate": discomfort / num_episodes,
                "collision_rate": collision_rate,
                "timeout_rate": timeout_rate,
                "avg_name_time": avg_nav_time,
                "cumulative_rewards": sum(cumulative_rewards) / len(cumulative_rewards),
                "average_returns": sum(average_returns) / len(average_returns),
            }
        )
        """

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
