"""Explorer objects which maintain a copy of the environment and policy weights.
The weights are updated periodically based on the schedule of trainer nodes.
Every so ofte, the trainer nodes will ask for the explorers most recent episode
history to queue for training."""

import threading
import copy
import time
from typing import Dict, List

import torch
from torch.distributed import rpc

from crowd_search import policy
from third_party.crowd_sim.envs import crowd_sim
from third_party.crowd_sim.envs.utils import agent
from third_party.crowd_sim.envs.utils import info


class Explorer:
    def __init__(
        self,
        environment: crowd_sim.CrowdSim,
        robot: agent.Robot,
        robot_policy: policy.CrowdSearchPolicy,
    ) -> None:
        """Initialize Explorer.

        Args:
            environment: The simulation environment to be explored.
            robot: The robot agent to use to explore the environment.
            robot_policy: The policy that governs the robot's decisions.
            learner_idx: The numerical id of the associated trainer node.
        """
        self.id = rpc.get_worker_info().id
        # Copy to make entirly sure these are owned by the explorer process.
        self.environment = copy.deepcopy(environment)
        self.robot = copy.deepcopy(robot)
        self.target_policy = copy.deepcopy(robot_policy)
        self.history = []
        self.lock = threading.Lock()

    @torch.no_grad()
    def continuous_exploration(self):
        """Have this explorer continiously explore the environment and collect
        the results. This function is called by a trainer node."""

        while True:
            with self.lock:
                # Returns list of history tensors. This is done in two steps to prevent
                # any locking issues. (Might be unnecessary, but not harmful).
                history = self.run_episode()
                self.history += history

            time.sleep(1.0)

    def get_history_len(self):
        """Helper function to return length of the history currently held."""
        with self.lock:
            return len(self.history)

    def get_history(self):
        """Return the history. This function will bee called by trainer nodes."""
        with self.lock:
            return self.history

    def clear_history(self):
        """Clear the history. Typically called by trainer node after new history has
        been recieved."""
        with self.lock:
            self.history.clear()

    def run_episode(self):
        """Run a single episode of the crowd search game.
        
        This function is passed a remote refference to an agent with
        an actual copy of the weights."""
        print("WELL?")
        phase = "train"
        assert phase in ["train", "val", "test"]
        self.running_episode = True
        print("WELL?")
        self.robot.policy.set_phase(phase)
        self.environment.set_robot(self.robot)

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
            states.append(self.robot.policy.last_state)
            actions.append(action)
            rewards.append(reward)
            print(reward)

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

        history = self._process_epoch(states, actions, rewards)

        return history

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

    def update_models(self, models: Dict[str, torch.Tensor]) -> None:
        """Pass in a dictionary of model_state_dicts that will be used to
        update this Explorer's local copies."""
        self.robot.policy.gnn.load_state_dict(models["gnn"])
        self.robot.policy.value_estimator.load_state_dict(models["value_estimator"])
        self.robot.policy.state_estimator.load_state_dict(models["state_estimator"])
