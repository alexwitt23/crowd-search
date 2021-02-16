"""Explorer objects which maintain a copy of the environment."""

from typing import Dict, List

import gym
import torch
from torch.distributed import rpc

from crowd_search import agents
from crowd_search import policy


class Explorer:
    def __init__(self, cfg) -> None:
        """Initialize Explorer.

        Args:
            environment: The simulation environment to be explored.
            robot: The robot agent to use to explore the environment.
            robot_policy: The policy that governs the robot's decisions.
            learner_idx: The numerical id of the associated trainer node.
        """
        self.id = rpc.get_worker_info().id
        # Copy to make entirly sure these are owned by the explorer process.
        self.environment = gym.make(
            "CrowdSim-v1",
            env_cfg=cfg.get("sim-environment"),
            incentive_cfg=cfg.get("incentives"),
            motion_planner_cfg=cfg.get("human-motion-planner"),
            human_cfg=cfg.get("human"),
            robot_cfg=cfg.get("robot"),
        )

        self.robot = agents.Robot(cfg.get("robot"))
        self.policy = policy.CrowdSearchPolicy(
            cfg.get("models"),
            cfg.get("robot"),
            cfg.get("human"),
            cfg.get("incentives"),
            cfg.get("action-space"),
            "cpu",
        )
        self.history = []

    def get_history_len(self):
        """Helper function to return length of the history currently held."""
        return len(self.history)

    def get_history(self):
        """Return the history. This function will bee called by trainer nodes."""
        return self.history

    def clear_history(self):
        """Clear the history. Typically called by trainer node after new history has
        been recieved."""
        self.history.clear()

    @torch.no_grad()
    def run_episode(self):
        """Run a single episode of the crowd search game.
        
        This function is passed a remote refference to an agent with
        an actual copy of the weights."""
        phase = "train"
        assert phase in ["train", "val", "test"]
        self.running_episode = True
        # self.robot.policy.set_phase(phase)

        # Keep track of the various social aspects of the robot's navigation
        # success = collision = timeout = discomfort = 0
        # cumulative_rewards = []
        # average_returns = []
        # success_times = []
        # collision_cases = []
        # collision_times = []
        # timeout_cases = []
        # timeout_times = []

        # Reset the environment at the beginning of each episode.
        observation = self.environment.reset(phase)

        states, actions, rewards = [], [], []
        goal_reached = False
        while not goal_reached:
            robot_state = self.environment.robot.get_full_state()

            action = self.policy.predict(robot_state, observation)
            observation, reward, goal_reached, socal_info = self.environment.step(
                action
            )
            states.append((robot_state, observation))
            actions.append(action)
            rewards.append(reward)

        history = self._process_epoch(states, actions, rewards)

        return history

    def _process_epoch(self, states: List, actions: List, rewards: List) -> None:
        """Extract the transition states from the episode."""
        history = []
        for idx, state in enumerate(states[:-1]):
            reward = rewards[idx]
            # define the value of states in IL as cumulative discounted rewards, which is the same in RL
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
        self.policy.gnn.load_state_dict(models["gnn"])
        self.policy.gnn.eval()
        self.policy.value_estimator.load_state_dict(models["value_estimator"])
        self.policy.value_estimator.eval()
        self.policy.state_estimator.load_state_dict(models["state_estimator"])
        self.policy.state_estimator.eval()
        print("updated models")
        self.policy.epsilon *= 0.8
