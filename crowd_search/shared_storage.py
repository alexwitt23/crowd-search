"""This shared storage objects sits between the explorer and trainer nodes
to act as a hub for collating the relavant information between the different
groups of nodes.

We'd like for the trainer to be able to upload its new policy weights and the
explorer to upload the history of each new game. We use this object as the
intermediate step so the other nodes can continue to execute their jobs."""

import copy
from typing import Dict, List
import uuid

import torch

from crowd_search import explorer


class SharedStorage:
    """Shared storage object. Simple interface which basically acts as a
    holding spot for different data in the training pipeline."""

    def __init__(self, max_history_storage: int) -> None:
        """Initialize the obect."""
        self.history = []
        self.episode_lengths = []
        self.success = 0
        self.timeout = 0
        self.collision = 0
        self.policy = None
        self.gamma = 0.99  # TODO(alex): make configurable.
        self.policy_id = None
        self.max_history_storage = max_history_storage
        self.epoch = 0

    def update_policy(self, policy):
        """Recieve a policy object and update local copy."""
        self.policy = policy
        self.policy_id = uuid.uuid4()

    def get_policy(self):
        """Return policy object."""
        return self.policy, self.policy_id

    def upload_history(self, history: explorer.GameHistory) -> None:
        """Add history to internal list."""
        history = copy.deepcopy(history)
        discounted_rewards = []
        discounted_reward = 0
        for idx, reward in enumerate(reversed(history.reward_history)):
            if idx == 0:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        rewards = torch.Tensor(discounted_rewards)
        if len(discounted_rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Stack all the robot states and human states
        robot_states = [state[0] for state in history.observation_history]
        human_states = [state[1] for state in history.observation_history]
        actions = history.action_history
        logprobs = history.logprobs

        datas = []
        for robot, human, action, reward, ureward, logprob in zip(
            robot_states, human_states, actions, rewards, discounted_rewards, logprobs
        ):
            datas.append(
                {
                    "robot_states": robot,
                    "human_states": human,
                    "action": action,
                    "reward": reward,
                    "undiscounted_reward": torch.Tensor([ureward]),
                    "logprobs": logprob,
                }
            )
            if len(datas) > self.max_history_storage:
                break

        self.history.append(datas)

        if history.reward_history[-1] == 1.0:
            self.success += 1
        elif history.reward_history[-1] == 0.0:
            self.timeout += 1
        else:
            self.collision += 1

        self.episode_lengths.append(len(history.logprobs))

    def get_history_amount(self) -> int:
        return sum(len(h) for h in self.history)

    def get_history_capacity(self) -> int:
        return self.max_history_storage

    def get_history(self) -> List[Dict[str, torch.Tensor]]:
        """Return the internal history list."""
        if not self.episode_lengths:
            return (None, None, None, None, None)

        total = sum(self.episode_lengths)
        mean_times = total / len(self.episode_lengths)

        return (
            self.history,
            mean_times,
            self.success / len(self.episode_lengths),
            self.timeout / len(self.episode_lengths),
            self.collision / len(self.episode_lengths),
        )

    def clear_history(self):
        """Clear out the internal history. Called after history is
        returned to a trainer node."""
        self.history = []
        self.success = 0
        self.timeout = 0
        self.collision = 0
        self.episode_lengths = []

    def get_epoch(self):
        return self.epoch
    
    def set_epoch(self, epoch):
        self.epoch = epoch