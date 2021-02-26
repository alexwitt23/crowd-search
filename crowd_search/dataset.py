"""A dataset that will hold the explored history. There isn't much to this dataset,
really, it might be removed in the future. The key part is the collation function
which allows us to batch together the different collections of input data."""

import copy
from typing import List, Tuple

import numpy
import torch
from torch.utils import data

from crowd_search import explorer


class Dataset(data.Dataset):
    """Dataset class."""

    def __init__(self, mu_zero_cfg):
        super().__init__()

        self.config = mu_zero_cfg
        self.gamma = 0.99
        self.robot_states = []
        self.human_states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.num_examples = 20000

    def __len__(self) -> int:
        """Return length of the dataset."""

        return len(self.robot_states)

    def __getitem__(self, idx: int):
        """Retrieve an item from the dataset and prepare it for training."""
        # Retrieve the game from the list of available games.

        return {
            "robot_states": self.robot_states[idx],
            "human_states": self.human_states[idx],
            "action": float(self.actions[idx]),
            "reward": self.rewards[idx].item(),
            "logprobs": self.logprobs[idx].item(),
        }

    def update(self, game_history: explorer.GameHistory):
        game_history = copy.deepcopy(game_history)
        discounted_rewards = []
        discounted_reward = 0
        for idx, reward in enumerate(reversed(game_history.reward_history)):
            if idx == 0:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        rewards = torch.tensor(discounted_rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Stack all the robot states and human states
        robot_states = [state[0] for state in game_history.observation_history]
        human_states = [state[1] for state in game_history.observation_history]
        self.robot_states.extend(robot_states)
        self.human_states.extend(human_states)
        self.actions.extend(game_history.action_history)
        self.rewards.extend(rewards)
        self.logprobs.extend(game_history.logprobs)

        self.check_length()

    def check_length(self) -> None:
        if len(self.robot_states) > self.num_examples:

            self.robot_states = self.robot_states[-self.num_examples :]
            self.human_states = self.human_states[-self.num_examples :]
            self.actions = self.actions[-self.num_examples :]
            self.rewards = self.rewards[-self.num_examples :]
            self.logprobs = self.logprobs[-self.num_examples :]


def collate(batches):
    """This function takes in a list of data from the various data loading
    threads and combines them all into one batch for training."""
    (
        robot_state_batch,
        human_state_batch,
        action_batch,
        reward_batch,
        logprob_batch,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )

    for data_batch in batches:
        robot_state_batch.append(data_batch["robot_states"])
        human_state_batch.append(data_batch["human_states"])
        action_batch.append(torch.Tensor([data_batch["action"]]))
        reward_batch.append(torch.Tensor([data_batch["reward"]]))
        logprob_batch.append(torch.Tensor([data_batch["logprobs"]]))

    return (
        torch.stack(robot_state_batch),
        torch.stack(human_state_batch),
        torch.stack(action_batch),
        torch.stack(reward_batch),
        torch.stack(logprob_batch),
    )
