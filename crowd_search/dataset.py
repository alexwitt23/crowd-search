"""A dataset that will hold the explored history."""

import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self):
        self.transitions = []
        self.trainable_length = 0

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]


def collate_fn(data_batches):
    """A custom collate function that will loop over the incoming list of dictionaries
    and stack the tensors into training-ready format."""
    robot_state = []
    humans_state = []
    value = []
    reward = []
    next_robot_state = []
    next_humans_state = []
    for batch in data_batches:
        robot_state.append(batch["robot_state"])
        humans_state.append(batch["humans_state"])
        value.append(batch["value"])
        reward.append(batch["reward"])
        next_robot_state.append(batch["next_robot_state"])
        next_humans_state.append(batch["next_humans_state"])

    robot_state = torch.stack(robot_state).transpose(-2, -1)
    humans_state = torch.stack(humans_state).transpose(-2, -1)
    value = torch.stack(value)
    reward = torch.stack(reward)
    next_robot_state = torch.stack(next_robot_state).transpose(-2, -1)
    next_humans_state = torch.stack(next_humans_state).transpose(-2, -1)

    return (
        robot_state,
        humans_state,
        value,
        reward,
        next_robot_state,
        next_humans_state,
    )
