"""This is a template that defines the required functions to add for any new policy.
This is made to support modularity between difference policy experiments."""

import abc
from typing import Dict

import torch
from torch import nn


class BasePolicy(nn.Module):
    """A basic policy class which defines the functions needed to interface with
    the training code."""

    def forward(self):
        """Defined since this object inherits nn.Module."""
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, robot_state: torch.Tensor, human_states: torch.Tensor):
        """Take in the robot state and human states and process them.
        This function is called during the _exploration_."""

        return NotImplementedError

    @abc.abstractmethod
    def evaluate(self, robot_state: torch.Tensor, human_states: torch.Tensor, action):
        """Take in the robot state and human states and process them.
        This function is called during the training stage."""

        return NotImplementedError

    @abc.abstractstaticmethod
    def process_history(history):
        """Process the history for traininh."""

        return NotImplementedError

    @abc.abstractstaticmethod
    def process_batch(data: Dict[str, torch.Tensor]):
        """Process the history for trainin. Returns a loss"""

        return NotImplementedError
