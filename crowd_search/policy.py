"""Describe the policy that governs the robot's decisions."""

import copy
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from crowd_search import models
from third_party.crowd_sim.envs.utils.agent_actions import ActionXY
from third_party.crowd_sim.envs.utils.utils import point_to_segment_dist


class CrowdSearchPolicy(nn.Module):
    def __init__(
        self,
        models_cfg: Dict,
        robot_cfg: Dict,
        human_cfg: Dict,
        incentive_cfg: Dict,
        action_space_cfg: Dict,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.human_radius = human_cfg.get("radius")
        self.robot_radius = robot_cfg.get("radius")
        self.incentives = incentive_cfg.copy()
        self.discomfort_distance = incentive_cfg.get("discomfort-distance")
        self.discomfort_distance_factor = incentive_cfg.get("discomfort-penalty-factor")
        self.device = device

        # Build the different models.
        gnn_cfg = models_cfg.get("gnn")
        self.gnn = models.GNN(
            gnn_cfg.get("robot-state-dimension"),
            gnn_cfg.get("human-state-dimension"),
            mlp_layer_channels=gnn_cfg.get("mlp-layer-channels"),
            num_attention_layers=gnn_cfg.get("gnn-layers"),
        )
        self.gnn.to(device=device)
        self.gnn.train()

        self.speed_samples = action_space_cfg.get("speed-samples")
        self.rotation_samples = action_space_cfg.get("rotation-samples")

        self.v_pref = 1.0
        self.build_action_space(self.v_pref)
        self.full_support_size = 601
        self.action_space_size = (self.speed_samples * self.rotation_samples) + 1
        # TODO(alex): replace with stacked observations
        self.action_predictor = models.PredictionNetwork(
            action_space_size=self.action_space_size, input_state_dim=32,
        )
        self.action_predictor.to(self.device)
        self.action_predictor.train()

        self.dynamics_encoded_state_network = models.DynamicsNetwork(32 + 1, 32)
        self.dynamics_reward_network = models.DynamicsNetwork(
            32, self.full_support_size
        )

    def get_action_space_size(self) -> int:
        return len(self.action_space)

    def build_action_space(self, preferred_velocity: float):
        """Given a desired number of speed and rotation samples, build the action
        space available to the model."""

        # Shift speeds to no 0.0 speed.
        self.speeds = np.linspace(
            0, preferred_velocity, self.speed_samples, endpoint=False
        )
        self.speeds += self.speeds[1]
        self.rotations = np.linspace(
            0, 2 * np.pi, self.rotation_samples, endpoint=False
        )

        # Add a stop action.
        self.action_space = [ActionXY(0, 0)]

        for speed in self.speeds:
            for rotation in self.rotations:
                self.action_space.append(
                    ActionXY(speed * np.cos(rotation), speed * np.sin(rotation))
                )

    def initial_inference(self, robot_state: torch.Tensor, human_states: torch.Tensor):
        """
        Args:
            robot_state: Tensor containing robot state. Comes in with size:
                [batch, num_previous_frames, num_robots, 9]
            human_states: Tensor containing human states. Comes in with size:
                [batch, num_previous_frames, num_robots, 5]
        """
        batch_size, *_, robot_dim = robot_state.shape
        *_, human_dim = human_states.shape
        # Unsqeeze to add batch dim and BNC -> BCN. Since all our layers are
        # convolutional tensor format.
        robot_state = (
            robot_state.view(batch_size, -1, robot_dim).transpose(1, 2).to(self.device)
        )
        human_states = (
            human_states.view(batch_size, -1, human_dim).transpose(1, 2).to(self.device)
        )
        # Get the embedding of the current robot and human states.
        encoded_state = self.gnn(robot_state, human_states)
        # encoded_state = encoded_state.view(1, -1, 1)
        policy_logits, value = self.action_predictor.forward(encoded_state)
        reward = (
            torch.zeros(1, self.full_support_size)
            .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
            .repeat(len(encoded_state), 1)
            .to(encoded_state.device)
        )
        return value, reward, policy_logits, encoded_state

    def dynamics(
        self, encoded_state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones((encoded_state.shape[0], 1, encoded_state.shape[2],))
            .to(action.device)
            .float()
        )
        action_one_hot = action[:, :, None] * action_one_hot / self.action_space_size
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state = self.dynamics_encoded_state_network(x)
        reward = self.dynamics_reward_network(next_encoded_state).max(-1).values

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.view(
            -1, next_encoded_state.shape[1], next_encoded_state.shape[2],
        ).min(2, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.view(
            -1, next_encoded_state.shape[1], next_encoded_state.shape[2],
        ).max(2, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state
        return next_encoded_state, reward

    def recurrent_inference(self, encoded_state: torch.Tensor, action: torch.Tensor):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.action_predictor.forward(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


def support_to_scalar(action_logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(action_logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits
