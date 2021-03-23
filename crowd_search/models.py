"""There are three models defined here:

1. A graph network that reasons about the interactions between the indivduals
   in the crowd and the robot.

2. A value estimation model that estimates the value of look ahead state space.

3. State prediction model that predicts the state of humans.
"""

import copy
from typing import List, Tuple

import torch
from torch import nn

from third_party.crowd_sim.envs.utils import agent_actions


def mlp(channels: list):
    """ Multi-layer perceptron """

    layers = []
    for i in range(1, len(channels)):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (len(channels) - 1):
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def attention(query, key, value):
    """Apply attention computation."""
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim ** 0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value)


class MultiHeadedAttention(nn.Module):
    """Multi-head attention to increase model expressivitiy."""

    def __init__(self, num_heads: int, d_model: int):
        """TODO(alex): docstring"""
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([copy.deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        """TODO(alex): docstring"""
        batch_dim = query.size(0)
        query, key, value = [
            layer(tensor).view(batch_dim, self.dim, self.num_heads, -1)
            for layer, tensor in zip(self.proj, (query, key, value))
        ]

        output = attention(query, key, value)
        return self.merge(
            output.contiguous().view(batch_dim, self.dim * self.num_heads, -1)
        )


class AttentionalPropagation(nn.Module):
    """TODO(alex): docstring"""

    def __init__(self, feature_dim: int, num_heads: int):
        """TODO(alex): docstring"""
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = mlp([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        """TODO(alex): docstring"""
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class GNN(nn.Module):
    def __init__(
        self,
        robot_state_dim: int,
        human_state_dim: int,
        mlp_layer_channels: List[int] = [32, 32],
        num_attention_layers: int = 1,
    ) -> None:
        """This model creates an encoded representation of the robot and human states.
        First, the robot and human state tensors are passed through an mlp which
        expands the tensors into the same channel dimension. Then the two states are
        concatenated into one and passed through a GNN."""
        super().__init__()

        # These mlps expand the robot/human state dimension into a common size.
        self.robot_mlp = mlp([robot_state_dim] + mlp_layer_channels)
        self.human_mlp = mlp([human_state_dim] + mlp_layer_channels)

        self.attn_layers = nn.ModuleList(
            [
                AttentionalPropagation(mlp_layer_channels[-1], 4)
                for _ in range(num_attention_layers)
            ]
        )

    def forward(self, robot_state: torch.Tensor, human_state: torch.Tensor):
        """TODO(alex): docstring"""
        robot_state = self.robot_mlp(robot_state)
        #human_state = self.human_mlp(human_state)
        # Concatenate the state tensors together
        #combined_state = torch.cat([robot_state, human_state], dim=-1)
        #for layer in self.attn_layers:
        #    combined_state = layer(combined_state, combined_state)
        return robot_state


# Similar to:
# https://github.com/werner-duvaud/muzero-general/blob/97e4931617e789e6880c87769d348d53dba20897/models.py#L390
class PredictionNetwork(nn.Module):
    """TODO(alex): docstring"""

    def __init__(self, input_state_dim: int, action_space: int) -> None:
        """TODO(alex): docstring"""
        super().__init__()
        out_size = 2 * input_state_dim
        self.action_predictor = mlp([input_state_dim, 2 * input_state_dim, 3 * input_state_dim, 3 * input_state_dim, out_size])
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.action_head = nn.Sequential(
            nn.Linear(out_size, action_space, bias=False), nn.Tanh()
        )

    def forward(
        self, embedded_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO(alex): docstring"""
        policy_logits = self.avg(self.action_predictor(embedded_state)).squeeze(-1)
        return self.action_head(policy_logits)


# Similar to:
# https://github.com/werner-duvaud/muzero-general/blob/97e4931617e789e6880c87769d348d53dba20897/models.py#L352
class DynamicsNetwork(nn.Module):
    """TODO(alex): docstring"""

    def __init__(
        self, num_channels, full_support_size,
    ):
        """TODO(alex): docstring"""
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.mlp = mlp(
            [num_channels, 2 * num_channels, 3 * num_channels,  3 * num_channels, full_support_size]
        )

    def forward(self, x):
        """Takes in a concatenated state embedding and action logits"""
        return self.avg(self.mlp(x)).squeeze(-1)
