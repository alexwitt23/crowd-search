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


def MLP(channels: list):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim ** 0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    """Multi-head attention to increase model expressivitiy."""

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([copy.deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [
            l(x).view(batch_dim, self.dim, self.num_heads, -1)
            for l, x in zip(self.proj, (query, key, value))
        ]

        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class GNN(nn.Module):
    def __init__(
        self,
        robot_state_dim: int,
        human_state_dim: int,
        attn_layer_channels: List[int] = [32, 32],
        num_attention_layers: int = 1,
    ) -> None:
        """Initialize graph model.

        Args:
            robot_state_dim: Typically 9 for
                [pos_x, pos_y, v_x, v_y, radius, goal_x, goal_y, angle, v_preferred]
            human_state_dim: 5 for [pos_x, pos_y, v_x, v_y, angle] TODO(alex): angle?
        
        """
        super().__init__()

        # These MLPs expand the robot/human state dimension into a common size.
        self.robot_mlp = MLP([robot_state_dim] + attn_layer_channels)
        self.human_mlp = MLP([human_state_dim] + attn_layer_channels)

        self.robot_layers = nn.ModuleList(
            [
                AttentionalPropagation(attn_layer_channels[-1], 4)
                for _ in range(num_attention_layers)
            ]
        )
        self.human_layers = nn.ModuleList(
            [
                AttentionalPropagation(attn_layer_channels[-1], 4)
                for _ in range(num_attention_layers)
            ]
        )

    def forward(self, robot_state: torch.Tensor, human_state: torch.Tensor):

        robot_state = self.robot_mlp(robot_state)
        human_state = self.human_mlp(human_state)

        robot_state = robot_state
        human_state = human_state

        for robot_layer, human_layer in zip(self.robot_layers, self.human_layers):
            robot_state = robot_layer(robot_state, human_state)
            human_state = human_layer(human_state, human_state)

        return robot_state, human_state


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, net_dims: List[int]) -> None:
        super().__init__()

        self.value_net = MLP([state_dim] + net_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.value_net(x)


class StateNet(nn.Module):
    """A model to predict the next state given the GNN's encoded state.
    Here, state is 5 terms: (pos_x, pos_y, v_x, v_y, direction)."""

    def __init__(self, time_step: float, state_dim: int) -> None:
        super().__init__()
        self.time_step = time_step
        self.motion_predictor = MLP([state_dim, 5])

    def forward(
        self,
        robot_state: torch.Tensor,
        human_state_embed: torch.Tensor,
        action: agent_actions.ActionXY,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function which computes the next robot state using linear
        position update from velocity and the timestep and uses a MLP to predict
        the position of the humans."""

        next_robot_state = self._compute_next_state(robot_state, action)

        # Predict the motion of the humans from the embedded space.
        # TODO(alex): Shouldn't the human motion predictor consider the state
        # of the robot?
        next_human_states = self.motion_predictor(human_state_embed)

        return next_robot_state, next_human_states

    def _compute_next_state(
        self, robot_state: torch.Tensor, action: agent_actions.ActionXY
    ) -> torch.Tensor:
        """Take robot action and update it's position and velocity."""
        if action is None:
            return robot_state

        # px, py, vx, vy, radius, gx, gy, v_pref, theta
        next_state = robot_state.clone()

        next_state[:, 0] = next_state[:, 0] + action.vx * self.time_step
        next_state[:, 1] = next_state[:, 1] + action.vy * self.time_step
        next_state[:, 2] = action.vx
        next_state[:, 3] = action.vy

        return next_state
