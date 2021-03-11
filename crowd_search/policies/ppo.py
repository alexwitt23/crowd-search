"""Continous PPO RL policy."""

from typing import Dict, List

import torch
from torch import nn
from torch import distributions

from crowd_search import models
from third_party.crowd_sim.envs.utils import agent_actions


class PPO(nn.Module):
    """Describe the policy that governs the robot's decisions."""

    def __init__(
        self,
        models_cfg: Dict,
        robot_cfg: Dict,
        human_cfg: Dict,
        incentive_cfg: Dict,
        device: torch.device,
        action_space: List[agent_actions.ActionXY] = None,
        **kwargs
    ) -> None:
        """Initialize the PPO"""
        super().__init__()

        self.human_radius = human_cfg.get("radius")
        self.robot_radius = robot_cfg.get("radius")
        self.incentives = incentive_cfg.copy()
        self.discomfort_distance = incentive_cfg.get("discomfort-distance")
        self.discomfort_distance_factor = incentive_cfg.get("discomfort-penalty-factor")
        self.device = device
        self.action_space = action_space

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

        self.v_pref = 1.0
        self.action_predictor = models.PredictionNetwork(
            action_space=len(self.action_space), input_state_dim=32
        )
        self.action_predictor.to(self.device)
        self.action_predictor.train()

        self.dynamics_reward_network = models.DynamicsNetwork(32, 1)

    def forward(self):
        """Defined since this object inherits nn.Module."""
        raise NotImplementedError

    @torch.no_grad()
    def act(
        self, robot_state: torch.Tensor, human_states: torch.Tensor
    ) -> agent_actions.ActionXY:
        """Function called by explorer."""
        human_states = human_states.transpose(0, -1)
        encoded_state = self.gnn(robot_state, human_states)
        action_mean = self.action_predictor(encoded_state).softmax(-1)

        dist = distributions.Categorical(action_mean)
        action = dist.sample()

        return action, self.action_space[action.item()], dist.log_prob(action)

    def evaluate(self, robot_state: torch.Tensor, human_states: torch.Tensor, action):
        """Function called during training."""

        encoded_state = self.gnn(robot_state, human_states)
        action_mean = self.action_predictor(encoded_state)

        dist = distributions.Categorical(action_mean)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.dynamics_reward_network(encoded_state)

        return action_logprobs, state_value.squeeze(-1), dist_entropy
