"""Continous PPO RL policy."""

from typing import Dict

import torch
from torch import nn
from torch import distributions

from crowd_search import models
from third_party.crowd_sim.envs.utils import agent_actions


class ContinuousPPO(nn.Module):
    """Describe the policy that governs the robot's decisions."""

    def __init__(
        self,
        models_cfg: Dict,
        robot_cfg: Dict,
        human_cfg: Dict,
        incentive_cfg: Dict,
        device: torch.device,
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
            models_cfg.get("gnn-output-depth"), action_space=2
        )
        self.action_predictor.to(self.device)
        self.action_predictor.train()

        self.dynamics_reward_network = models.DynamicsNetwork(models_cfg.get("gnn-output-depth"), 1)
        self.action_var = torch.full((2,), 0.5 ** 2)

    def forward(self):
        """Defined since this object inherits nn.Module."""
        raise NotImplementedError

    @torch.no_grad()
    def act(self, robot_state: torch.Tensor, human_states: torch.Tensor):
        """Function called by explorer."""
        human_states = human_states.transpose(0, -1)
        encoded_state = self.gnn(robot_state, human_states)
        action_mean = self.action_predictor(encoded_state)

        cov_mat = torch.diag(self.action_var).to(self.device)
        dist = distributions.MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()

        action_tensor = action.clamp(-self.v_pref, self.v_pref)
        action = agent_actions.ActionXY(action_tensor[0, 0].item(), action_tensor[0, 1].item())

        return action_tensor, action, dist.log_prob(action_tensor)

    def evaluate(self, robot_state: torch.Tensor, human_states: torch.Tensor, action):
        """Function called during training."""

        encoded_state = self.gnn(robot_state, human_states)
        action_mean = self.action_predictor(encoded_state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = distributions.MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.dynamics_reward_network(encoded_state)

        return action_logprobs, state_value.squeeze(-1), dist_entropy
