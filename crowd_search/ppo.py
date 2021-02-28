from typing import Dict, Tuple

import torch
from torch import nn
from torch import distributions
import numpy as np

from crowd_search import models
from third_party.crowd_sim.envs.utils.agent_actions import ActionXY


class PPO(nn.Module):
    """Describe the policy that governs the robot's decisions."""

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
        self.action_space_size = (self.speed_samples * self.rotation_samples) + 1
        # TODO(alex): replace with stacked observations
        self.action_predictor = models.PredictionNetwork(
            action_space_size=self.action_space_size, input_state_dim=4,
        )
        self.action_predictor.to(self.device)
        self.action_predictor.train()

        self.dynamics_reward_network = models.DynamicsNetwork(4, 1)

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, robot_state: torch.Tensor, human_states: torch.Tensor):
        #encoded_state = self.gnn(robot_state, human_states)
        policy_logits = self.action_predictor(robot_state)
        dist = distributions.Categorical(policy_logits)
        action = dist.sample()

        return action, dist.log_prob(action)

    def evaluate(self, robot_state: torch.Tensor, human_states: torch.Tensor, action):

        #encoded_state = self.gnn(robot_state, human_states)
        policy_logits = self.action_predictor(robot_state)
        dist = distributions.Categorical(policy_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.dynamics_reward_network(robot_state)

        return action_logprobs, state_value.squeeze(-1), dist_entropy

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
