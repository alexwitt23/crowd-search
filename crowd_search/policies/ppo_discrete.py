"""Continous PPO RL policy."""

import copy
from typing import Dict, List

import torch
from torch import nn
from torch import distributions

from crowd_search import explorer
from crowd_search import models
from third_party.crowd_sim.envs.utils import agent_actions
from crowd_search.policies import base_policy


class DiscretePPO(base_policy.BasePolicy):
    """Describe the policy that governs the robot's decisions."""

    data_keys = [
        "robot_states",
        "human_states",
        "action",
        "reward",
        "undiscounted_reward",
        "logprobs",
    ]

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
            models_cfg.get("gnn-output-depth"), len(self.action_space)
        )
        self.action_predictor.to(self.device)
        self.action_predictor.train()

        self.dynamics_reward_network = models.DynamicsNetwork(
            models_cfg.get("gnn-output-depth"), 1
        )

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

    @staticmethod
    def process_history(history):
        history = copy.deepcopy(history)
        discounted_rewards = []
        discounted_reward = 0
        for idx, reward in enumerate(reversed(history.reward_history)):
            if idx == 0:
                discounted_reward = 0
            discounted_reward = reward + (0.99 * discounted_reward)
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

        return datas


    def process_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        robot_state_batch = batch["robot_states"].to(self.device).transpose(1, 2)
        human_state_batch = batch["human_states"].to(self.device).transpose(1, 2)
        action_batch = batch["action"].to(self.device)
        target_reward = batch["reward"].to(self.device).squeeze(-1)
        logprobs_batch = batch["logprobs"].to(self.device).squeeze(-1)

        logprobs, state_values, dist_entropy = self.evaluate(robot_state_batch, human_state_batch, action_batch)

        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(logprobs - logprobs_batch.detach())

        # Finding Surrogate Loss:
        advantages = target_reward - state_values.detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        # TODO(alex): Maybe need to tweak these weights?
        loss = (
            -torch.min(surr1, surr2)
            + 0.5 * nn.functional.mse_loss(state_values, target_reward)
            - 0.01 * dist_entropy
        )
        return loss.mean()
