"""Describe the policy that governs the robot's decisions."""

import random
from typing import Dict

import numpy as np
import torch

from crowd_search import models
from third_party.crowd_sim.envs.utils.agent_actions import ActionXY
from third_party.crowd_sim.envs.utils.utils import point_to_segment_dist


class CrowdSearchPolicy:
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

        # Build the models.
        gnn_cfg = models_cfg.get("gnn")
        self.gnn = models.GNN(
            gnn_cfg.get("robot-state-dimension"),
            gnn_cfg.get("human-state-dimension"),
            attn_layer_channels=gnn_cfg.get("mlp-layer-channels"),
            num_attention_layers=gnn_cfg.get("gnn-layers"),
        )
        self.gnn.to(device=device)
        self.gnn.train()

        value_net_cfg = models_cfg.get("value-net")
        self.value_estimator = models.ValueNet(
            value_net_cfg.get("channel-depth"), value_net_cfg.get("net-dims")
        )
        self.value_estimator.to(self.device)
        self.value_estimator.train()

        state_net_cfg = models_cfg.get("state-net")
        self.state_estimator = models.StateNet(
            state_net_cfg.get("time-step"), state_net_cfg.get("channel-depth")
        )
        self.state_estimator.to(device)
        self.state_estimator.train()

        self.time_step = state_net_cfg.get("time-step")

        self.phase = "train"
        self.epsilon = 0.5
        self.action_space = None

        self.speed_samples = action_space_cfg.get("speed-samples")
        self.rotation_samples = action_space_cfg.get("rotation-samples")
        self.planning_depth = 1
        self.planning_width = 1
        self.gamma = 0.9
        self.v_pref = 1.0

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_phase(self, phase):
        self.phase = phase

    def set_time_step(self, time_step):
        self.time_step = time_step
        self.state_estimator.time_step = time_step

    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

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

    def predict(self, robot_state: torch.Tensor, human_states: torch.Tensor):
        """
        Args:
            robot_state: Tensor containing robot state. Comes in with size [1, 9]
            human_states: Tensor containing human states. Comes in with size [Num_humans, 5]
        """
        if self.action_space is None:
            self.build_action_space(self.v_pref)

        if self.reached_goal(robot_state):
            return ActionXY(0, 0)

        # Select random action ocassionally
        if self.phase == "train" and np.random.uniform() < self.epsilon:
            max_action = random.choice(self.action_space)
        else:
            max_action = None
            max_value = float("-inf")
            max_traj = None

            # Unsqeeze to add batch dim and BNC -> BCN. Since all our layers are
            # convolutions.
            robot_state = robot_state.unsqueeze(0).transpose(1, 2).to(self.device)
            human_states = human_states.unsqueeze(0).transpose(1, 2).to(self.device)
            # Get the embedding of the current robot and human states.
            state_embed = self.gnn(robot_state, human_states)
            # Given an known action from our potential actions space, estimate the state that
            # results from taking that action.
            for action in self.action_space:

                # Get next robot state and predicted human states given the action.
                next_robot_state, next_human_states = self.state_estimator(
                    robot_state=robot_state,
                    human_state_embed=state_embed[1],
                    action=action,
                )
                max_next_return, max_next_traj = self.V_planning(
                    next_robot_state, next_human_states, 2, self.planning_width,
                )
                reward_est = self.estimate_reward(robot_state, human_states, action)
                value = reward_est + self.get_normalized_gamma() * max_next_return
                if value > max_value:
                    max_value = value
                    max_action = action
                    max_traj = [
                        ((robot_state, human_states), action, reward_est)
                    ] + max_next_traj

            if max_action is None:
                raise ValueError("Value network is not well trained.")

        if self.phase == "train":
            self.last_state = (robot_state, human_states)
        else:
            self.traj = max_traj

        return max_action

    def V_planning(
        self, robot_state: torch.Tensor, human_state: torch.Tensor, depth, width
    ):
        """ Plans n steps into future. Computes the value for the current state as well as the trajectories
        defined as a list of (state, action, reward) triples

        """
        robot_state_embed, human_state_embed = self.gnn(robot_state, human_state)
        current_state_value = self.value_estimator(robot_state_embed)
        if depth == 1:
            return current_state_value, [((robot_state, human_state), None, None)]

        returns = []
        trajs = []

        for action in self.action_space:
            next_robot_state, next_human_state = self.state_estimator(
                robot_state, human_state_embed, action
            )
            reward_est = self.estimate_reward(
                next_robot_state, next_human_state, action
            )
            next_value, next_traj = self.V_planning(
                next_robot_state, next_human_state, depth - 1, self.planning_width
            )
            return_value = current_state_value / depth + (depth - 1) / depth * (
                self.get_normalized_gamma() * next_value + reward_est
            )

            returns.append(return_value)
            trajs.append([((robot_state, human_state), action, reward_est)] + next_traj)

        max_index = np.argmax(returns)
        max_return = returns[max_index]
        max_traj = trajs[max_index]

        return max_return, max_traj

    # TODO(alex): Could we have this estimator be a learned model?
    def estimate_reward(
        self,
        robot_state: torch.Tensor,
        next_human_states: torch.Tensor,
        action: ActionXY,
    ) -> float:
        # Squeeze out the batch since these environment explorations are always
        # non-batched. (The batch is just for the models)
        robot_state = robot_state.transpose(1, 2).squeeze(0)
        next_human_states = next_human_states.transpose(1, 2).squeeze(0)
        next_human_pos = next_human_states[:, :2]

        # Calculate where the robot will be given an action.
        x = robot_state[:, 0] + action.vx * self.time_step
        y = robot_state[:, 1] + action.vx * self.time_step
        next_robot_pos = torch.Tensor([[x, y]])  # size: [1, 2]

        distances = torch.linalg.norm(
            next_human_pos - next_robot_pos, dim=-1, keepdim=True
        )

        # If any of the distances to humans are less than the sum of their respective radii,
        # we have a collision.
        if (distances < self.robot_radius + self.human_radius).any():
            return self.incentives.get("collision")

        # If not collision, we want to check for discomfort levels.
        discomfort = distances[
            distances + self.discomfort_distance < self.robot_radius + self.human_radius
        ]
        if discomfort.numel():
            return (
                (torch.min(discomfort) - self.discomfort_distance)
                * self.discomfort_distance_factor
                * self.time_step
            )

        goal_position = torch.Tensor([robot_state[:, 4], robot_state[:, 5]]).to(
            self.device
        )
        reaching_goal = torch.norm(next_robot_pos - goal_position) < robot_state[0, -2]

        if reaching_goal:
            return 1.0
        else:
            return 0.0

    def reached_goal(self, robot_state: torch.Tensor) -> bool:
        """Return if the robot is within radius of its goal."""

        distance_to_goal = torch.norm(
            torch.cat([robot_state[:, 0], robot_state[:, 1]])
            - torch.cat([robot_state[:, 4], robot_state[:, 5]])
        )

        return distance_to_goal < robot_state[:, -2]
