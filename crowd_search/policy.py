"""Describe the policy that governs the robot's decisions."""

import random
from typing import Dict

import torch
import numpy as np
from torch.distributed import rpc

from crowd_search import models
from third_party.crowd_sim.envs.utils.agent_actions import ActionRot, ActionXY
from third_party.crowd_sim.envs.utils.state import tensor_to_joint_state
from third_party.crowd_sim.envs.utils.utils import point_to_segment_dist


class CrowdSearchPolicy:
    def __init__(
        self,
        models_cfg: Dict,
        robot_cfg: Dict,
        human_cfg: Dict,
        incentive_cfg: Dict,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.human_radius = human_cfg.get("radius")
        self.robot_radius = robot_cfg.get("radius")
        self.incentives = incentive_cfg.copy()

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

        value_net_cfg = models_cfg.get("value-net")
        self.value_estimator = models.ValueNet(
            value_net_cfg.get("channel-depth"), value_net_cfg.get("net-dims")
        )
        self.value_estimator.to(self.device)

        state_net_cfg = models_cfg.get("state-net")
        self.state_estimator = models.StateNet(
            state_net_cfg.get("time-step"), state_net_cfg.get("channel-depth")
        )
        self.state_estimator.to(device)

        self.time_step = state_net_cfg.get("time-step")

        self.phase = "train"
        self.epsilon = 0.0
        self.action_space = None
        self.kinematics = "holonomic"
        self.speed_samples = 5
        self.rotation_samples = 4
        self.sampling = "exponential"
        self.query_env = False
        self.rotation_constraint = np.pi / 3
        self.action_group_index = []
        self.sparse_speed_samples = 2
        self.sparse_rotation_samples = 8
        self.do_action_clip = False
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
        # if self.reach_destination(state):
        #    return ActionXY(0, 0) if self.kinematics == "holonomic" else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(self.v_pref)

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
                    human_state_embed=state_embed[0],
                    action=action,
                )

                # TODO(alex): Since planning_depth is 1 right now, the V_planning just
                # performs a gnn embedding of the next state and returns the value of
                # being in that state.
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

    def action_clip(self, state, action_space, width, depth=1):
        values = []

        for action in action_space:
            next_state_est = self.state_estimator(state, action)
            next_return, _ = self.V_planning(next_state_est, depth, width)
            reward_est = self.estimate_reward(state, action)
            value = reward_est + self.get_normalized_gamma() * next_return
            values.append(value)

        if self.sparse_search:
            # self.sparse_speed_samples = 2
            # search in a sparse grained action space
            added_groups = set()
            max_indices = np.argsort(np.array(values))[::-1]
            clipped_action_space = []
            for index in max_indices:
                if self.action_group_index[index] not in added_groups:
                    clipped_action_space.append(action_space[index])
                    added_groups.add(self.action_group_index[index])
                    if len(clipped_action_space) == width:
                        break
        else:
            max_indexes = np.argpartition(np.array(values), -width)[-width:]
            clipped_action_space = [action_space[i] for i in max_indexes]

        return clipped_action_space

    @torch.no_grad()
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

        # if self.do_action_clip:
        #    action_space_clipped = self.action_clip(state, self.action_space, width)
        # else:
        action_space_clipped = self.action_space

        returns = []
        trajs = []

        for action in action_space_clipped:
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

    @torch.no_grad()
    def estimate_reward(
        self, robot_state: torch.Tensor, human_states: torch.Tensor, action: ActionXY
    ) -> float:
        # TODO(alex): this should really be batched
        # BCN -> BNC -> NC
        robot_state = robot_state.transpose(1, 2).squeeze(0)
        human_states = human_states.transpose(1, 2).squeeze(0)

        dmin = float("inf")
        collision = False
        for i, human in enumerate(human_states):

            px = human[0] - robot_state[:, 0]
            py = human[1] - robot_state[:, 1]
            vx = human[2] - action.vx
            vy = human[3] - action.vy
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = (
                point_to_segment_dist(px, py, ex, ey, 0, 0)
                - human[4]  # radius
                - robot_state[0, 4]  # radius
            )
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # check if reaching the goal
        if self.kinematics == "holonomic":
            px = robot_state[:, 0] + action.vx * self.time_step
            py = robot_state[:, 1] + action.vy * self.time_step
        else:
            theta = robot_state.theta + action.r
            px = robot_state.px + np.cos(theta) * action.v * self.time_step
            py = robot_state.py + np.sin(theta) * action.v * self.time_step

        start_pos = torch.Tensor([robot_state[:, 0], robot_state[:, 1]]).to(self.device)
        end_position = torch.Tensor([px, py]).to(self.device)
        goal_position = torch.Tensor([robot_state[:, 5], robot_state[:, 6]]).to(
            self.device
        )
        reaching_goal = torch.norm(end_position - goal_position) < robot_state[0, 4]
        furthur_away = torch.norm(end_position - goal_position) > torch.norm(
            start_pos - goal_position
        )
        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        elif dmin < 0.2:
            # adjust the reward based on FPS
            reward = (dmin - 0.2) * 0.5 * self.time_step
        elif furthur_away:
            reward = -0.01
        else:
            reward = 0.01

        return reward

    @staticmethod
    def reach_destination(state):
        robot_state = state.robot_state
        if (
            np.linalg.norm(
                (robot_state.py - robot_state.gy, robot_state.px - robot_state.gx)
            )
            < robot_state.radius
        ):
            return True
        else:
            return False
