from typing import Any, Dict

import torch


import torch


class Agent:
    """A state class to house the state of both human or robots."""

    def __init__(self) -> None:
        # [pos_x, pos_y, vel_x, vel_y, goal_x, goal_y, direction, radius, v_pref]
        self.state_tensor = None

    def get_full_state(self) -> torch.Tensor:
        return self.state_tensor

    def get_position(self) -> torch.Tensor:
        return torch.Tensor([self.state_tensor[0], self.state_tensor[1]])

    def get_velocity(self) -> torch.Tensor:
        return torch.Tensor([self.state_tensor[2], self.state_tensor[3]])

    def get_goal_position(self) -> torch.Tensor:
        return torch.Tensor([self.state_tensor[4], self.state_tensor[5]])

    def get_radius(self) -> float:
        return self.state_tensor[7]

    def get_preferred_velocity(self) -> float:
        return self.state_tensor[8]

    def get_observable_state(self) -> torch.Tensor:
        """Return what is observable to other agents:
        pos_x, pos_y, vel_x, vel_y, preferred_vel"""
        return torch.Tensor(self.state_tensor[:4].tolist() + [self.state_tensor[7]])

    def step(self, action):
        self.state_tensor[2] = action.vx
        self.state_tensor[3] = action.vy


class Human(Agent):
    """Human class. Check the config.yaml for the parameters that control a human."""

    def __init__(self, human_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.radius = human_cfg.get("radius")
        self.preferred_velocity = human_cfg.get("preferred-velocity")

    def set_state(
        self,
        position_x: float,
        position_y: float,
        velocity_x: float,
        velocity_y: float,
        goal_position_x: float,
        goal_position_y: float,
        direction: float,
    ):
        """Set the state tensor which has all the information about this agent."""
        self.state_tensor = torch.Tensor(
            [
                position_x,
                position_y,
                velocity_x,
                velocity_y,
                goal_position_x,
                goal_position_y,
                direction,
                self.radius,
                self.preferred_velocity,
            ]
        )

    def get_radius(self) -> float:
        return self.radius

    def get_preferred_velocity(self) -> float:
        return self.preferred_velocity

    def get_full_state(self) -> torch.Tensor:
        return self.state_tensor.unsqueeze(0)

class Robot(Agent):
    """This class is very similar to the Human class execpt that we also
    allow the robot to _act_ in the environment by giving it a policy to control."""

    def __init__(self, robot_cfg: Dict) -> None:
        super().__init__()
        self.radius = robot_cfg.get("radius")
        self.preferred_velocity = robot_cfg.get("preferred-velocity")

    def get_radius(self) -> float:
        return self.radius

    def get_preferred_velocity(self) -> float:
        return self.preferred_velocity

    def set_state(
        self,
        position_x: float,
        position_y: float,
        velocity_x: float,
        velocity_y: float,
        goal_position_x: float,
        goal_position_y: float,
        direction: float,
    ):
        """Set the state tensor which has all the information about this agent."""
        self.state_tensor = torch.Tensor(
            [
                position_x,
                position_y,
                velocity_x,
                velocity_y,
                goal_position_x,
                goal_position_y,
                direction,
                self.radius,
                self.preferred_velocity,
            ]
        )

    def compute_position(self, action, delta_t: float):
        pos = self.get_position() + torch.Tensor([action.vx, action.vy]) * delta_t
        return pos


    def get_full_state(self) -> torch.Tensor:
        return self.state_tensor.unsqueeze(0)