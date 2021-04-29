"""Basic agents classes meant to simplify and standardize state checking (getting
position, velocity, etc)."""

from typing import Any, Dict

import torch
import random

from third_party.crowd_sim.envs.utils import agent_actions
world_len = 20



class Agent:
    """A state class to house the state of both human or robots."""

    def __init__(self, state_tensor: torch.Tensor = None) -> None:
        # [pos_x, pos_y, vel_x, vel_y, goal_x, goal_y, direction, radius, v_pref]
        self.state_tensor = state_tensor
        self.radius = None
        self.preferred_velocity = None

    def get_full_state(self) -> torch.Tensor:
        """Get agent's full state. Unsqueeze to add dimension for 'agents'.

        Usage:
            >>> agent = Agent(torch.Tensor([0, 1, 2, 3, -1, -2, 0, 5, 4]))
            >>> agent.get_full_state()
            tensor([[ 0.,  1.,  2.,  3., -1., -2.,  0.,  5.,  4.]])
        """
        return self.state_tensor.unsqueeze(0)

    def get_position(self) -> torch.Tensor:
        """Get agent's position.

        Usage:
            >>> agent = Agent(torch.Tensor([0, 1, 2, 3, -1, -2, 0, 5, 4]))
            >>> agent.get_position()
            tensor([0., 1.])
        """
        return torch.Tensor([self.state_tensor[0], self.state_tensor[1]])

    def get_velocity(self) -> torch.Tensor:
        """Get agent's velocity.

        Usage:
            >>> agent = Agent(torch.Tensor([0, 1, 2, 3, -1, -2, 0, 5, 4]))
            >>> agent.get_velocity()
            tensor([2., 3.])
        """
        return torch.Tensor([self.state_tensor[2], self.state_tensor[3]])

    def get_goal_position(self) -> torch.Tensor:
        """Get goal position of the agent.

        Usage:
            >>> agent = Agent(torch.Tensor([0, 1, 2, 3, -1, -2, 0, 5, 4]))
            >>> agent.get_goal_position()
            tensor([-1., -2.])
        """
        return torch.Tensor([self.state_tensor[4], self.state_tensor[5]])

    def get_radius(self) -> float:
        """Get radius of agent.

        Usage:
            >>> agent = Agent(torch.Tensor([0, 1, 2, 3, -1, -2, 0, 5, 4]))
            >>> agent.get_radius()
            tensor(5.)
        """
        return self.state_tensor[7]

    def get_preferred_velocity(self) -> float:
        """Get agent's preferred velocity.

        Usage:
            >>> agent = Agent(torch.Tensor([0, 1, 2, 3, -1, -2, 0, 5, 4]))
            >>> agent.get_preferred_velocity()
            tensor(4.)
        """
        return self.state_tensor[-1]

    def get_observable_state(self) -> torch.Tensor:
        """Return what is observable to other agents:
        pos_x, pos_y, vel_x, vel_y, radius

        TODO(alex): do we want to add direction?

        Usage:
            >>> agent = Agent(torch.Tensor([0, 1, 2, 3, -1, -2, 0, 5, 4]))
            >>> agent.get_observable_state()
            tensor([0., 1., 2., 3., 5.])
        """
        return torch.Tensor(self.state_tensor[:4].tolist() + [self.state_tensor[-2]])

    def step(self, action: agent_actions.ActionXY, time_step: float) -> None:
        """Step the agent's position using the supplied action velocity vector."""
        self.state_tensor[0] = self.state_tensor[0] + action.vx * time_step
        self.state_tensor[1] = self.state_tensor[1] + action.vy * time_step
        self.state_tensor[2] = action.vx
        self.state_tensor[3] = action.vy

    def set_state(
        self,
        position_x: float,
        position_y: float,
        velocity_x: float,
        velocity_y: float,
        goal_position_x: float,
        goal_position_y: float,
        direction: float,
    ) -> None:
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


class Human(Agent):
    """Human class. Check the config.yaml for the parameters that control a human."""

    def __init__(self, human_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.radius = human_cfg.get("radius")
        self.preferred_velocity = human_cfg.get("preferred-velocity")
        self.group_id = 0

    def get_radius(self) -> float:
        return self.radius

    def get_preferred_velocity(self) -> float:
        return self.preferred_velocity

    def reached_destination(self) -> bool:
        return (
            torch.norm(self.get_goal_position() - self.get_position())
            < self.get_radius()
        )

    def set_group_id(self, id):
        self.troup_id=id



class Robot(Agent):
    """This class is very similar to the Human class."""

    def __init__(self, robot_cfg: Dict) -> None:
        super().__init__()
        self.radius = robot_cfg.get("radius")
        self.preferred_velocity = robot_cfg.get("preferred-velocity")

    def get_radius(self) -> float:
        return self.radius

    def get_preferred_velocity(self) -> float:
        return self.preferred_velocity

    def compute_position(
        self, action: agent_actions.ActionXY, time_step: float
    ) -> torch.Tensor:
        """Take an action (velocity vector) and timestep and return the new position."""

        pos = self.get_position() + torch.Tensor([action.vx, action.vy]) * time_step
        return pos

    def step(self, action, time_step: float):
        """Update the position and velocity."""
        pos = self.compute_position(action, time_step)
        self.state_tensor[0] = pos[0]
        self.state_tensor[1] = pos[1]
        self.state_tensor[2] = action.vx
        self.state_tensor[3] = action.vy

    def get_full_state(self) -> torch.Tensor:
        """Get agent's full state. Unsqueeze to add dimension for 'agents'.
        """
        return torch.Tensor(
            [
                self.state_tensor[0],
                self.state_tensor[1],
                self.state_tensor[4],
                self.state_tensor[5],
            ]
        ).unsqueeze(0)


class Group:
    """A state class to house the state of groups """
    def __init__(self, index_, group_cfg: Dict[str, Any]) -> None:
        # [start_x, start_y, goal_x, goal_y, direction, radius, v_pref]
        self.group_id=index_
        self.num_of_agents=  random.randint(group_cfg.get("min_humans"), group_cfg.get("max_humans")-1)
        self.radius = group_cfg.get("radius")
        self.average_velocity = group_cfg.get("avg_velocities")
        self.average_distances= group_cfg.get("avg_distances")
        # print("start_xs", group_cfg.get("start_positions_x"))
        # input("--")
        if index_<len(group_cfg.get("start_positions_x"))-1:
            self.start_positions= [group_cfg.get("start_positions_x")[index_], group_cfg.get("start_positions_y")[index_]]
            self.goal_positions= [group_cfg.get("goal_positions_x")[index_], group_cfg.get("goal_positions_y")[index_]]
        else:
            positionx = round(random.uniform(-world_len, world_len), 1)
            positiony = round(random.uniform(-world_len, world_len), 1)
            self.start_positions= [positionx, positiony]
            goalx = round(random.uniform(-world_len, world_len), 1)
            goaly = round(random.uniform(-world_len, world_len), 1)
            self.goal_positions= [goalx, goaly]

        self.agents=[]

    def add_agents(self, human: Human) -> None:
        self.agents.append(human)
        #genereate agents around start position


    def get_full_state(self) -> torch.Tensor:
        """Get agent's full state. Unsqueeze to add dimension for 'agents'.

        Usage:
            >>> agent = Agent(torch.Tensor([0, 1, 2, 3 ]))
            >>> agent.get_full_state()
            tensor([[ 0.,  1.,  2.,  3.]])
        """
        return self.state_tensor.unsqueeze(0)

    def generate_full_state(self) -> torch.Tensor:
        """Get agent's full state. Unsqueeze to add dimension for 'agents'.

        Usage:
            >>> agent = Agent(torch.Tensor([0, 1, 2, 3 ]))
            >>> agent.get_full_state()
            tensor([[ 0.,  1.,  2.,  3.]])
        """
        return self.state_tensor.unsqueeze(0)




