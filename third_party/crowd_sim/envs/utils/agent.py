"""Define a base agent class that is shared across robots and human."""

import abc
import logging
from typing import List

import torch
import numpy as np
from numpy.linalg import norm

from third_party.crowd_sim.envs.policy.policy_factory import policy_factory
from third_party.crowd_sim.envs.utils.agent_actions import ActionXY, ActionRot
from third_party.crowd_sim.envs.utils import state
from third_party.crowd_sim.envs.utils.state import (
    ObservableState,
    FullState,
    JointState,
)


class Agent:
    def __init__(self, config, section):
        self.visible = getattr(config, section).visible
        self.v_pref = getattr(config, section).v_pref
        self.radius = getattr(config, section).radius
        self.policy = policy_factory[getattr(config, section).policy]()

        self.sensor = getattr(config, section).sensor
        self.kinematics = self.policy.kinematics if self.policy is not None else None
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None

    def print_info(self):
        logging.info(
            "Agent is {} and has {} kinematic constraint".format(
                "visible" if self.visible else "invisible", self.kinematics
            )
        )

    def set_policy(self, policy):
        if self.time_step is None:
            raise ValueError("Time step is None")
        policy.set_time_step(self.time_step)
        self.policy = policy
        self.kinematics = "holonomic"

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set_state(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.sx = px
        self.sy = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self):
        return torch.Tensor([self.px, self.py, self.vx, self.vy, self.radius])

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == "holonomic":
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_vx = action.v * np.cos(self.theta)
            next_vy = action.v * np.sin(self.theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return torch.Tensor(
            [
                self.px,
                self.py,
                self.vx,
                self.vy,
                self.radius,
                self.gx,
                self.gy,
                self.v_pref,
                self.theta,
            ]
        )

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_start_position(self):
        return self.sx, self.sy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == "holonomic":
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == "holonomic":
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t

        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        if self.kinematics == "holonomic":
            self.vx = action.vx
            self.vy = action.vy
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

    def reached_destination(self):
        return (
            norm(np.array(self.get_position()) - np.array(self.get_goal_position()))
            < self.radius
        )


class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.id = None

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, human_states: torch.Tensor) -> torch.Tensor:
        """Given the observable human states, act.
        
        Args:
            human_states: A tensor of the human states in size Num_human, 5.
                5 for dimension of human state.
        """
        if self.policy is None:
            raise AttributeError("Policy attribute has to be set!")
        # Get the robot's state.
        robot_state = self.get_full_state().unsqueeze(0)
        action = self.policy.predict(robot_state, human_states)

        return action
