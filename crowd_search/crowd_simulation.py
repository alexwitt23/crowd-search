import logging
import random
import math
from typing import Dict, Union

import torch
import gym
import numpy as np
from numpy.linalg import norm

from crowd_search import agents
from crowd_search import environment

from third_party.crowd_sim.envs.policy.policy_factory import policy_factory
from third_party.crowd_sim.envs.utils import agent
from third_party.crowd_sim.envs.utils import agent_actions
from third_party.crowd_sim.envs.utils.info import *
from third_party.crowd_sim.envs.utils.utils import point_to_segment_dist


class CrowdSim(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        env_cfg: Dict,
        incentive_cfg: Dict,
        motion_planner_cfg: Dict,
        human_cfg: Dict,
        robot_cfg: Dict,
    ) -> None:
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = env_cfg.get("time-limit")
        self.time_step = env_cfg.get("time-step")
        self.human_num = env_cfg.get("num-humans")

        self.human_cfg = human_cfg

        self.robot = agents.Robot(robot_cfg)
        self.humans = None
        self.global_time = None
        self.robot_sensor_range = None

        # Collect the various incentives.
        self.success_reward = incentive_cfg.get("success")
        self.collision_penalty = incentive_cfg.get("collision")
        self.discomfort_dist = incentive_cfg.get("discomfort-distance")
        self.discomfort_penalty_factor = incentive_cfg.get("discomfort-penalty-factor")

        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.current_scenario = None
        self.square_width = None
        self.circle_radius = 4
        self.nonstop_human = None
        # TODO(alex): Look into different planners. Socialforce models?
        self.motion_planner = environment.ORCA(motion_planner_cfg)

        # for visualization
        self.states = []
        self.phase = None

    def __str__(self) -> str:
        """Print a verbal description of the simulation environment."""
        self_str = f"number of humans: {self.human_num}\n"
        self_str += f"time limit: {self.time_limit}\n"
        self_str += f"time step: {self.time_step}\n"
        self_str += f"success reward: {self.success_reward}\n"
        self_str += f"collision penalty: {self.collision_penalty}\n"
        self_str += f"discomfort dist: {self.discomfort_dist}\n"
        self_str += f"discomfort penalty factor: {self.discomfort_penalty_factor}\n"

        return self_str

    def _generate_human(self) -> agent.Human:
        """Generate a circular blob representing a human."""

        # Create initial human agent.
        human = agents.Human(self.human_cfg)

        # Generate random human attributes and make sure the blob does not
        # collide with any of the existing blobs or robot.
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.get_preferred_velocity()
            py_noise = (np.random.random() - 0.5) * human.get_preferred_velocity()
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise

            for agent in [self.robot] + self.humans:
                min_dist = (
                    human.get_radius() + agent.get_radius() + self.discomfort_dist
                )
                if torch.norm(torch.Tensor([px, py]) - agent.get_position()) < min_dist:
                    # Collide, generate another human
                    break
            else:
                # No collision, set this human state.
                break

        human.set_state(
            position_x=px,
            position_y=py,
            velocity_x=0,
            velocity_y=0,
            goal_position_x=-px,
            goal_position_y=-py,
            direction=0.0,
        )

        return human

    def reset(self, phase: str = "test") -> torch.Tensor:
        """Reset the simulation environment.
        
        TODO(alex) implement phases
        """

        self.global_time = 0
        # TODO(alex): Set different states?

        self.robot.set_state(
            position_x=-self.circle_radius,
            position_y=0,
            velocity_x=0,
            velocity_y=self.circle_radius,
            goal_position_x=0,
            goal_position_y=0,
            direction=np.pi / 2,
        )

        # Generate the humans.
        self.humans = []
        for _ in range(self.human_num):
            self.humans.append(self._generate_human())

        # Get the robot's first initial observation
        robot_observation = self.collate_robot_observation()

        return robot_observation

    def step(
        self,
        action: Union[agent_actions.ActionXY, agent_actions.ActionRot],
        update=True,
    ):
        """Compute actions for all agents, detect collision, update environment and return
        (ob, reward, done, info). ORCA is used to control human motion."""

        # Get the human states and have the planner predict the human actions.
        human_actions = self.motion_planner.predict(self.humans, [self.robot])

        # collision detection
        dmin = float("inf")
        collision = False
        for i, human in enumerate(self.humans):
            px, py = (human.get_position() - self.robot.get_position()).tolist()
            vx, vy = (human.get_velocity() - self.robot.get_velocity()).tolist()

            ex = px + vx * self.time_step
            ey = py + vy * self.time_step

            # closest distance between boundaries of two agents
            closest_dist = (
                point_to_segment_dist(px, py, ex, ey, 0, 0)
                - human.radius
                - self.robot.get_radius()
            )
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # check if reaching the goal
        previous_robot_pos = self.robot.get_position()
        end_position = self.robot.compute_position(action, self.time_step)
        reaching_goal = (
            torch.norm(end_position - self.robot.get_goal_position())
            < self.robot.get_radius()
        )
        further_away = torch.norm(
            end_position - self.robot.get_goal_position()
        ) > torch.norm(previous_robot_pos - self.robot.get_goal_position())

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # adjust the reward based on FPS
            reward = (
                (dmin - self.discomfort_dist)
                * self.discomfort_penalty_factor
                * self.time_step
            )
            done = False
            info = Discomfort(dmin)
        elif further_away:
            reward = -0.01
            done = False
            info = Nothing()

        else:
            reward = 0.01
            done = False
            info = Nothing()

        if update:

            # update all agents
            self.robot.step(action)
            for human, action in zip(self.humans, human_actions):
                human.step(action)
                if self.nonstop_human and human.reached_destination():
                    self._generate_human(human)

            self.global_time += self.time_step
            ob = self.collate_robot_observation()

        else:
            ob = [
                human.get_next_observable_state(action)
                for human, action in zip(self.humans, human_actions)
            ]

        return ob, reward, done, info

    def compute_observation_for(self, agent):
        if agent == self.robot:
            ob = []
            for human in self.humans:
                ob.append(human.get_observable_state())
        else:
            ob = [
                other_human.get_observable_state()
                for other_human in self.humans
                if other_human != agent
            ]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
        return ob

    def collate_robot_observation(self):
        """The robot observation consists of the observable states of all other
        entities in the simulation."""

        return torch.stack([human.get_observable_state() for human in self.humans])

    def collate_human_observation(self, human: agent.Human):
        """A human's observations are the robots and other humans if they are
        visible."""
        other_humans = [
            human2.get_observable_state() for human2 in self.humans if human2 != human
        ]
        return other_humans + self.robot.get_observable_state()