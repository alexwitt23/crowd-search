import logging
import random
import math
from typing import Dict, Union

import torch
import gym
import matplotlib.lines as mlines
from matplotlib import patches
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

        self.robot = None
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
        self.circle_radius = None
        self.nonstop_human = None
        # TODO(alex): Look into different planners. Socialforce models?
        self.motion_planner = environment.ORCA(motion_planner_cfg)

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.robot_actions = None
        self.rewards = None
        self.As = None
        self.Xs = None
        self.feats = None
        self.trajs = list()
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.test_scene_seeds = []
        self.dynamic_human_num = []
        self.human_starts = []
        self.human_goals = []

        self.phase = None

    def configure(self, config):

        self.config = config
        self.randomize_attributes = config.env.randomize_attributes
        self.robot_sensor_range = config.env.robot_sensor_range
        self.case_capacity = {
            "train": np.iinfo(np.uint32).max - 2000,
            "val": 1000,
            "test": 1000,
        }
        self.case_size = {
            "train": config.env.train_size,
            "val": config.env.val_size,
            "test": config.env.test_size,
        }
        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
        self.square_width = config.sim.square_width
        self.circle_radius = config.sim.circle_radius

        self.nonstop_human = config.sim.nonstop_human
        self.case_counter = {"train": 0, "test": 0, "val": 0}

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

    def set_robot(self, robot):
        self.robot = robot

    def _generate_human(self) -> agent.Human:
        """Generate a circular blob representing a human."""

        # Create initial human agent.
        human = agents.Human(self.human_cfg)

        # Generate random human attributes and make sure the blob does not
        # collide with any of the existing blobs or robot.
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent_.radius + self.discomfort_dist
                if (
                    norm((px - agent_.px, py - agent_.py)) < min_dist
                    or norm((px - agent_.gx, py - agent_.gy)) < min_dist
                ):
                    collide = True
                    break
            if not collide:
                break

        human.set_state(px=px, py=py, gx=-px, gy=-py, vx=0, vy=0, theta=0)

        return human

    def reset(self, phase: str = "test", test_case=None):
        """Reset the simulation environment."""

        assert phase in ["train", "val"]
        self.phase = phase
        if self.robot is None:
            raise AttributeError("Robot has to be set!")

        self.global_time = 0
        # TODO(alex): Set different states?
        self.robot.set_state(
            -self.circle_radius, 0, 0, self.circle_radius, 0, 0, np.pi / 2,
        )

        if phase in ["train", "val"]:
            # only CADRL trains in circle crossing simulation
            self.current_scenario = "circle_crossing"
        else:
            self.current_scenario = self.test_scenario

        # Generate the humans.
        self.humans = []
        for _ in range(self.human_num):
            self.humans.append(self._generate_human())

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        # if self.centralized_planning:
        #    self.centralized_planner.time_step = self.time_step

        self.states = list()
        self.robot_actions = list()
        self.rewards = list()
        if hasattr(self.robot.policy, "action_values"):
            self.action_values = list()
        if hasattr(self.robot.policy, "get_attention_weights"):
            self.attention_weights = list()
        if hasattr(self.robot.policy, "get_matrix_A"):
            self.As = list()
        if hasattr(self.robot.policy, "get_feat"):
            self.feats = list()
        if hasattr(self.robot.policy, "get_X"):
            self.Xs = list()
        if hasattr(self.robot.policy, "trajs"):
            self.trajs = list()

        ob = self.collate_robot_observation()

        return ob

    def step(
        self,
        action: Union[agent_actions.ActionXY, agent_actions.ActionRot],
        update=True,
    ):
        """Compute actions for all agents, detect collision, update environment and return
        (ob, reward, done, info). ORCA is used to control human motion."""

        # Get the human states and have the planner predict the human actions.
        human_states = [human.get_full_state() for human in self.humans]
        human_actions = self.motion_planner.predict(
            human_states, self.robot.get_full_state()
        )

        # collision detection
        dmin = float("inf")
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            vx = human.vx - action.vx
            vy = human.vy - action.vy
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step

            # closest distance between boundaries of two agents
            closest_dist = (
                point_to_segment_dist(px, py, ex, ey, 0, 0)
                - human.radius
                - self.robot.radius
            )
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # check if reaching the goal
        previous_robot_pos = (self.robot.px, self.robot.py)
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = (
            norm(end_position - np.array(self.robot.get_goal_position()))
            < self.robot.radius
        )
        further_away = norm(
            end_position - np.array(self.robot.get_goal_position())
        ) > norm(previous_robot_pos - np.array(self.robot.get_goal_position()))

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
            # store state, action value and attention weights
            if hasattr(self.robot.policy, "action_values"):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, "get_attention_weights"):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            if hasattr(self.robot.policy, "get_matrix_A"):
                self.As.append(self.robot.policy.get_matrix_A())
            if hasattr(self.robot.policy, "get_feat"):
                self.feats.append(self.robot.policy.get_feat())
            if hasattr(self.robot.policy, "get_X"):
                self.Xs.append(self.robot.policy.get_X())
            if hasattr(self.robot.policy, "traj"):
                self.trajs.append(self.robot.policy.get_traj())

            # update all agents
            self.robot.step(action)
            for human, action in zip(self.humans, human_actions):
                human.step(action)
                if self.nonstop_human and human.reached_destination():
                    self._generate_human(human)

            self.global_time += self.time_step

            self.states.append(
                [
                    self.robot.get_full_state(),
                    [human.get_full_state() for human in self.humans],
                    [human.id for human in self.humans],
                ]
            )
            self.robot_actions.append(action)
            self.rewards.append(reward)

            # compute the observation
            if self.robot.sensor == "coordinates":
                ob = self.collate_robot_observation()
            elif self.robot.sensor == "RGB":
                raise NotImplementedError
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
