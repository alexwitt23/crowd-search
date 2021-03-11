"""This defined the OpenAI Gym game that we'll use to simulate a crowd of humans
that our robot(s) can navigate through."""

import random
from typing import Dict

import gym
import numpy as np
import torch

from crowd_search import agents
from crowd_search import environment
from third_party.crowd_sim.envs.utils import agent_actions

# from third_party.crowd_sim.envs.utils import info
from third_party.crowd_sim.envs.utils.utils import point_to_segment_dist


class CrowdSim(gym.Env):
    """Crowd simulation environment."""

    def __init__(
        self,
        env_cfg: Dict,
        incentive_cfg: Dict,
        motion_planner_cfg: Dict,
        human_cfg: Dict,
        robot_cfg: Dict,
    ) -> None:
        """Movement simulation for agents plus robot. The humans will be controlled by
        some known policy like ORCA or socialforce rules. The robot learns a policy
        to navigate through the crowd."""

        self.time_limit = env_cfg.get("time-limit")
        self.time_step = env_cfg.get("time-step")
        self.human_num = env_cfg.get("num-humans")
        self.world_width = env_cfg.get("world-width") // 2
        self.world_height = env_cfg.get("world-height") // 2
        self.goal_location_width = env_cfg.get("goal-location-width") // 2
        self.goal_location_height = env_cfg.get("goal-location-height") // 2

        self.human_cfg = human_cfg
        self.humans = []
        self.robot = agents.Robot(robot_cfg)
        # Used to keep track of when simulation times out
        self.global_time = None
        self.robot_sensor_range = None

        # Collect the various incentives.
        self.success_reward = incentive_cfg.get("success")
        self.collision_penalty = incentive_cfg.get("collision")
        self.discomfort_dist = incentive_cfg.get("discomfort-distance")
        self.discomfort_penalty_factor = incentive_cfg.get("discomfort-penalty-factor")

        # TODO(alex): Look into different planners. Socialforce models?
        # TODO(alex): build a human policy factory
        self.motion_planner = environment.ORCA(motion_planner_cfg)
        # TODO(alex): un hardcode
        self.speed_samples = 5
        # The number of rotation samples to consider.
        self.rotation_samples = 16

        self.build_action_space(preferred_velocity=1.0)

    def __str__(self) -> str:
        """Print a verbal description of the simulation environment. Can be helpful
        for debugging."""
        self_str = f"number of humans: {self.human_num}\n"
        self_str += f"time limit: {self.time_limit}\n"
        self_str += f"time step: {self.time_step}\n"
        self_str += f"success reward: {self.success_reward}\n"
        self_str += f"collision penalty: {self.collision_penalty}\n"
        self_str += f"discomfort dist: {self.discomfort_dist}\n"
        self_str += f"discomfort penalty factor: {self.discomfort_penalty_factor}\n"

        return self_str

    def _generate_human(self, human: agents.Human = None) -> agents.Human:
        """Generate a circular blob representing a human.
        
        If a human is passed in, its state is reset. A human is typically passed
        in when an episode is ongoing but the human has reached its destination.
        The human is given a goal at a random location anywhere within the world
        specified in the config.yaml
        """

        # Create initial human agent.
        if human is None:
            human = agents.Human(self.human_cfg)

        # Generate random human attributes and make sure the human does not
        # collide with any of the existing humans or robot.
        while True:
            positionx = round(random.uniform(-self.world_width, self.world_width), 5)
            positiony = round(random.uniform(-self.world_height, self.world_height), 5)

            for agent in [self.robot] + self.humans:
                min_dist = (
                    human.get_radius() + agent.get_radius() + self.discomfort_dist
                )
                position = torch.Tensor([positionx, positiony])
                if torch.norm(position - agent.get_position()) < min_dist:
                    # Collide, generate another human
                    break
            else:
                # No collision with any other human or robot, set this human state.
                break

        # TODO(alex): Do we want more fine-grained control over human motion like
        # specifying the goal position?
        # TODO(alex): Make sure velocity (0,0) is ok.
        human.set_state(
            position_x=positionx,
            position_y=positiony,
            velocity_x=0.0,
            velocity_y=0.0,
            goal_position_x=round(
                random.uniform(-self.world_width, self.world_width), 5
            ),
            goal_position_y=round(
                random.uniform(-self.world_height, self.world_height), 5
            ),
            direction=0.0,
        )
        return human

    def reset(self) -> torch.Tensor:
        """Reset the simulation environment. Reset the global time, set a random
        robot state, and regenerate the human agents."""

        self.global_time = 0
        # TODO(alex): Do better state checking

        pos = [0, 0]
        while True:
            x = random.uniform(-self.world_width, -1)
            if -1 <= x <= 1:
                continue
            else:
                pos[0] = x
                break
        while True:
            y = random.uniform(-self.world_height, -1)
            if -1 <= y <= 1:
                continue
            else:
                pos[1] = y
                break

        goal_x = random.uniform(-self.goal_location_width, self.goal_location_width)
        goal_y = random.uniform(-self.goal_location_height, self.goal_location_height)
        self.robot.set_state(
            position_x=round(pos[0], 5),
            position_y=round(pos[1], 5),
            velocity_x=0.0,
            velocity_y=0.0,
            goal_position_x=round(goal_x, 5),
            goal_position_y=round(goal_y, 5),
            direction=0.0,
        )
        # Generate the humans.
        self.humans = []
        for _ in range(self.human_num):
            self.humans.append(self._generate_human())

        # Get the robot's first initial observation
        robot_observation = self.collate_robot_observation()

        # Reset the human motion planner simulation.
        self.motion_planner.sim = None
        return robot_observation

    def step(
        self, action: agent_actions.ActionXY,
    ):
        """Compute actions for all agents, detect collision, update environment and return
        (ob, reward, done, info). ORCA is used to control human motion."""

        # Get the human states and have the planner predict the human actions.
        human_actions = self.motion_planner.predict(self.humans, [self.robot])

        # collision detection
        dmin = float("inf")
        collision = False
        for human in self.humans:
            pos_x, pos_y = (human.get_position() - self.robot.get_position()).tolist()
            vel_x, vel_y = (human.get_velocity() - self.robot.get_velocity()).tolist()
            future_x = pos_x + vel_x * self.time_step
            future_y = pos_y + vel_y * self.time_step

            # closest distance between boundaries of two agents
            goal_x, goal_y = self.robot.get_goal_position()
            closest_dist = (
                point_to_segment_dist(pos_x, pos_y, future_x, future_y, goal_x, goal_y)
                - human.get_radius()
                - self.robot.get_radius()
            )
            if closest_dist < 0:
                collision = True
                break
            if closest_dist < dmin:
                dmin = closest_dist

        # check if reaching the goal
        end_position = self.robot.compute_position(action, self.time_step)

        reaching_goal = (
            torch.norm(end_position - self.robot.get_goal_position())
            < self.robot.get_radius()
        )
        dist_to_goal_fut = torch.norm(end_position - self.robot.get_goal_position())
        dist_to_goal_now = torch.norm(
            self.robot.get_goal_position() - self.robot.get_position()
        )
        further_away = dist_to_goal_now - dist_to_goal_fut

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
        elif collision:
            reward = self.collision_penalty
            done = True
        elif reaching_goal:
            reward = self.success_reward
            done = True
        elif dmin < self.discomfort_dist:
            # adjust the reward based on FPS
            reward = (
                (dmin - self.discomfort_dist)
                * self.discomfort_penalty_factor
                * self.time_step
            )
            done = False
        else:
            reward = 0.0
            done = False

        # update all agents
        self.robot.step(action, self.time_step)

        for human, human_action in zip(self.humans, human_actions):
            human.step(human_action, self.time_step)

            if human.reached_destination():
                self._generate_human(human)

        self.global_time += self.time_step
        robot_observation = self.collate_robot_observation()
        return robot_observation, reward, done

    def collate_robot_observation(self):
        """The robot observation consists of the observable states of all other
        entities in the simulation."""

        return torch.stack([human.get_observable_state() for human in self.humans])

    def legal_actions(self):
        return len(self.action_space)

    def render(self, mode):
        raise NotImplementedError

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
        self.action_space = [agent_actions.ActionXY(0, 0)]

        for speed in self.speeds:
            for rotation in self.rotations:
                self.action_space.append(
                    agent_actions.ActionXY(
                        round(speed * np.cos(rotation), 5),
                        round(speed * np.sin(rotation), 5),
                    )
                )
