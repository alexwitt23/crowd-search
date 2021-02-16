"""Simulation environment for crowd search and navigation."""

from typing import Any, Dict, List

import torch
import rvo2

from crowd_search import agents
from third_party.crowd_sim.envs.utils import agent_actions


class ORCA:
    def __init__(self, environment_cfg: Dict[str, Any]) -> None:
        """
        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.
        """
        # Parse the environment configuration.
        self.kinematics = environment_cfg.get("kinematics")
        self.safety_space = environment_cfg.get("safety-space")
        self.neighbor_dist = environment_cfg.get("neighbor-distance")
        self.max_neighbors = environment_cfg.get("max-neighbors")
        self.time_horizon = environment_cfg.get("time-horizon")
        self.time_horizon_obst = environment_cfg.get("time-horizon-obst")
        self.radius = environment_cfg.get("radius")
        self.max_speed = environment_cfg.get("max-speed")
        self.time_step = environment_cfg.get("time-step")
        self.sim = None

    def predict(
        self, human_states, robot_states,
    ):
        """Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        num_agents = len(human_states) + len(robot_states)
        # If this is the first sime `predict` is called, we need to create the simulation
        # environment. Create the initial simulation environment and add our agents.
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(
                self.time_step,
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                self.radius,
                self.max_speed,
            )
            # Add the robot state. Currently this is just one robot, but well act like
            # there are multiple just to set ourselves up for future work with more than
            # robot, possibly.
            for robot_state in robot_states:
                self.sim.addAgent(
                    tuple(robot_state.get_position().tolist()),
                    self.neighbor_dist,
                    self.max_neighbors,
                    self.time_horizon,
                    self.time_horizon_obst,
                    robot_state.get_radius() + 0.01 + self.safety_space,
                    robot_state.get_preferred_velocity(),
                    tuple(robot_state.get_velocity().tolist()),
                )

            # We want the ORCA planner to plan the human motion while considering the
            # robot's position. Here we add all humans and the robot to the simulation.
            for human in human_states:
                self.sim.addAgent(
                    tuple(human.get_position().tolist()),
                    self.neighbor_dist,
                    self.max_neighbors,
                    self.time_horizon,
                    self.time_horizon_obst,
                    human.get_radius() + 0.01 + self.safety_space,
                    self.max_speed,
                    tuple(human.get_velocity().tolist()),
                )

        else:
            for idx, agent_state in enumerate(robot_states + human_states):
                self.sim.setAgentPosition(
                    idx, tuple(agent_state.get_position().tolist())
                )
                self.sim.setAgentVelocity(
                    idx, tuple(agent_state.get_velocity().tolist())
                )

                # Since the RVO2 simulation doesn't directly support setting goal
                # points for the agents, we calculate the velocity needed to get to each
                # agent's goal and set this to the preferred velocity of the agents.
                # Get the positional difference between the agent and its goal.
                velocity = agent_state.get_goal_position() - agent_state.get_position()
                speed = torch.norm(velocity)
                velocity = velocity / speed if speed > 1 else velocity
                self.sim.setAgentPrefVelocity(idx, tuple(velocity.tolist()))

        # Setp the simulation and extract the human agent actions.
        self.sim.doStep()
        actions = [
            agent_actions.ActionXY(*self.sim.getAgentVelocity(idx))
            for idx in range(num_agents)
        ][len(robot_states) :]

        return actions
