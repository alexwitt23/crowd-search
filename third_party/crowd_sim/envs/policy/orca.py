import numpy as np
import rvo2
from third_party.crowd_sim.envs.policy.policy import Policy
from third_party.crowd_sim.envs.utils.agent_actions import ActionXY


class ORCA(Policy):
    def __init__(self):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__()
        self.name = "ORCA"
        self.trainable = False
        self.multiagent_training = True
        self.kinematics = "holonomic"
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None
        self.time_step = 0.25

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def predict(self, human_states, robot_state):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        params = (
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
        )
        if (
            self.sim is not None
            and self.sim.getNumAgents() != len(state.human_states) + 1
        ):
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(
                self.time_step, *params, self.radius, self.max_speed
            )
            self.sim.addAgent(
                (robot_state[0], robot_state[1]),
                *params,
                robot_state[4] + 0.01 + self.safety_space,
                robot_state[7],
                (robot_state[2], robot_state[3]),
            )
            for human_state in human_states:
                self.sim.addAgent(
                    (human_state[0], human_state[1]),
                    *params,
                    human_state[4] + 0.01 + self.safety_space,
                    self.max_speed,
                    (human_state[2], human_state[3]),
                )
        else:
            self.sim.setAgentPosition(0, robot_state.position)
            self.sim.setAgentVelocity(0, robot_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array(
            (robot_state[5] - robot_state[0], robot_state[6] - robot_state[1])
        )
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for idx, _ in enumerate(human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(idx + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = (robot_state, human_states)

        return action


class CentralizedORCA(ORCA):
    def __init__(self):
        super().__init__()

    def predict(self, state):
        """ Centralized planning for all agents """
        params = (
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
        )
        if self.sim is not None and self.sim.getNumAgents() != len(state):
            del self.sim
            self.sim = None

        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(
                self.time_step, *params, self.radius, self.max_speed
            )
            for agent_state in state:
                self.sim.addAgent(
                    (agent_state[0], agent_state[1]),
                    *params,
                    agent_state[4] + 0.01 + self.safety_space,
                    self.max_speed,
                    (agent_state[2], agent_state[3]),
                )
        else:
            for i, agent_state in enumerate(state):
                self.sim.setAgentPosition(i, agent_state.position)
                self.sim.setAgentVelocity(i, agent_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        for i, agent_state in enumerate(state):
            velocity = np.array(
                (agent_state.gx - agent_state.px, agent_state.gy - agent_state.py)
            )
            speed = np.linalg.norm(velocity)
            pref_vel = velocity / speed if speed > 1 else velocity
            self.sim.setAgentPrefVelocity(i, (pref_vel[0], pref_vel[1]))

        self.sim.doStep()
        actions = [ActionXY(*self.sim.getAgentVelocity(i)) for i in range(len(state))]

        return actions
