"""Define different states for the different types of observations that can
be made from the different agents. A robot can only know position, velocity,
and radius of a person. TODO(alex): add direction?."""

import torch


class FullState:
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (
            self.px,
            self.py,
            self.vx,
            self.vy,
            self.radius,
            self.gx,
            self.gy,
            self.v_pref,
            self.theta,
        )

    def __str__(self):
        return " ".join(
            [
                str(x)
                for x in [
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
            ]
        )

    def to_tensor(self) -> torch.Tensor:
        """Return the state as a tensor. Unsqueeze to add dimension
        for 'number of robots'."""
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
        ).unsqueeze(0)

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)


class ObservableState:
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return " ".join(
            [str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]]
        )

    def to_tensor(self) -> torch.Tensor:
        """Return the state as a tensor."""
        return torch.Tensor([self.px, self.py, self.vx, self.vy, self.radius])


class JointState(object):
    def __init__(self, robot_state, human_states):
        assert isinstance(robot_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.robot_state = robot_state
        self.human_states = human_states

    def to_tensor(self, add_batch_size=False, device=None):
        robot_state_tensor = self.robot_state.to_tensor()

        human_states_tensor = torch.stack(
            [human_state.to_tensor() for human_state in self.human_states]
        )

        return robot_state_tensor.unsqueeze(0), human_states_tensor.unsqueeze(0)


def tensor_to_joint_state(state):
    robot_state, human_states = state

    robot_state = robot_state.squeeze()
    robot_state = FullState(
        robot_state[0],
        robot_state[1],
        robot_state[2],
        robot_state[3],
        robot_state[4],
        robot_state[5],
        robot_state[6],
        robot_state[7],
        robot_state[8],
    )
    human_states = [
        ObservableState(
            human_state[0],
            human_state[1],
            human_state[2],
            human_state[3],
            human_state[4],
        )
        for human_state in human_states.squeeze(0)
    ]

    return JointState(robot_state, human_states)
