"""This models humans traveling in groups.

Please refer to `crowd_search/configs/crowd.py` for the parameters that control
this model.
"""
import copy
import pathlib
import random
from typing import Dict, List, Tuple

import imageio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from crowd_search import agents
from crowd_search import environment
from crowd_search.configs import crowd
from third_party.crowd_sim.envs.utils import agent_actions

_COLORS = list(mcolors.BASE_COLORS)
_GROUP_COLORS = {}


class HumanGroup:
    """A class to control a single group of n humans."""

    def __init__(
        self,
        num_humans: Tuple[int, int],
        world_width: Tuple[int, int],
        world_height: Tuple[int, int],
    ) -> None:
        """Create the group of humans."""
        self.num_humans = num_humans
        self.world_width = world_width
        self.world_height = world_height

        center_x = random.randint(*self.world_width)
        center_y = random.randint(*self.world_height)
        self._create_goal()
        self.humans = []
        # Now choose a random number of humans and create the group
        for human in range(random.randint(*self.num_humans)):
            human = agents.Human(crowd.HUMAN_CFG)

            # Generate random human attributes and make sure the human does not
            # collide with any of the existing humans or robot.
            while True:
                pos_x = random.uniform(
                    center_x - crowd.GROUP_RADIUS, center_x + crowd.GROUP_RADIUS
                )
                pos_y = random.uniform(
                    center_y - crowd.GROUP_RADIUS, center_y + crowd.GROUP_RADIUS
                )

                # Loop over the humans already in the group and check if this new
                # candidate human is too close to others. # TODO(alex): is this
                # needed? ORCA will handle this eventually.
                for agent in self.humans:
                    min_dist = human.get_radius() + agent.get_radius() + 0.2
                    position = torch.Tensor([pos_x, pos_y])
                    if torch.norm(position - agent.get_position()) < min_dist:
                        # Collide, generate another human
                        break
                else:
                    # No collision with any other human or robot, set this human state.
                    break

            human.set_state(
                position_x=pos_x,
                position_y=pos_y,
                velocity_x=0.0,
                velocity_y=0.0,
                goal_position_x=self.goal_x,
                goal_position_y=self.goal_x,
                direction=0.0,
            )
            self.humans.append(human)

    def goal_reached(self) -> bool:
        """Check if the group's goal position has been reached."""
        positions = torch.stack([human.get_position() for human in self.humans])
        
        # Center
        center = torch.mean(positions, 0)
        if torch.norm(self.goal - center) < 10:
            return True
        else:
            return False

    def _create_goal(self):
        self.goal_x = random.choice(self.world_width)
        self.goal_y = random.choice(self.world_height)
        self.goal = torch.Tensor([self.goal_x, self.goal_y])
        print(self.goal, self.world_width, self.world_height)

    def set_new_goal(self):
        self._create_goal()
        for human in self.humans:
            human.set_goal(self.goal_x, self.goal_y)


class GroupSimulation:
    """A class to simulate humans moving in various different groups."""

    def __init__(self) -> None:
        self.group_range = (crowd.MIN_NUM_GROUPS, crowd.MAX_NUM_GROUPS)
        self.num_humans_range = (crowd.MIN_NUM_PER_GROUP, crowd.MAX_NUM_PER_GROUP)
        self.world_width = (-crowd.WORLD_WIDTH, crowd.WORLD_WIDTH)
        self.world_height = (-crowd.WORLD_HEIGHT, crowd.WORLD_HEIGHT)

        self.motion_planner = environment.ORCA(crowd.MOTION_PLANNER_CFG)
        self.groups = self._generate_crowd()

    def _generate_crowd(self) -> List[agents.Human]:
        """"""
        groups = {}
        for group in range(random.randint(*self.group_range)):
            groups[group] = HumanGroup(
                self.num_humans_range, self.world_width, self.world_height
            )

        return groups

    def step(self) -> None:
        """"""
        # Get the human states and have the planner predict the human actions.
        humans = [human for group in self.groups.values() for human in group.humans]
        human_actions = self.motion_planner.predict(humans, [])

        for human, human_action in zip(humans, human_actions):
            human.step(human_action, 0.25)

        for group in self.groups.values():
            if group.goal_reached():
                group.set_new_goal()
                print("new goal")


def plot_state(idx, human_states: List[agents.Human]):
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, (group_idx, group) in enumerate(human_states.items()):
        if group_idx in _GROUP_COLORS:
            color = _GROUP_COLORS[group_idx]
        else:
            color = random.choice(_COLORS)
            _GROUP_COLORS[group_idx] = random.choice(_COLORS)

        for human in group.humans:
            state = human.get_position()
            ax.plot(
                [state[0].item()],
                [state[1].item()],
                marker="o",
                markersize=3,
                color=color,
            )
    ax.set_ylim(-crowd.WORLD_HEIGHT, crowd.WORLD_HEIGHT)
    ax.set_xlim(-crowd.WORLD_WIDTH, crowd.WORLD_WIDTH)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def plot_history(crowd_history, save_path: pathlib.Path):
    save_path.parent.mkdir(exist_ok=True, parents=True)

    imageio.mimsave(
        str(save_path),
        [plot_state(idx, state) for idx, state in enumerate(crowd_history)],
        fps=30,
    )


if __name__ == "__main__":

    env = GroupSimulation()

    frames = []

    for _ in range(400):
        frames.append(copy.deepcopy(env.groups))
        env.step()

    plot_history(frames, pathlib.Path("/tmp/viz.gif"))
