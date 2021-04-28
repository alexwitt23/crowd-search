"""This models humans traveling in groups.

Please refer to `crowd_search/configs/crowd.py` for the parameters that control
this model.
"""
import copy
import pathlib
import random
from typing import Dict, List

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

_COLORS = list(mcolors.CSS4_COLORS)


class GroupSimulation:
    """A class to simulate humans moving in various different groups."""

    def __init__(self) -> None:
        self.group_range = (crowd.MIN_NUM_GROUPS, crowd.MAX_NUM_GROUPS)
        self.num_humans_range = (crowd.MIN_NUM_PER_GROUP, crowd.MAX_NUM_PER_GROUP)
        self.world_width = (-crowd.WORLD_WIDTH, crowd.WORLD_WIDTH)
        self.world_height = (-crowd.WORLD_HEIGHT, crowd.WORLD_HEIGHT)

        self.humans = []
        self.motion_planner = environment.ORCA(crowd.MOTION_PLANNER_CFG)
        self.groups = self._generate_crowd()

    def _generate_crowd(self) -> List[agents.Human]:
        """"""
        groups = {}
        for group in range(random.randint(*self.group_range)):
            groups[group] = []
    
            # Give each group a center of mass and a goal point
            center_x = random.randint(*self.world_width)
            center_y = random.randint(*self.world_height)
            goal_x = random.randint(*self.world_width)
            goal_y = random.randint(*self.world_height)

            # Now choose a random number of humans and create the group
            for human in range(random.randint(*self.num_humans_range)):
                human = agents.Human(crowd.HUMAN_CFG)

                # Generate random human attributes and make sure the human does not
                # collide with any of the existing humans or robot.
                while True:
                    pos_x = random.uniform(center_x - crowd.GROUP_RADIUS, center_x + crowd.GROUP_RADIUS)
                    pos_y = random.uniform(center_y - crowd.GROUP_RADIUS, center_y + crowd.GROUP_RADIUS)

                    # Loop over the humans already in the group and check if this new
                    # candidate human is too close to others. # TODO(alex): is this
                    # needed? ORCA will handle this eventually.
                    for agent in groups[group]:
                        min_dist = (
                            human.get_radius() + agent.get_radius() + 0.2
                        )
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
                    goal_position_x=goal_x,
                    goal_position_y=goal_y,
                    direction=0.0,
                )
                groups[group].append(human)
                self.humans.append(human)

        return groups

    def step(self) -> None:
        """"""
        # Get the human states and have the planner predict the human actions.
        human_actions = self.motion_planner.predict(self.humans, [])

        for idx, (human, human_action) in enumerate(
            zip(self.humans, human_actions)
        ):
            human.step(human_action, 0.25)
            self.humans[idx] = human


def plot_state(idx, human_states: List[agents.Human]):
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, (group_idx, humans) in enumerate(human_states.items()):
        for human in humans:
            state = human.get_position()
            ax.plot(
                [state[0].item()],
                [state[1].item()],
                marker="o",
                markersize=3,
                color=_COLORS[idx],
            )
    ax.set_ylim(-25, 25)
    ax.set_xlim(-25, 25)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def plot_history(crowd_history, save_path: pathlib.Path):
    save_path.parent.mkdir(exist_ok=True, parents=True)

    imageio.mimsave(
        str(save_path),
        [
            plot_state(idx, state)
            for idx, state in enumerate(crowd_history)
        ],
        fps=15,
    )


if __name__ == "__main__":

    env = GroupSimulation()

    frames = []

    for _ in range(100):
        frames.append(copy.deepcopy(env.groups))
        env.step()
        print(env.humans[0].get_position())
    
    plot_history(frames, pathlib.Path("/tmp/viz.gif"))
    