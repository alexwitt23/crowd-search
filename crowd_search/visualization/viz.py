"""A library of functions for taking in a game history and plotting
the results to a gif."""

import pathlib

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from crowd_search import agents


def plot_state(robot_state: torch.Tensor, human_states: torch.Tensor):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        [robot_state[0, 0].item()],
        [robot_state[0, 1].item()],
        marker="o",
        markersize=3,
        color="red",
    )
    ax.plot(
        [robot_state[0, 4].item()],
        [robot_state[0, 5].item()],
        marker="o",
        markersize=3,
        color="green",
    )
    for human in human_states:
        ax.plot(
            [human[0].item()],
            [human[1].item()],
            marker="o",
            markersize=3,
            color="blue",
        )
    ax.set_ylim(-10, 10)
    ax.set_xlim(-10, 10)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


def plot_history(game_history, save_path: pathlib.Path):
    save_path.parent.mkdir(exist_ok=True, parents=True)
    kwargs_write = {"fps": 30.0, "quantizer": "nq"}
    imageio.mimsave(
        str(save_path),
        [
            plot_state(robot_state, human_states)
            for robot_state, human_states in game_history.observation_history
        ],
        fps=1,
    )
