"""A dataset to keep track of the states, actions, and rewards during
exploration of the environment."""

import copy
import time
import random

import ray

from crowd_search import shared_storage


@ray.remote
class ReplayBuffer:
    def __init__(self, buffer_size: int = 10000) -> None:
        self.episode_buffer = []
        self.number_episodes = 0
        self.buffer_size = buffer_size

    def save_game(self, game_history, storage: shared_storage.SharedStorage) -> None:
        self.episode_buffer += copy.deepcopy(game_history)
        self.number_episodes += 1
        storage.set_info.remote("num_played_games", self.number_episodes)

        if len(self.episode_buffer) > self.buffer_size:
            random.shuffle(self.episode_buffer)
            self.episode_buffer = self.episode_buffer[self.buffer_size]

    def get_batch(self, batch_size: int = 200):
        while len(self.episode_buffer) < batch_size:
            time.sleep(1.0)

        robot_states = []
        human_states = []
        rewards = []
        next_robot_state = []
        next_human_state = []

        for data in self.episode_buffer:
            robot_states.append(data[0])
            human_states.append(data[1])
            rewards.append(data[3])
            next_robot_state.append(data[4])
            next_human_state.append(data[5])

            if len(next_human_state) > batch_size:
                return (
                    robot_states,
                    human_states,
                    rewards,
                    next_robot_state,
                    next_human_state,
                )
