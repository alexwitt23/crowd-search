"""A dataset that will hold the explored history. There isn't much to this dataset,
really, it might be removed in the future. The key part is the collation function
which allows us to batch together the different collections of input data."""

from typing import Tuple

import numpy
import torch
from torch.utils import data

from crowd_search import explorer


class Dataset(data.Dataset):
    """Dataset class."""

    def __init__(self):
        super().__init__()
        self.transitions = []
        self.config = {
            "support-size": 300,
            "root-dirichlet-alpha": 0.25,
            "root-exploration-fraction": 0.25,
            "num-simulations": 10,
            "stacked-observations": 32,
            "pb-c-base": 19652,
            "pb-c-init": 1.25,
            "players": list(range(1)),
            "discount": 0.997,
            "td-steps": 10,
            "PER": True,
            "PER-alpha": 1,
            "num-unroll-steps": 5,
            "action-space": list(range(41)),
        }

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.transitions)

    def __getitem__(self, idx: int):
        """Retrieve an item from the dataset and prepare it for training."""
        # Retrieve the game from the list of available games.
        game_history = self.transitions[idx]

        # Choose a position in the episode's history based on the computed priorities.
        game_pos, pos_prob = self.sample_position(game_history)

        values, rewards, policies, actions = self.make_target(game_history, game_pos)
        observation_batch = game_history.get_stacked_observations(
            game_pos, self.config["stacked-observations"]
        )
        gradient_scale_batch = [
            min(
                self.config["num-unroll-steps"],
                len(game_history.action_history) - game_pos,
            )
        ] * len(actions)

        # TODO(alex): fix the 1.0
        if self.config["PER"]:
            weight_batch = 1 / (self.__len__() * 1.0 * pos_prob)

        return {
            "observation": observation_batch,
            "action": actions,
            "value": values,
            "reward": rewards,
            "policy": policies,
            "weight": weight_batch,
            "gradient": gradient_scale_batch,
        }

    @staticmethod
    def sample_position(game_history: explorer.GameHistory) -> Tuple[int, float]:
        """Sample position from game either uniformly or according to some priority.
        See paper appendix Training."""
        position_probs = game_history.priorities / sum(game_history.priorities)
        position_index = numpy.random.choice(len(position_probs), p=position_probs)
        position_prob = position_probs[position_index]

        return position_index, position_prob

    def make_target(self, game_history: explorer.GameHistory, state_index: int):
        """Generate targets for every unroll steps."""

        target_values, target_rewards, target_policies, actions = [], [], [], []
        for current_index in range(
            state_index, state_index + self.config["num-unroll-steps"] + 1
        ):
            value = self.compute_target_value(game_history, current_index)

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(game_history.action_history[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(self.config["action-space"]))

        return target_values, target_rewards, target_policies, actions

    def compute_target_value(self, game_history: explorer.GameHistory, index: int):
        """Docstring"""
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config["td-steps"]
        if bootstrap_index < len(game_history.root_values):
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            last_step_value = (
                root_values[bootstrap_index]
                if game_history.to_play_history[bootstrap_index]
                == game_history.to_play_history[index]
                else -root_values[bootstrap_index]
            )

            value = last_step_value * self.config["discount"] ** self.config["td-steps"]
        else:
            value = 0

        for i, reward in enumerate(
            game_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            value += (
                reward
                if game_history.to_play_history[index]
                == game_history.to_play_history[index + i]
                else -reward
            ) * self.config["discount"] ** i

        return value


def collate(batches):
    """This function takes in a list of data from the various data loading
    threads and combines them all into one batch for training."""
    (
        robot_state_batch,
        human_state_batch,
        action_batch,
        reward_batch,
        value_batch,
        policy_batch,
        gradient_scale_batch,
        weight_batch,
    ) = ([], [], [], [], [], [], [], [])

    for data_batch in batches:
        robot_state_batch.append(data_batch["observation"][0])
        human_state_batch.append(data_batch["observation"][1])
        action_batch.append(torch.Tensor(data_batch["action"]))
        value_batch.append(torch.Tensor(data_batch["value"]))
        reward_batch.append(torch.Tensor(data_batch["reward"]))
        policy_batch.append(torch.Tensor(data_batch["policy"]))
        gradient_scale_batch.append(torch.Tensor(data_batch["gradient"]))
        weight_batch.append(torch.Tensor([data_batch["weight"]]))

    robot_state_batch = torch.stack(robot_state_batch)
    human_state_batch = torch.stack(human_state_batch)
    action_batch = torch.stack(action_batch)
    value_batch = torch.stack(value_batch)
    reward_batch = torch.stack(reward_batch)
    policy_batch = torch.stack(policy_batch)
    gradient_scale_batch = torch.stack(gradient_scale_batch)
    weight_batch = torch.stack(weight_batch)
    weight_batch = weight_batch / torch.max(weight_batch, dim=1, keepdim=True).values

    return (
        robot_state_batch,
        human_state_batch,
        action_batch,
        value_batch,
        reward_batch,
        policy_batch,
        gradient_scale_batch,
        weight_batch,
        weight_batch,
    )
