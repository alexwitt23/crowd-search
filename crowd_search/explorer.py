"""Explorer objects which maintain a copy of the environment."""
import math
import copy
from typing import Dict, List

import numpy
import gym
import torch
from torch.distributed import rpc

from crowd_search import agents
from crowd_search import policy2


class Explorer:
    def __init__(self, cfg, storage_node) -> None:
        """Initialize Explorer.z

        Args:
            environment: The simulation environment to be explored.
            robot: The robot agent to use to explore the environment.
            robot_policy: The policy that governs the robot's decisions.
            learner_idx: The numerical id of the associated trainer node.
        """
        self.id = rpc.get_worker_info().id
        # Copy to make entirly sure these are owned by the explorer process.
        self.environment = gym.make(
            "CrowdSim-v1",
            env_cfg=cfg.get("sim-environment"),
            incentive_cfg=cfg.get("incentives"),
            motion_planner_cfg=cfg.get("human-motion-planner"),
            human_cfg=cfg.get("human"),
            robot_cfg=cfg.get("robot"),
        )

        self.robot = agents.Robot(cfg.get("robot"))
        self.policy = policy2.CrowdSearchPolicy(
            cfg.get("models"),
            cfg.get("robot"),
            cfg.get("human"),
            cfg.get("incentives"),
            cfg.get("action-space"),
            "cpu",
        )
        self.config = {
            "support-size": 300,
            "action-space": list(range(len(self.policy.action_space))),
            "root-dirichlet-alpha": 0.25,
            "root-exploration-fraction": 0.25,
            "num-simulations": 10,
            "stacked-observations": 32,
            "pb-c-base": 19652,
            "pb-c-init": 1.25,
            "players": list(range(1)),
            "discount": 0.997,
            "td-steps": 10,
            "PER-alpha": True,
        }
        self.history = []
        self.storage_node = storage_node
        self._update_policy()

    def _update_policy(self):
        new_policy = None
        while new_policy is None:
            new_policy = self.storage_node.rpc_sync().get_policy()
        self.policy.load_state_dict(new_policy.state_dict())

    def get_history_len(self):
        """Helper function to return length of the history currently held."""
        return len(self.history)

    def get_history(self):
        """Return the history. This function will bee called by trainer nodes."""
        return self.history

    def clear_history(self):
        """Clear the history. Typically called by trainer node after new history has
        been recieved."""
        self.history.clear()

    @torch.no_grad()
    def run_continuous_episode(self):
        """Run a single episode of the crowd search game."""

        while True:
            states, actions, rewards = [], [], []
            simulation_done = False

            # Reset the environment at the beginning of each episode and add initial
            # information to replay memory.
            game_history = GameHistory()
            observation = self.environment.reset("train")
            robot_state = self.environment.robot.get_full_state()
            game_history.observation_history.append((robot_state, observation))
            game_history.reward_history.append(0)
            game_history.action_history.append(0)

            # Loop over simulation steps until we are done. The simulation terminates
            # when the goal is reached or some timeout based on the number of steps.
            while not simulation_done:

                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config["stacked-observations"],
                )

                root, mcts_info = MCTS(self.config).run(
                    self.policy,
                    stacked_observations,
                    self.environment.legal_actions(),
                    0,
                    True,
                )
                temperature_threshold = None
                temperature = 0
                action = self.select_action(
                    root,
                    temperature
                    if not temperature_threshold
                    or len(game_history.action_history) < temperature_threshold
                    else 0,
                )
                observation, reward, simulation_done = self.environment.step(
                    self.policy.action_space[action]
                )

                game_history.store_search_statistics(root, self.config["action-space"])
                game_history.observation_history.append(
                    (
                        copy.deepcopy(self.environment.robot.get_full_state()),
                        copy.deepcopy(observation),
                    )
                )
                game_history.reward_history.append(copy.deepcopy(reward))
                game_history.action_history.append(copy.deepcopy(action))
                game_history.to_play_history.append(0)

            history = self._process_epoch(game_history)
            self.send_history(history)
            self._update_policy()

    def send_history(self, history):
        self.storage_node.rpc_sync().upload_history(history)
        self.clear_history()

    def _process_epoch(self, game_history) -> None:

        # Initial priorities for the prioritized replay (See paper appendix Training)
        priorities = []
        for i, root_value in enumerate(game_history.root_values):
            priority = (
                numpy.abs(root_value - self.compute_target_value(game_history, i))
                ** self.config["PER-alpha"]
            )
            priorities.append(priority)

        game_history.priorities = numpy.array(priorities, dtype="float32")
        game_history.game_priority = numpy.max(game_history.priorities)

        return game_history

    def compute_target_value(self, game_history, index):
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

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action


# Game independent
class MCTS:
    """Core Monte Carlo Tree Search algorithm. To decide on an action, we run N
    simulations, always starting at the root of the search tree and traversing
    the tree according to the UCB formula until we reach a leaf node."""

    def __init__(self, config):
        self.config = config

    def run(
        self,
        policy,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation. We then run a Monte Carlo Tree Search
        using only action sequences and the model learned by the network.
        """
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)

            # Add batch dimension to robot and human state tensors.
            robot_states = observation[0].unsqueeze(0)
            human_states = observation[1].unsqueeze(0)

            (
                root_predicted_value,
                reward,
                policy_logits,
                hidden_state,
            ) = policy.initial_inference(robot_states, human_states)

            root_predicted_value = policy2.support_to_scalar(
                root_predicted_value, self.config["support-size"]
            ).item()
            reward = policy2.support_to_scalar(
                reward, self.config["support-size"]
            ).item()

            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            # assert set(legal_actions).issubset(
            #    set(self.config["action-space"])
            # ), "Legal actions should be a subset of the action space."
            root.expand(
                legal_actions, to_play, reward, policy_logits, hidden_state,
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config["root-dirichlet-alpha"],
                exploration_fraction=self.config["root-exploration-fraction"],
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config["num-simulations"]):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config["players"]):
                    virtual_to_play = self.config["players"][virtual_to_play + 1]
                else:
                    virtual_to_play = self.config["players"][0]

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = policy.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]]).to(parent.hidden_state.device),
            )
            value = policy2.support_to_scalar(value, self.config["support-size"]).item()
            reward = policy2.support_to_scalar(
                reward, self.config["support-size"]
            ).item()
            node.expand(
                self.config["action-space"],
                virtual_to_play,
                reward,
                policy_logits,
                hidden_state,
            )

            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = numpy.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config["pb-c-base"] + 1)
                / self.config["pb-c-base"]
            )
            + self.config["pb-c-init"]
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config["discount"]
                * (
                    child.value()
                    if len(self.config["players"]) == 1
                    else -child.value()
                )
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config["players"]) == 1:
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(
                    node.reward + self.config["discount"] * node.value()
                )

                value = node.reward + self.config["discount"] * value

        elif len(self.config.players) == 2:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(
                    node.reward + self.config["discount"] * -node.value()
                )

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config["discount"] * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state
        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )

            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(self, index: int, num_stacked_observations: int):
        """Generated stacked observations.

        Start at some index and get the previous num_stacked_observations from the
        history. If there aren't enough previous histories to complete the stack,
        padd with obseervations of zeros.
        """
        # Convert to positive index. i.e. -1 % 2 = 1. So we'd start at index 1 and
        # go back from there.
        index = index % len(self.observation_history)
        # This stacked_observations is a tuple of stacked robot state tensors
        # and human state tensors.
        stacked_observations = copy.deepcopy(self.observation_history[index])

        # Loop over the observation history and retrieve, or pad with the previous
        # history.
        previous_observations = []
        for past_observation_index in range(
            index - 1, index - num_stacked_observations, -1
        ):
            if 0 <= past_observation_index:
                previous_observations.append(
                    (
                        self.observation_history[past_observation_index][0],
                        self.observation_history[past_observation_index][1],
                    )
                )
            else:
                previous_observations.append(
                    (
                        torch.zeros_like(stacked_observations[0]),
                        torch.zeros_like(stacked_observations[1]),
                    )
                )

        # Now stack all the previous observation tensors.
        robot_states = torch.stack([state[0] for state in previous_observations])
        human_states = torch.stack([state[1] for state in previous_observations])
        stacked_observations_r = torch.cat(
            [stacked_observations[0].unsqueeze(0), robot_states]
        )
        stacked_observations_h = torch.cat(
            [stacked_observations[1].unsqueeze(0), human_states]
        )

        return (stacked_observations_r, stacked_observations_h)


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
