"""Explorer objects which maintain a copy of the environment."""
import copy
import math
import time

import gym
import torch
from torch.distributed import rpc

from crowd_search import agents
from crowd_search.policies import policy_factory
from third_party.crowd_sim.envs.utils.agent_actions import ActionXY


class Explorer:
    def __init__(self, cfg, storage_node) -> None:
        """Initialize Explorer.

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

        kwargs = {
            "device": torch.device("cpu"),
            "action_space": self.environment.action_space,
        }
        self.policy = policy_factory.make_policy(cfg.get("policy"), kwargs=kwargs)
        self.policy_id = None
        self.storage_node = storage_node
        self._update_policy()
        self.num_epochs = cfg.get("training").get("num-epochs")
        self.run_continuous_episode()

    def _update_policy(self):
        new_policy = None
        while new_policy is None:
            new_policy, policy_id = self.storage_node.rpc_sync().get_policy()
            time.sleep(2.0)

        # If same policy, keep exploring
        if self.policy_id == policy_id:
            return
        else:
            self.policy_id = policy_id
            self.policy.load_state_dict(copy.deepcopy(new_policy.state_dict()))
            self.policy.eval()

    @torch.no_grad()
    def run_continuous_episode(self):
        """Run a single episode of the crowd search game."""

        while True:
            simulation_done = False
            # Reset the environment at the beginning of each episode and add initial
            # information to replay memory.
            game_history = GameHistory()
            observation = copy.deepcopy(self.environment.reset())

            # Loop over simulation steps until we are done. The simulation terminates
            # when the goal is reached or some timeout based on the number of steps.
            while not simulation_done:
                robot_state = copy.deepcopy(self.environment.robot.get_full_state())
                game_history.observation_history.append(
                    (robot_state, copy.deepcopy(observation))
                )
                action_tensor, action, action_log_prob = self.policy.act(
                    robot_state.unsqueeze(-1), observation.unsqueeze(-1)
                )
                observation, reward, simulation_done = self.environment.step(action)
                game_history.reward_history.append(copy.deepcopy(reward))
                game_history.action_history.append(action_tensor)
                game_history.logprobs.append(copy.deepcopy(action_log_prob))

            self.send_history(game_history)
            self._update_policy()

            if self.storage_node.rpc_sync().get_epoch() == self.num_epochs:

                return
            
            while (
                self.storage_node.rpc_sync().get_history_amount()
                > self.storage_node.rpc_sync().get_history_capacity()
            ):
                time.sleep(5.0)
                if self.storage_node.rpc_sync().get_epoch() > self.num_epochs:
                    return


    def send_history(self, history):
        self.storage_node.rpc_async().upload_history(history)



class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.logprobs = []
