#!/usr/bin/env python3
"""Load the given model and run through an environment."""

import pathlib
import yaml

import torch
import gym
from crowd_search.policies import policy_factory


run_dir = pathlib.Path("~/runs/crowd-search/2021-03-25T19.34.41").expanduser()

cfg = yaml.safe_load((run_dir / "config.yaml").read_text())

environment = gym.make(
    "CrowdSim-v1",
    env_cfg=cfg.get("sim-environment"),
    incentive_cfg=cfg.get("incentives"),
    motion_planner_cfg=cfg.get("human-motion-planner"),
    human_cfg=cfg.get("human"),
    robot_cfg=cfg.get("robot"),
)
kwargs = {
    "device": torch.device("cpu"),
    "action_space": 2,
}
policy = policy_factory.make_policy(cfg.get("policy"), kwargs=kwargs)
policy.load_state_dict(torch.load(run_dir / "best_success.pt", map_location="cpu"))

simulation_done = False
observation = environment.reset()

# Loop over simulation steps until we are done. The simulation terminates
# when the goal is reached or some timeout based on the number of steps.
while not simulation_done:
    robot_state = environment.robot.get_full_state()
    action_tensor, action, action_log_prob = policy.act(
        robot_state.unsqueeze(-1), observation.unsqueeze(-1)
    )
    observation, reward, simulation_done = environment.step(action)

    print(simulation_done, reward)
