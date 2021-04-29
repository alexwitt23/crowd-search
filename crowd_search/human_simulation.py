#!/usr/bin/env python3
"""Load the given model and run through an environment."""

import pathlib
import yaml

import torch
import gym

from crowd_search import explorer
from crowd_search.policies import policy_factory
from crowd_search.visualization import viz

run_dir = pathlib.Path("~/runs/crowd-search/2021-03-26T20.29.20").expanduser()
save_path = pathlib.Path("gifs/graphic.gif")

cfg = yaml.safe_load((run_dir / "config.yaml").read_text())

environment = gym.make(
    "GroupCrowdSim-v1",
    env_cfg=cfg.get("sim-environment"),
    incentive_cfg=cfg.get("incentives"),
    motion_planner_cfg=cfg.get("human-motion-planner"),
    human_cfg=cfg.get("human"),
    group_cfg=cfg.get("group"),
    robot_cfg=cfg.get("robot"),
)
print("environment", environment )
# input("--")
kwargs = {
    "device": torch.device("cpu"),
    "action_space": environment.action_space,
}
policy = policy_factory.make_policy(cfg.get("policy"), kwargs=kwargs)
policy.load_state_dict(torch.load(run_dir / "fewest_steps.pt", map_location="cpu"))
policy.eval()

simulation_done = False
observation = environment.reset()
histories = []
# Loop over simulation steps until we are done. The simulation terminates
# when the goal is reached or some timeout based on the number of steps.
while not simulation_done:
    robot_state = environment.robot.get_full_state()
    history = {}
    history["robot_states"] = robot_state
    history["human_states"] = observation
    histories.append(history)
    action = policy.act_static(
        robot_state.unsqueeze(-1), observation.unsqueeze(-1)
    )
    print(action)
    observation, reward, simulation_done = environment.step(action)

    print(simulation_done, reward)
print("reward", reward)
if reward != 0:
    viz.plot_history(histories, save_path)
