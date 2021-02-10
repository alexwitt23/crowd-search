#!/usr/bin/env python3

"""Trainer script to train crowd search agent."""

import argparse
import datetime
import os
import pathlib

import copy
import gym
import ray
import torch
from torch.utils import data
import yaml

from crowd_search import config
from crowd_search import explorer
from crowd_search import policy
from crowd_search import models
from crowd_search import trainer
from crowd_search import shared_storage
from crowd_search import replay_buffer
from third_party.crowd_sim.envs.policy import orca
from third_party.crowd_sim.envs.utils import agent

_LOG_DIR = pathlib.Path("~/runs/crowd-search").expanduser()


def train(local_rank: int, world_size: int, cfg: dict, run_dir: pathlib.Path) -> None:

    device = torch.device("cpu")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    # Make the environment registered in `third_party.crowd_sim.__init__.py`
    environment = gym.make(
        "CrowdSim-v0",
        env_cfg=cfg.get("sim-environment"),
        incentive_cfg=cfg.get("incentives"),
    )
    robot_policy = policy.CrowdSearchPolicy(
        cfg.get("models"),
        cfg.get("robot"),
        cfg.get("human"),
        cfg.get("incentives"),
        device,
    )
    sim_robot = agent.Robot(config.BaseEnvConfig(), "robot")
    sim_robot.time_step = environment.time_step
    sim_robot.set_policy(robot_policy)
    environment.configure(config.BaseEnvConfig())
    environment.set_robot(sim_robot)
    print(environment)
    training_worker = trainer.Trainer.options(num_cpus=0, num_gpus=0).remote(
        cfg.get("models"), device
    )

    play_workers = [
        explorer.Explorer.options(num_cpus=0, num_gpus=0).remote(
            environment, sim_robot, robot_policy, gamma=0.9
        )
        for _ in range(3)
    ]
    replay_buffer_worker = replay_buffer.ReplayBuffer.remote()
    shared_storage_worker = shared_storage.SharedStorage.remote()
    [
        play_worker.continuous_play.remote(replay_buffer_worker, shared_storage_worker)
        for play_worker in play_workers
    ]
    training_worker.continous_weight_update.remote(
        replay_buffer_worker, shared_storage_worker
    )
    import time

    while True:
        time.sleep(1.0)
        pass

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--config_path",
        type=pathlib.Path,
        default=pathlib.Path("crowd_search/config.yaml"),
    )
    args = parser.parse_args()

    train_cfg = yaml.safe_load(args.config_path.read_text())

    timestamp = datetime.datetime.now().isoformat().split(".")[0].replace(":", ".")

    run_dir = _LOG_DIR / timestamp
    run_dir.mkdir(exist_ok=True, parents=True)

    use_cuda = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_cuda else 1  # GPUS or a CPU

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"

    ray.init(num_cpus=1, num_gpus=torch.cuda.device_count())
    if world_size > 1:
        torch.multiprocessing.spawn(
            train, (world_size, train_cfg, run_dir), nprocs=world_size, join=True,
        )
    else:
        train(0, world_size, train_cfg, run_dir)
