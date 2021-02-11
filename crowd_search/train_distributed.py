import argparse
import pathlib
import os

import gym
import torch
from torch import distributed
from torch import multiprocessing
import yaml

from crowd_search import config
from crowd_search import policy
from crowd_search import trainer
from crowd_search import explorer_distributed
from third_party.crowd_sim.envs.utils import agent


_NUM_LEARNERS = 2
_NUM_AGENTS = 3  # per learner node


def _get_learner_explorers(learner_rank, num_explorers, num_learners):
    return list(
        range(
            (learner_rank * num_explorers) + num_learners,
            (learner_rank * num_explorers) + num_learners + num_explorers,
        )
    )


def _find_group(explorer_rank, learner_explorer_groups):
    for learner_rank, explorers in learner_explorer_groups.items():
        if explorer_rank in explorers:
            return learner_rank


def train(local_rank: int, cfg: dict, num_learners: int, num_explorers: int,) -> None:

    # Check if cuda is available and if this is an agent node which will be
    # learning.
    use_cuda = torch.cuda.is_available()
    is_explorer = True
    device = torch.device("cpu")
    if use_cuda and local_rank < num_learners:
        torch.cuda.set_device(local_rank)
        is_explorer = False
        device = torch.device(f"cuda:{local_rank}")

    is_main = local_rank == 0
    # If we are using distributed training, initialize the backend through which process
    # can communicate to each other.
    if not (num_learners == num_explorers == 1):
        torch.distributed.init_process_group(
            "gloo",
            world_size=num_learners * num_explorers + num_learners,
            rank=local_rank,
        )
    learner_explorer_groups = {
        learner_rank: _get_learner_explorers(learner_rank, _NUM_AGENTS, _NUM_LEARNERS)
        for learner_rank in range(num_learners)
    }

    groups = {
        learner_node: distributed.new_group([learner_node] + explorers, backend="gloo")
        for learner_node, explorers in learner_explorer_groups.items()
    }

    if not is_explorer:

        trainer_node = trainer.Trainer(
            models_cfg=cfg.get("models"),
            device=device,
            process_group=groups[local_rank],
            num_explorers=num_explorers,
        )
        trainer_node.continous_train()
    else:
        learner_node = _find_group(local_rank, learner_explorer_groups)
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
        e = explorer_distributed.Explorer(
            environment,
            sim_robot,
            robot_policy,
            gamma=0.9,
            process_group=groups[learner_node],
            learner_idx=learner_node,
        )
        e.continuous_play()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--config_path",
        type=pathlib.Path,
        default=pathlib.Path("crowd_search/config.yaml"),
    )
    args = parser.parse_args()

    # Read config
    train_cfg = yaml.safe_load(args.config_path.read_text())

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"

    multiprocessing.spawn(
        train,
        (train_cfg, _NUM_LEARNERS, _NUM_AGENTS),
        nprocs=_NUM_LEARNERS * _NUM_AGENTS + _NUM_LEARNERS,
        join=True,
    )
