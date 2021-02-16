#!/usr/bin/env python3

import argparse
import datetime
import pathlib

import torch
from torch.distributed import rpc
from torch import distributed
from torch import multiprocessing
import yaml

from crowd_search import trainer

_LOG_DIR = pathlib.Path("~/runs/crowd-search").expanduser()


def _get_learner_explorers(learner_rank, num_explorers, num_learners):
    """Helper function to assign explorer process ranks to trainers."""
    return list(
        range(
            (learner_rank * num_explorers) + num_learners,
            (learner_rank * num_explorers) + num_learners + num_explorers,
        )
    )


torch.set_num_threads(1)


def train(
    local_rank: int, cfg: dict, num_learners: int, num_explorers: int, world_size: int
) -> None:
    """Main entrypoint function

    Args:
        local_rank: The rank of this process relative to the world.
        cfg: The configuration dictionary governing this experiment.
        num_learners: How many learning nodes are present.
        num_explorers: How many exploration nodes _per trainer_ there are.
        world_size: The total number of processes.
    """
    # Check if cuda is available and if this is an agent node which will be
    # learning.
    use_cuda = torch.cuda.is_available()
    is_explorer = True
    device = torch.device("cpu")
    if use_cuda and local_rank < num_learners and local_rank < num_learners:
        torch.cuda.set_device(local_rank)
        is_explorer = False
        device = torch.device(f"cuda:{local_rank}")

    train_cfg = cfg.get("training")
    num_learners = train_cfg.get("num-learners")
    num_explorers = train_cfg.get("num-explorers")

    learner_explorer_groups = {
        learner_rank: _get_learner_explorers(learner_rank, num_explorers, num_learners)
        for learner_rank in range(num_learners)
    }
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method="tcp://localhost:29501", rpc_timeout=2400
    )

    if not is_explorer:
        # Create process to distributed model training across trainer processes.
        distributed.init_process_group(
            "gloo",
            rank=local_rank,
            world_size=num_learners,
            init_method="tcp://localhost:29500",
        )
        rpc.init_rpc(
            name=f"Trainer:{local_rank}",
            rank=local_rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        trainer_node = trainer.Trainer(
            cfg=cfg,
            models_cfg=cfg.get("models"),
            device=device,
            num_learners=num_learners,
            explorer_nodes=learner_explorer_groups[local_rank],
            rank=local_rank,
            run_dir=_LOG_DIR
            / datetime.datetime.now().isoformat().split(".")[0].replace(":", "."),
            batch_size=cfg["training"]["batch-size"]
        )
        trainer_node.continous_train()
    else:
        # If not a training node, wait to launch explorers within the training nodes
        # object so the pipline can be established.
        rpc.init_rpc(
            name=f"Explorer:{local_rank}",
            rank=local_rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--config_path",
        type=pathlib.Path,
        default=pathlib.Path("crowd_search/config.yaml"),
    )
    args = parser.parse_args()

    # Read config
    cfg = yaml.safe_load(args.config_path.read_text())
    train_cfg = cfg.get("training")
    num_learners = train_cfg.get("num-learners")
    num_explorers = train_cfg.get("num-explorers")

    world_size = num_learners * num_explorers + num_learners

    multiprocessing.spawn(
        train,
        (cfg, num_learners, num_explorers, world_size),
        nprocs=world_size,
        join=True,
    )
