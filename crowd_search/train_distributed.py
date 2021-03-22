#!/usr/bin/env python3
"""Main script to run and start training job."""

import argparse
import datetime
import pathlib
import shutil
import tempfile
from typing import List

import torch
from torch.distributed import rpc
from torch import distributed
from torch import multiprocessing
import yaml

from crowd_search import trainer

_LOG_DIR = pathlib.Path("~/runs/crowd-search").expanduser()
torch.set_num_threads(1)


def _get_learner_explorers(
    learner_rank: int, num_explorers: int, num_learners: int
) -> List[int]:
    """Helper function to assign explorer process ranks to trainers."""
    return list(
        range(
            (learner_rank * num_explorers) + num_learners,
            (learner_rank * num_explorers) + num_learners + num_explorers,
        )
    )


def train(
    local_rank: int,
    cfg: dict,
    num_learners: int,
    num_explorers: int,
    world_size: int,
    cache_dir: pathlib.Path,
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
    is_explorer = False
    is_shared_storage = False
    device = torch.device("cpu")
    if local_rank < num_learners:
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
    elif num_learners <= local_rank < num_learners + (num_explorers * num_learners):
        is_explorer = True
        torch.cuda.set_device(-1)
    elif local_rank >= num_learners + (num_explorers * num_learners):
        is_shared_storage = True
        torch.cuda.set_device(-1)

    learner_explorer_groups = {
        learner_rank: _get_learner_explorers(learner_rank, num_explorers, num_learners)
        for learner_rank in range(num_learners)
    }
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method="tcp://localhost:29501", rpc_timeout=300
    )

    if not is_explorer and not is_shared_storage:
        storage_node = num_learners + num_learners * num_explorers + local_rank
        # Create process to distributed model training across trainer processes.
        if num_learners > 1:
            distributed.init_process_group(
                "nccl",
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
        run_dir = _LOG_DIR / datetime.datetime.now().isoformat().split(".")[0].replace(
            ":", "."
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.dump(cfg))
        print(f"Writting experiment to {run_dir}.")
        trainer.Trainer(
            cfg=cfg,
            device=device,
            num_learners=num_learners,
            explorer_nodes=learner_explorer_groups[local_rank],
            rank=local_rank,
            run_dir=run_dir,
            batch_size=cfg["training"]["batch-size"],
            storage_node=storage_node,
            cache_dir=cache_dir,
        )
    elif is_shared_storage:
        rpc.init_rpc(
            name=f"Storage:{local_rank}",
            rank=local_rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
    else:
        # If not a training node, wait to launch explorers within the training nodes
        # object so the pipline can be established.
        rpc.init_rpc(
            name=f"Explorer:{local_rank}",
            rank=local_rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

    rpc.shutdown(graceful=True)


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

    world_size = num_learners * num_explorers + num_learners + num_learners

    cache_dir = tempfile.TemporaryDirectory().name
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    try:
        multiprocessing.spawn(
            train,
            (cfg, num_learners, num_explorers, world_size, cache_dir),
            nprocs=world_size,
            join=True,
        )
    finally:
        shutil.rmtree(cache_dir)
