import os

import torch
from torch import multiprocessing
import yaml

from crowd_search import trainer


_NUM_LEARNERS = 2
_NUM_AGENTS = 3  # per learner node


def train(cfg: dict, local_rank: int, num_learners: int, num_explorers: int) -> None:

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
            "gloo", world_size=(num_learners + 1) * num_explorers, rank=local_rank
        )

    if not is_explorer:
        trainer_node = trainer.Trainer(
            models_cfg=cfg.gete("models"),
            agent_nodes=
            device=device
        )


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
        (config, _NUM_LEARNERS, _NUM_AGENTS),
        nprocs=(_NUM_LEARNERS + 1) * _NUM_AGENTS,
        join=True,
    )
