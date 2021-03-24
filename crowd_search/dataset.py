"""A dataset that will hold the explored history. There isn't much to this dataset,
really, it might be removed in the future. The key part is the collation function
which allows us to batch together the different collections of input data."""

import pathlib
import time
from typing import Any, Dict, List
import uuid
import shutil

import torch
from torch.utils import data


class Dataset(data.Dataset):
    """Dataset class."""

    def __init__(self, save_dir: pathlib.Path, policy: Any) -> None:
        """initialize"""
        super().__init__()

        self.gamma = 0.99
        self.save_dir = save_dir
        self.items = []
        self.policy = policy
        self.save_dir_nested = None

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.items)

    def __getitem__(self, idx: int):
        """Retrieve an item from the dataset and prepare it for training."""
        # Retrieve the game from the list of available games.
        return torch.load(self.items[idx], map_location="cpu")

    def update(
        self, game_histories: List[Dict[str, torch.Tensor]], idx: int, is_main: bool
    ):
        """Take in new data, clear what was previous in the save_dir and write to cache
        directory."""
        self.save_dir_nested = self.save_dir / f"{idx}"
        self.save_dir_nested.mkdir(exist_ok=True)

        previous_save_dir = self.save_dir / f"{idx - 2}"
        if previous_save_dir.is_dir() and is_main:
            shutil.rmtree(previous_save_dir)

        for game_history in game_histories:
            for data_item in game_history:
                torch.save(data_item, self.save_dir_nested / f"{uuid.uuid4()}")

    def prepare_for_epoch(self) -> None:
        self.items = list(self.save_dir_nested.glob("*"))

    def collate(self, batches):
        """This function takes in a list of data from the various data loading
        threads and combines them all into one batch for training."""
        output = {key: [] for key in self.policy.data_keys}

        for data_batch in batches:
            for key in output:
                output[key].append(data_batch[key])

        for key in output:
            output[key] = torch.stack(output[key])

        return output
