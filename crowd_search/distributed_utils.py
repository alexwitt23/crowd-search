"""Central file to store utilities that are shared across distributed
processing."""

from typing import Union

from torch import nn
from torch.nn import parallel


def unwrap_ddp(model: Union[nn.Module, parallel.DistributedDataParallel]) -> nn.Module:
    """Take in a model and unwrap it if it's a DPP object."""
    if isinstance(model, parallel.DistributedDataParallel):
        model = model.module

    return model
