from typing import Union

from torch import nn
from torch.nn import parallel


def unwrap_ddp(model: Union[nn.Module, parallel.DistributedDataParallel]) -> nn.Module:
    if isinstance(model, parallel.DistributedDataParallel):
        model = model.module

    return model
