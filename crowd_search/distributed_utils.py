from typing import List, Union

from torch import distributed
from torch import nn
from torch.nn import parallel
from torch.distributed import rpc


def unwrap_ddp(model: Union[nn.Module, parallel.DistributedDataParallel]) -> nn.Module:
    if isinstance(model, parallel.DistributedDataParallel):
        model = model.module

    return model
