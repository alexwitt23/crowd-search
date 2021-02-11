from typing import List

from torch import distributed
from torch.distributed import rpc


def broadcast_models(models: List[dict], process_group, src_node: int):
    """Common function for explorers and trainers to enter so that
    the latest trained weights can be broadcasted out to the explorers."""

    distributed.broadcast_object_list(models, src=src_node, group=process_group)
    distributed.barrier()

    return models


def collate_explorations(
    explorations, output, process_group, src_node: int,
):
    if explorations:
        distributed.gather_object(explorations, dst=src_node, group=process_group)
    else:
        distributed.gather_object(
            explorations, output, dst=src_node, group=process_group
        )

    distributed.barrier()

    return explorations


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)
