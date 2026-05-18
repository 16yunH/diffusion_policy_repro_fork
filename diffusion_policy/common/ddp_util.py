"""Utilities for DistributedDataParallel training.

All functions are safe to call in non-DDP (single-GPU) mode — they return
sensible defaults and no-ops when torch.distributed is not initialized.
"""
import os
import torch
import torch.distributed as dist


def is_ddp_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_ddp_enabled():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_ddp_enabled():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def init_ddp():
    """Initialize DDP process group from torchrun environment variables.

    No-op when not launched via torchrun (LOCAL_RANK not set).
    Returns the local rank if DDP was initialized, None otherwise.
    """
    if 'LOCAL_RANK' not in os.environ:
        return None
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return local_rank


def get_device(cfg_device: str) -> torch.device:
    """Resolve the correct device for the current rank."""
    if is_ddp_enabled():
        return torch.device('cuda', get_rank())
    return torch.device(cfg_device)


def wrap_ddp(model, find_unused_parameters=False):
    """Wrap model with DistributedDataParallel. No-op in non-DDP mode."""
    if not is_ddp_enabled():
        return model
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[get_rank()],
        find_unused_parameters=find_unused_parameters,
    )


def get_ddp_model(model):
    """Return the underlying model, unwrapping DDP if needed."""
    if is_ddp_enabled() and isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def prepare_dataloader(dataset, **dataloader_kwargs):
    """Create a DataLoader with DistributedSampler when DDP is enabled."""
    from torch.utils.data import DataLoader, DistributedSampler

    shuffle = dataloader_kwargs.pop('shuffle', True)
    sampler = None
    if is_ddp_enabled():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
        )
        shuffle = False

    return DataLoader(
        dataset,
        sampler=sampler,
        shuffle=shuffle,
        **dataloader_kwargs,
    )


def barrier():
    """Synchronize all processes. No-op in non-DDP mode."""
    if is_ddp_enabled():
        dist.barrier()
