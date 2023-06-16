import os
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from torch.distributed import init_process_group


def prepare_distributed_sampler(
    dataset: Dataset,
    rank: int,
    world_size: int,
    batch_size: int = 32,
    pin_memory: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Returns a DistributedSampler wrapped within a DataLoader for easy iterations in a
    multiprocessing environment.

    Args:
        dataset (Dataset): your original dataset
        rank (int): rank of current process
        world_size (int): world size available to the model
        batch_size (int, optional): batch size of the dataloader. Defaults to 32.
        pin_memory (bool, optional): Whether to use pin memory. Defaults to False.
        num_workers (int, optional): num_workers for DataLoader. Defaults to 0.

    Returns:
        DataLoader: final DataLoader for the dataset
    """
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )
    return dataloader


def ddp_setup(backend="nccl"):
    init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # os.environ["NCCL_SOCKET_IFNAME"] = "eno1"
    # debugging
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    # os.environ["HYDRA_FULL_ERROR"] = "1"
