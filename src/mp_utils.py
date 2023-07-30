import os
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.distributed import init_process_group

from datetime import timedelta
import socket
from contextlib import closing

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

def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def ddp_setup(backend, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    print(f"Creating DDP process group with {backend.upper()} backend on port :{port}")
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    os.environ["NCCL_P2P_DISABLE"] = "1" # required unless ACS is disabled on host. Ref: https://github.com/NVIDIA/nccl/issues/199
    
    # group_gloo = dist.new_group(backend="gloo")
    # if int(os.environ["LOCAL_RANK"]) not in [1]:
    #     dist.monitored_barrier(group=group_gloo, timeout=timedelta(seconds=2))
    
    # debugging
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    print("Created DDP process group")
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["HYDRA_FULL_ERROR"] = "1"


