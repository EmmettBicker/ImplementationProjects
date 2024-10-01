import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import wraps

class VirtualGPU:
    def __init__(self, num_virtual_gpus):
        self.num_virtual_gpus = num_virtual_gpus
        self.original_device_count = torch.cuda.device_count

    def __enter__(self):
        torch.cuda.device_count = lambda: self.num_virtual_gpus
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.device_count = self.original_device_count

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

class DistributedRunner:
    def __init__(self, fn, num_virtual_gpus=None):
        self.fn = fn
        self.num_virtual_gpus = num_virtual_gpus

    def run_distributed(self, rank, world_size, *args):
        setup(rank, world_size)
        self.fn(rank, world_size, *args)
        cleanup()

    def __call__(self, *args, **kwargs):
        if self.num_virtual_gpus is not None:
            with VirtualGPU(self.num_virtual_gpus):
                world_size = torch.cuda.device_count()
                mp.spawn(self.run_distributed, args=(world_size,) + args, nprocs=world_size, join=True)
        else:
            world_size = torch.cuda.device_count()
            mp.spawn(self.run_distributed, args=(world_size,) + args, nprocs=world_size, join=True)

def distribute(num_virtual_gpus=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            runner = DistributedRunner(fn, num_virtual_gpus)
            runner(*args, **kwargs)
        return wrapper
    return decorator

# Example usage
@distribute(num_virtual_gpus=4)
def train(rank, world_size, num_epochs):
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Your training loop here
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} on GPU {rank}")
        # Add your actual training code here

if __name__ == "__main__":
    mp.set_start_method('spawn')
    train(num_epochs=5)