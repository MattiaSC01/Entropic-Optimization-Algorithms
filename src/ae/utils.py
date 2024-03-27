import torch
from torch import nn
import numpy as np
import random
import hashlib
import os
import sys


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def compute_md5(file_path):
    with open(file_path, 'rb') as file:
        md5_hash = hashlib.md5()
        for chunk in iter(lambda: file.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


class Sigmoid(nn.Module):
    """
    implement the function x -> sigmoid(k * x) = 1 / (1 + exp(-k * x))
    """
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return torch.sigmoid(self.k * x)


def clean_wandb():
    """
    Remove all subdirectories in the wandb directory whose name is not "latest-run".
    """
    import os, shutil
    wandb_dir = "wandb"
    assert os.path.exists(wandb_dir), f"wandb directory not found: {wandb_dir}"
    assert os.path.isdir(wandb_dir), f"not a directory: {wandb_dir}"
    for file in os.listdir(wandb_dir):
        if not os.path.isdir(os.path.join(wandb_dir, file)) or file == "latest-run":
            continue
        shutil.rmtree(os.path.join(wandb_dir, file))


class SuppressStdout:
    """
    Context manager to suppress standard output.
    Does not work with jupyter notebooks.
    """
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self.original_stdout
        return False # propagate exceptions
