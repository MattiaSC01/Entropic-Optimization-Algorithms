import cProfile
import pstats
import contextlib


# prepare the environment
import torch
from torch.utils.data import DataLoader
from ae.utils import set_seed
from ae.dataset import load_mnist
from ae.model import AutoEncoder
from ae.evaluation import Eval
import json
import numpy as np
chkpt_dir = "checkpoints/mnist-train"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, chkpt_metadata = AutoEncoder.from_pretrained(chkpt_dir, device)
data, dataset_metadata = load_mnist(root="../data", log_to_wandb=False)
set_seed(42)
train_size = chkpt_metadata["hyperparameters"]["train_size"]
test_size = chkpt_metadata["hyperparameters"]["test_size"]
test_data = data[train_size:train_size+test_size]
test_loader = DataLoader(test_data, batch_size=chkpt_metadata["hyperparameters"]["batch_size"], shuffle=False)
print("Environment is ready.")

def profile_flatness_profile():
    noise_strengths = np.linspace(0.0, 0.1, 10)
    losses = Eval.flatness_profile(model, test_loader, noise_strengths, n_iters=2)


# Profile the function
cProfile.runctx("profile_flatness_profile()", globals(), locals(), "profile_stats.prof")

print("Profiling is done.")

# Load the stats
stats = pstats.Stats("profile_stats.prof")

# Load the stats and redirect output to a file when printing
with open("profile_output.txt", "w") as f:
    with contextlib.redirect_stdout(f):
        stats = pstats.Stats("profile_stats.prof")
        stats.strip_dirs().sort_stats("time").print_stats()

print("Profiling results are saved in profile_output.txt")
