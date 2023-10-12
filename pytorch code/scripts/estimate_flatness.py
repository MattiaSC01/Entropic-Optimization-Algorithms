import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns
import pandas as pd

import wandb
import os

import sys
sys.path.append(os.pat.abspath('..'))
from Autoencoder import *


device = 'cuda:2' if torch.cuda.is_available() else 'cpu'                   # set device
print(f'Using device {device}.')

path = "/home/pittorino/mattia/data/replicated"                               # choose folder
print(f"Saving to directory {path}")

wb = True                                                                   # if True, log to wandb
project = 'replicated_schedule'                                             # project in wandb
log_interval = 50                                                           # pass to wandb.watch to track gradients

# login to wandb (force relogin)
if wb:
    print(f"Logging to wandb project {project}.")
    wandb.login(key="a8eec0fe99318c31fadcd8a90cce23d57a642ddb", relogin=True, force=False)   # pittorino
    # wandb.login(key="aaaa6a0838d0e4f2d3f68ffbc76dcb40602660cf", relogin=True, force=False)   # mattia

# checkpoint to save weights when estimating flatness
chkpt = 'replicated0.pt'
print(f"Saving weights to path {chkpt}")


# data hyperparameters
Ns = [800]                                                            # input dimension
alphas = [0.02, 0.04, 0.08, 0.16]                                     # D / N
betas = [100, 10, 1, 0.3]                                             # P / N
p = 0.5                                                               # Bernoulli parameter for latent binary patterns
sigmas = [ (Sigmoid(5), 'sigmoid5') ]                                 # nonlinearity, name

# bottleneck and final activations
bneck_activations = {'single': nn.ReLU(), 'multiple': nn.ReLU()}
bneck_acts = {'single': 'relu', 'multiple': 'relu'}
final_activations = {'single': nn.Sigmoid(), 'multiple': nn.Sigmoid()}
final_acts = {'single': 'sigmoid', 'multiple': 'sigmoid'}

# architecture and replicas
H = 800                                                               # hidden layer width (multilayer)
bneck_ratios = np.array(                                              # ratio bottleneck / D
    [0.1, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 10.0]
)
y = 5
model_type = f'replicated{y}'                                         # 'vanilla', 'denoising{std}', or 'replicated{y}'

std = 0.0
lr = 0.0001

# focusing shcedules conditioned on beta (gamma0, gamma1, gamma_freq)
schedules = {
    100: [(0.001, 0.06, 1), (0.001, 0.05, 1), (0.001, 0.04, 1), (0.001, 0.03, 1)],
    10: [(0.001, 0.06, 10), (0.001, 0.05, 10), (0.001, 0.04, 10), (0.001, 0.03, 10)],
    1: [(0.001, 0.06, 100), (0.001, 0.05, 100), (0.001, 0.04, 100), (0.001, 0.03, 100)],
    0.3: [(0.001, 0.06, 340), (0.001, 0.05, 340), (0.001, 0.04, 340), (0.001, 0.03, 340)],
}

# training hyperparameters
l2_rates = [1e-2]                                                     # weight decay parameter
epochss = {100: 100, 10: 1000, 1: 10000, 0.3: 34000}                  # max epochs conditioned on beta (for convenience, should be on P)
batch_size = 128                                                      # batch size
patience = None                                                       # patience (epochs) for early stopping. can be None
improvement = None                                                    # what constitutes an improvement in early stopping
zero = 0.0                                                            # stop training when RE hits this value
verbose = 0                                                           # pass to train_loop_ae
T = 2                                                                 # train each model T times

# flatness/denoising RE
weight_iters = 20                                                      # independent perturbations to estimate flatness
input_iters = 20                                                       # independent perturbations to estimate denoising RE
val_dim = 80000                                                        # samples in flatness data
test_dim = 5000                                                        # test set
num_evaluations = 20

# stddevs for perturbation of weights and inputs
input_noise = list(np.linspace(0.0, 0.5, 21))
weight_noise = [0.0, 0.0001, 0.0005] + list(np.linspace(0.001, 0.01, 10)) + \
                           list(np.linspace(0.015, 0.05, 8)) + list(np.linspace(0.06, 0.1, 5))

for _ in range(T):
    for sigma, name in sigmas:
        for N in Ns:
            for alpha in alphas:

                # if alpha is too small to have integer D, go the next
                if alpha * N < 1:
                    continue

                for beta in betas:

                    # data hyperparameters
                    D = int(alpha * N)
                    P = int(beta * N)
                    print(f"Set N = {N}, D = {D}, P = {P}. Nonlinearity: {name}")

                    # fix P-dependent parameters
                    epochs = epochss[beta]
                    val_interval = epochs // num_evaluations

                    # generate data and wrap it inside a dataloader
                    data, test_data = hidden_manifold(N, D, P, sigma, p, device, test=test_dim)
                    val_data = data[:val_dim]  # view?
                    train_dataset = MyDataset(data)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    for l2_rate in l2_rates:
                        for model in ['single', 'multiple']:
                            for b in bneck_ratios:
                                for tup in schedules[beta]:

                                    # replicated schedule
                                    gamma0, gamma1, gamma_freq = tup

                                    # Bottleneck width
                                    B = int(b * D)
                                    print(f"{model}, Bottleneck {B}, weight decay {l2_rate}")
                                    print(f"gamma0 = {gamma0}, gamma1 = {gamma1}")

                                    # activations
                                    bneck_activation, bneck_act = bneck_activations[model], bneck_acts[model]
                                    final_activation, final_act = final_activations[model], final_acts[model]

                                    if wb:

                                        hidden = 3 if model == 'multiple' else 1                # number of hidden layers
                                        hidden_width = H if model == 'multiple' else None       # hidden layer width
                                        hidden_act = 'relu' if model == 'multiple' else None    # hidden layer activation
                                        stddev = std if std > 0 else None
                                        val_dimension = min(P, val_dim)                         # number of samples for flatness

                                        config = {
                                            'dataset': 'HM',                                    # hardcoded
                                            'N': N,
                                            'D': D,
                                            'P': P,
                                            'sparsity_latent': p,
                                            'sigma': name,
                                            'model_type': model_type,
                                            'stddev': stddev,
                                            'hidden': hidden,
                                            'H': hidden_width,
                                            'bottleneck': B,
                                            'bneck_act': bneck_act,
                                            'final_act': final_act,
                                            'hidden_act': hidden_act,                           # hardcoded (above)
                                            'l2_rate': l2_rate,
                                            'optimizer': 'AdamW',                               # hardcoded
                                            'lr': lr,
                                            'loss_fn': 'MSE',                                   # hardcoded
                                            'weight_init': 'kaiming_uniform',                   # hardcoded
                                            'epochs': epochs,
                                            'batch_size': batch_size,
                                            'patience': patience,
                                            'improvement': improvement,
                                            'zero': zero,
                                            'weight_iters': weight_iters,
                                            'input_iters': input_iters,
                                            'val_dim': val_dim,
                                            'test_dim': test_dim,
                                            'gamma0': gamma0,
                                            'gamma1': gamma1,
                                        }

                                        run = wandb.init(
                                            project=project,
                                            config=config,
                                            reinit=True,
                                            save_code=True,
                                            entity='torchnn',   # wandb team
                                        )

                                    replicas = []
                                    optimizers = []

                                    for i in range(y + 1):

                                        # instantiate AE
                                        if model == 'single':
                                            encoder = nn.Sequential(*makelist([N, B], [bneck_activation], [nn.init.kaiming_uniform_]))
                                            decoder = nn.Sequential(*makelist([B, N], [final_activation], [nn.init.kaiming_uniform_]))
                                        elif model == 'multiple':
                                            encoder = nn.Sequential(*makelist([N, H, B], [nn.ReLU(), bneck_activation], [nn.init.kaiming_uniform_, nn.init.kaiming_uniform_]))
                                            decoder = nn.Sequential(*makelist([B, H, N], [nn.ReLU(), final_activation], [nn.init.kaiming_uniform_, nn.init.kaiming_uniform_]))
                                        ae = AE(encoder, decoder)
                                        ae.to(device)

                                        # create y replicas and 1 baricenter (tester)
                                        if i == 0:
                                            tester = ae
                                        else:
                                            replicas.append(ae)
                                            # associate an independent optimizer to each replica
                                            optimizer = torch.optim.AdamW(ae.parameters(), weight_decay=l2_rate)
                                            optimizers.append(optimizer)

                                    # select loss
                                    loss_fn = nn.MSELoss()

                                    # regularly log gradients and parameter magnitude to wandb
                                    if wb:
                                        wandb.watch(replicas[0], log_freq=log_interval, log='all')

                                    metrics = train_loop_replicated(train_dataloader, replicas, tester, loss_fn, optimizers,
                                                                    device=device, epochs=epochs, gamma0=gamma0, gamma1=gamma1,
                                                                    stddev=std, val_data=val_data,
                                                                    flatness=True, weight_iters=weight_iters, input_iters=input_iters,
                                                                    weight_noise=weight_noise, input_noise=input_noise, chkpt=chkpt,
                                                                    verbose=verbose, patience=patience, improvement=improvement,
                                                                    log_frequency=25, val_frequency=val_interval, zero=zero,
                                                                    labels=False, wb=wb, gamma_freq=gamma_freq, test_data=test_data)

                                    if wb:
                                        run.finish()
