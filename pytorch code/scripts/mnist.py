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
import shutil
import os

import sys
sys.path.append(os.pat.abspath('..'))
from Autoencoder import *


device = 'cuda:2' if torch.cuda.is_available() else 'cpu'  # set device
print(f'Using device {device}.')

path = "/Data/pittorino/mattia/mnist/"  # choose folder
print(f"Saving to directory {path}")

wb = True  # if True, log to wandb
project = 'Mnist'

os.makedirs(path, exist_ok=True)  # root folder for experiment results
os.makedirs(os.path.join(path, 'Data'), exist_ok=True)  # folder for synthetic data
os.makedirs(os.path.join(path, 'Weights'), exist_ok=True)  # folder for optimized weights
os.makedirs(os.path.join(path, 'Diagnostics'), exist_ok=True)  # folder for various graphs
os.makedirs(os.path.join(path, 'Dfs'), exist_ok=True)  # folder for dataframes

N = 784
D = 14   # estimated
P = 70000


bneck_activation = nn.Sigmoid()  # bottleneck activation function
final_activation = Sigmoid(10)  # final activation function

l2_rates = [1e-6, 1e-7, 1e-8, 1e-9, 0.0]  # l2 regularization importance
bneck_ratios = np.array([0.5, 1.0, 1.5, 2.0, 3.0])  # B / D

epochs = 100  # max epochs
batch_size = 512  # batch size
patience = 5  # patience (epochs) for early stopping. can be None
improvement = 1e-5  # what constitutes an improvement in early stopping
zero = 1e-6  # stop training when RE hits this value
# T = 1                                                                 # times each hyperparameter combination is tried (warning: T>1 not supported!)
verbose = 1  # pass to train_loop_ae

# generate bottleneck widths to try as function of D based on bneck_ratios
bnecks = {}
bnecks[D] = (D * bneck_ratios).astype('int32')

# plotting
dpi = 100
w = 16
h = 8

# login to wandb
if wb:
    wandb.login(key="aaaa6a0838d0e4f2d3f68ffbc76dcb40602660cf", relogin=True)

# watch gradients (wandb)
log_freq = 10


# load mnist dataset
data = load_mnist()
data = data.to(device)

data_id = f"mnist_P={P}"
data_title = f"mnist, P = {P}"

# Wrap the data inside a DataLoader object
train_dataset = MyDataset(data)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for model in ['single', 'multiple']:
    errors = {}
    print(model, ':')

    for B in bnecks[D]:
        temp = {}
        print(f'B = {B}:')

        for l2_rate in l2_rates:
            print(f"l2 = {l2_rate}")

            # strings to save plots
            id = model + f"_mnist_P={P}_B={B}_l2={l2_rate}"  # append to path
            title = model + f" mnist, P = {P}, B = {B}, l2 = {l2_rate}"  # append to title

            if wb:
                hidden = 3 if model == 'multiple' else 1  # number of hidden layers (counting bottleneck)

                config = {
                    'dataset': 'MNIST',
                    'hidden': hidden,
                    'P': P,
                    'l2_rate': l2_rate,
                    'bottleneck': B,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'patience': patience,
                    'improvement': improvement,
                    'zero': zero,
                }

                run = wandb.init(
                    project=project,
                    config=config,
                    reinit=True,
                )

            # initialize AE
            if model == 'single':
                encoder = MLP([N, B], [bneck_activation], [nn.init.xavier_uniform_])
                decoder = MLP([B, N], [final_activation], [nn.init.xavier_uniform_])
            elif model == 'multiple':
                encoder = MLP([N, N, B], [nn.ReLU(), bneck_activation],
                              [nn.init.xavier_uniform_, nn.init.xavier_uniform_])
                decoder = MLP([B, N, N], [nn.ReLU(), final_activation],
                              [nn.init.xavier_uniform_, nn.init.xavier_uniform_])
            else:
                raise Exception()
            ae = AE(encoder, decoder)
            ae.to(device)

            # log gradients to wandb every log_freq batches
            if wb:
                wandb.watch(ae, log_freq=log_freq, log='all')

            # select loss and optimizer
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(ae.parameters(), amsgrad=True)

            # train AE
            losses, l2s = train_loop_ae(train_dataloader, ae, loss_fn, optimizer, device=device,
                                        epochs=epochs, patience=patience, improvement=improvement,
                                        zero=zero, l2_rate=l2_rate, verbose=verbose, labels=False, wb=wb)

            # save optimized weights
            filename = 'weights_' + id + '.pt'
            torch.save(ae.state_dict(), os.path.join(path, 'Weights', filename))

            # save to wandb. Use a workaround to avoid needing permission for symlink when calling
            # wandb.save: copy the file that must be logged to wandb inside the run directory of wandb.
            if wb:
                copy_path = os.path.join(wandb.run.dir, filename)
                shutil.copy(os.path.join(path, 'Weights', filename), copy_path)
                wandb.save(copy_path)

            # plot training loss
            fig = plt.figure(figsize=(w, h), dpi=dpi)
            plt.plot(losses)
            plt.title('Training RE - ' + title)
            filename = 'loss_' + id + '.png'
            plt.savefig(os.path.join(path, 'Diagnostics', filename), dpi=dpi)
            plt.close(fig)

            # plot average l2 norm per weight during training
            fig = plt.figure(figsize=(w, h), dpi=dpi)
            plt.plot(l2s)
            plt.title('l2 norm per weight during training - ' + title)
            filename = 'l2_' + id + '.png'
            plt.savefig(os.path.join(path, 'Diagnostics', filename), dpi=dpi)
            plt.close(fig)

            # set up plot of final weight distributions
            ncols = 2 if model == 'multiple' else 1
            fig, ax = plt.subplots(nrows=2, ncols=ncols, figsize=(16, 8))
            fig.suptitle('weights ' + title)

            # encoder
            i = 0
            for layer in ae.encoder.layers:
                try:
                    if model == 'single':
                        current = ax[0]
                    elif model == 'multiple':
                        current = ax[0, i]
                    else:
                        raise Exception('model should be either \'single\' or \'multiple\'')
                    kernel = layer.weight.detach().cpu()
                    current.hist(kernel.reshape(-1), bins=50)
                    current.set_title(f'encoder {i}')
                    i += 1
                except:
                    pass

            # decoder
            i = 0
            for layer in ae.decoder.layers:
                try:
                    if model == 'single':
                        current = ax[1]
                    elif model == 'multiple':
                        current = ax[1, i]
                    else:
                        raise Exception('model should be either \'single\' or \'multiple\'')
                    kernel = layer.weight.detach().cpu()
                    current.hist(kernel.reshape(-1), bins=50)
                    current.set_title(f'decoder {i}')
                    i += 1
                except:
                    pass

            # save figure
            filename = "weights_" + id + ".png"
            plt.savefig(os.path.join(path, 'Diagnostics', filename), dpi=dpi)
            plt.close(fig)

            # save to wandb. Use a workaround to avoid needing permission for symlink in wandb.save.
            # copy the file you want to log to wandb inside the run directory of wandb.
            if wb:
                copy_path = os.path.join(wandb.run.dir, filename)
                shutil.copy(os.path.join(path, 'Diagnostics', filename), copy_path)
                wandb.save(copy_path)

            # free memory
            del kernel

            # compute activations of the last layer
            with torch.no_grad():
                z = ae.encoder(data[:100])
                for layer in ae.decoder.layers[:-1]:
                    z = layer(z)

            # plot activations of the last layer
            fig = plt.figure(figsize=(w, h), dpi=dpi)
            plt.hist(z.detach().cpu().reshape(-1), bins=50, density=True)
            plt.title('Sigmoid activations last layer - ' + title)

            # save figure
            filename = 'activations_' + id + '.png'
            plt.savefig(os.path.join(path, 'Diagnostics', filename), dpi=dpi)
            plt.close(fig)

            # save to wandb. Use a workaround to avoid needing permission for symlink in wandb.save.
            # copy the file you want to log to wandb inside the run directory of wandb.
            if wb:
                copy_path = os.path.join(wandb.run.dir, filename)
                shutil.copy(os.path.join(path, 'Diagnostics', filename), copy_path)
                wandb.save(copy_path)

            # free memory
            del z

            # save final loss with given l2_rate
            temp[str(l2_rate)] = losses[-1]
            print(f"done in {len(losses)} epochs. Achieved ", "{:.7f} RE.".format(losses[-1]))

            if wb:
                run.finish()

        # save losses obtained with given bottleneck width
        errors[B] = temp

    # create a pandas df to store final REs
    errors_by_rate = {}
    for l2_rate in l2_rates:
        errors_by_rate[str(l2_rate)] = []
    for B in bnecks[D]:
        for l2_rate in l2_rates:
            errors_by_rate[str(l2_rate)].append(errors[B][str(l2_rate)])
    reconstruction_errors = pd.DataFrame(data=errors_by_rate, index=bnecks[D], dtype=float)
    filename = model + '_' + data_id + '.csv'
    reconstruction_errors.to_csv(os.path.join(path, 'Dfs', filename))

    # plot best RE as function of bottleneck width B
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    reconstruction_errors = pd.read_csv(os.path.join(path, 'Dfs', filename), index_col=0)
    x_axis = bnecks[D]
    y = []
    for B in bnecks[D]:
        best = np.infty
        for l2_rate in l2_rates:
            if reconstruction_errors.loc[B, str(l2_rate)] < best:
                best = reconstruction_errors.loc[B, str(l2_rate)]
        y.append(best)
    plt.plot(x_axis, y, 'or')
    plt.plot(x_axis, y, '-', label=('RE ' + model))

    # adjust and save
    plt.legend()
    plt.xticks(x_axis)
    plt.title(f"RE vs Bottleneck {model} - " + data_title)
    plt.xlabel('Bottleneck dimension')
    plt.ylabel(f'best RE')
    filename = model + '_' + data_id + '.png'
    plt.savefig(os.path.join(path, filename), dpi=dpi)
    plt.close(fig)

# load pandas dfs and plot comparison of single and multiple
fig = plt.figure(figsize=(w, h), dpi=dpi)
for model in ['single', 'multiple']:
    filename = model + '_' + data_id + ".csv"
    reconstruction_errors = pd.read_csv(os.path.join(path, 'Dfs', filename), index_col=0)
    x_axis = bnecks[D]
    y = []
    for B in bnecks[D]:
        best = np.infty
        for l2_rate in l2_rates:
            if reconstruction_errors.loc[B, str(l2_rate)] < best:
                best = reconstruction_errors.loc[B, str(l2_rate)]
        y.append(best)
    plt.plot(x_axis, y, 'or')
    plt.plot(x_axis, y, '-', label=('RE ' + model))

# adjust and save
plt.xticks(x_axis)
plt.xlabel('Bottleneck dimension')
plt.ylabel(f'best RE')
plt.title(f"RE vs Bottleneck width - " + data_title)
plt.legend()
filename = "comparison_" + data_id + ".png"
plt.savefig(os.path.join(path, filename), dpi=dpi)
plt.close(fig)
