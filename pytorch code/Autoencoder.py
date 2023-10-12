import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
from math import ceil

import wandb


# basic AutoEncoder shell. Tolerates different architectures for encoder and decoder.
# l2 regularization and injection of noise on inputs are decided at train time.

class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


# create a sequence of alternating linear layers and activation functions. Initialize weights (biases use
# default init). return a list of layers.

def makelist(
        neurons,                # layer widths, including input and output (e.g. [784, 784, 14])
        activations,            # activations (e.g. [nn.ReLU(), nn.Sigmoid()])
        initializations=None,   # weight initializers in-place (e.g. [nn.init.xavier_uniform_, nn.init.xavier_uniform_])
        bias=True,              # whether to include bias in linear layers
):

    # default initialization
    if initializations is None:
        initializations = len(activations)*[nn.init.xavier_uniform_]

    last = neurons[0]
    layers = []

    for n, act, init in zip(neurons[1:], activations, initializations):

        # instantiate linear layer
        linear = nn.Linear(last, n, bias=bias)

        # initialize weights (does this initialize bias?) - notice initialization in-place
        if init == nn.init.kaiming_uniform_:
            init(linear.weight, nonlinearity='relu')   # only support relu (not leaky_relu) for simplicity
        else:
            init(linear.weight)

        # add layer
        layers.append(linear)
        last = n

        # add activation function
        if act is not None:
            layers.append(act)

    return layers


# add Gaussian noise to model's weights, in-place and magnitude-aware

def add_weight_noise_(
        model,
        stddev=0.0
):

    # get model weights as an Ordered Dictionary
    d = model.state_dict()

    # add noise to each Tensor (magnitude-aware)
    for key in d:
        noise = torch.randn_like(d[key]) * stddev
        d[key] = d[key] * (1 + noise)

    # load updated weights into the model
    model.load_state_dict(d)


def estimate_flatness(
        model,
        loss_fn,
        val_data,
        noise,
        iters,
        out,                # write outputs here
        epoch,
        val_frequency,
        chkpt,              # save model checkpoint here (path with .pt extension)
):
    model.eval()  # eval mode
    n = epoch // val_frequency

    # save current weights
    torch.save(model.state_dict(), chkpt)

    with torch.no_grad():

        # repeatedly perturb model weights starting from initial configuration
        for i in range(iters):
            for s, stddev in enumerate(noise):
                # add noise to model weights
                add_weight_noise_(model, stddev)

                # compute loss with perturbed weights
                loss = loss_fn(model(val_data), val_data.flatten(start_dim=1))
                out[n * iters + i][s] = loss

                # reset weights to their initial values
                model.load_state_dict(torch.load(chkpt))

            # write current epoch
            out[n * iters + i][len(noise)] = epoch

    return out


def estimate_denoising(
        model,
        loss_fn,
        val_data,
        noise,
        iters,
        out,                # write outputs here
        epoch,
        val_frequency,
):
    model.eval()  # eval mode
    n = epoch // val_frequency

    with torch.no_grad():

        # repeatedly denoise noisy inputs
        for i in range(iters):
            for s, stddev in enumerate(noise):

                # add noise
                noisy = val_data + torch.randn_like(val_data) * stddev

                # compute denoising RE
                loss = loss_fn(model(noisy), val_data.flatten(start_dim=1))
                out[n * iters + i][s] = loss

            # write current epoch
            out[n * iters + i][len(noise)] = epoch

    return out


def get_metrics(
        losses,
        noise,              # list of stddevs
        iters,
        epoch,
        val_frequency,
        type,               # 'w' or 'i'
):
    n = epoch // val_frequency
    avgs = losses[n*iters: n*iters + iters-1].mean(dim=0)
    stds = losses[n*iters: n*iters + iters-1].std(dim=0)

    dict = {f"{type}avg{s}": avgs[i] for i, s in enumerate(noise)}
    dict.update({f"{type}std{s}": stds[i] for i, s in enumerate(noise)})

    return dict


# train AE for one epoch. Return average value of loss_fn over batches and final l2 norm of
# weights normalized by number of weights. Apply l2 regularization and possibly inject noise.

def train_ae(
        dataloader,     # dataloader object
        model,          # an instance of the AE class
        loss_fn,        # pytorch loss function - expects reduction='mean'
        optimizer,      # pytorch optimizer - must handle weight decay
        device,
        stddev=0.0,     # intensity of Gaussian noise (train denoising)
        verbose=1,
        log_frequency=25,   # number of batches before printing losses
        labels=False,   # if dataloader spits pairs (input, label), pass True
):
    size = len(dataloader.dataset)   # number of datapoints
    batches = len(dataloader)        # number of batches
    average_error = torch.tensor(0.0, device=device)   # average of loss_fn over batches
    total_params = sum(p.numel() for p in model.parameters())

    model.train()       # enter training mode

    # loop over batches (one epoch)
    for batch, x in enumerate(dataloader):

        # if labels are present, throw them away
        if labels:
            x = x[0]   # assume data has the form (datapoint, label)

        # add Gaussian noise to the inputs for noisy reconstruction (if stddev=0, normal autoencoder)
        with torch.no_grad():
            noisy = x + stddev * torch.randn_like(x)

        # copy the batch on the appropriate device (same as for model!)
        x = x.to(device)
        noisy = noisy.to(device)

        # compute reconstruction error.
        error = loss_fn(model(noisy), x.flatten(start_dim=1))
        loss = error   # let the optimizer handle weight decay

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update average reconstruction error
        with torch.no_grad():
            average_error += error

        # log metrics
        if verbose and batch % log_frequency == 0:

            # compute l2 norm of weights
            with torch.no_grad():
                l2_norm = torch.sqrt(sum([(p ** 2).sum() for p in model.parameters()]))
                l2_norm /= total_params

            # print out
            error, l2_norm, current = error.item(), l2_norm.item(), batch * len(x)
            print(f"error: {error:>7f}   l2 norm per weight: {l2_norm:>7f}   [{current:>5d}/{size:>5d}]")

    # normalize by number of batches (since loss_fn already gives average error in the batch by default)
    average_error = average_error / batches

    # compute l2 norm of weights, normalized by the number of weights
    # WARNING: ALL weights are counted
    with torch.no_grad():
        l2_norm = torch.sqrt(sum([(p ** 2).sum() for p in model.parameters()]))
        total_params = sum(p.numel() for p in model.parameters())
        average_l2 = l2_norm / np.sqrt(total_params)

    # N.B.: before 18/03, l2_norm was normalized by total_params, not by its root.

    return average_error.cpu(), average_l2.cpu()


# train AE calling repeatedly train_ae. Handles early stopping. Saves RE and l2 norm as returned
# by train_ae after each epoch.

def train_loop_ae(
        dataloader,        # dataloader object
        model,             # an instance of the AE class
        loss_fn,           # pytorch loss function - expects reduction='mean'
        optimizer,         # pytorch optimizer
        device,
        epochs,            # max number of epochs. should be multiple of val_frequency.
        stddev=0.0,        # intensity of Gaussian noise
        val_data=None,     # use to estimate flatness and denoising capabilities
        flatness=True,    # whether to estimate flatness and denoising
        weight_iters=3,
        input_iters=3,
        weight_noise=None, # list - use to perturb weights
        input_noise=None,  # list - use to perturb inputs
        chkpt=None,        # path to save model checkpoint to estimate flatness (.pt extension)
        verbose=1,         # 0, 1, 2
        patience=None,     # early stopping
        improvement=None,  # what is an improvement in early stopping?
        zero=0.0,          # stop training if RE goes below this value
        log_frequency=25,  # number of batches before printing losses
        val_frequency=10,  # number of epochs before assessing flatness
        labels=False,      # if dataloader spits pairs (input, label), pass True
        wb=False,          # whether to log to wandb
        test_data=None     # test data
):
    losses = []                       # store reconstruction errors
    l2s = []                          # store l2 norms
    best, count = np.inf, 0           # for early stopping
    flag = 1 if verbose >= 2 else 0   # verbose argument for train_ae

    model.to(device)

    # set up flatness estimation
    if flatness:

        # default stddevs
        if weight_noise is None:
            weight_noise = [0.0, 0.0001, 0.0005] + list(np.linspace(0.001, 0.01, 10)) + \
                           list(np.linspace(0.015, 0.05, 8)) + list(np.linspace(0.06, 0.1, 5))
        if input_noise is None:
            input_noise = list(np.linspace(0.0, 0.5, 21))

        # default val_data. N.B.: Better to pass same data for all runs (default does not!)
        if val_data is None:
            n = len(dataloader.dataset)
            val_data = dataloader.dataset[:n//10]

        # default checkpoint path
        if chkpt is None:
            chkpt = 'perturbation_checkpoint.pt'

        # store results of landscape flatness
        h = (ceil(epochs / val_frequency) + 1) * weight_iters   # one row per flatness profile
        w = len(weight_noise) + 1                               # stddevs, current epoch
        weight_res = torch.zeros((h, w))

        # store results of denoising
        h = (ceil(epochs / val_frequency) + 1) * input_iters
        w = len(input_noise) + 1
        input_res = torch.zeros((h, w))

        if test_data is not None:

            # store results of landscape flatness
            h = (ceil(epochs / val_frequency) + 1) * weight_iters  # one row per flatness profile
            w = len(weight_noise) + 1  # stddevs, current epoch
            test_weight_res = torch.zeros((h, w))

            # store results of denoising
            h = (ceil(epochs / val_frequency) + 1) * input_iters
            w = len(input_noise) + 1
            test_input_res = torch.zeros((h, w))

    # initial metrics (prior to training)
    if wb:
        total_params = sum(p.numel() for p in model.parameters())

        with torch.no_grad():

            # loss
            model.eval()
            loss = loss_fn(model(dataloader.dataset.data), dataloader.dataset.data)

            # l2 norm
            l2_norm = torch.sqrt(sum([(p ** 2).sum() for p in model.parameters()])) / np.sqrt(total_params)

        # log to wandb
        wandb.log({'RE': loss, 'l2': l2_norm, 'epoch': 0})

    # training loop
    for t in range(epochs):

        # estimate flatness and denosing RE
        if flatness and t % val_frequency == 0:
            # compute losses
            weight_res = estimate_flatness(model, loss_fn, val_data, noise=weight_noise, iters=weight_iters,
                                           out=weight_res, epoch=t, val_frequency=val_frequency, chkpt=chkpt)
            input_res = estimate_denoising(model, loss_fn, val_data, noise=input_noise, iters=input_iters,
                                           out=input_res, epoch=t, val_frequency=val_frequency)
            # compute metrics
            weight_dict = get_metrics(weight_res, weight_noise, iters=weight_iters, epoch=t,
                                      val_frequency=val_frequency, type='w')
            input_dict = get_metrics(input_res, input_noise, iters=input_iters, epoch=t,
                                      val_frequency=val_frequency, type='i')
            # log metrics to wandb
            if wb:
                wandb.log(weight_dict, commit=False)
                wandb.log(input_dict, commit=False)

            # do the same for the test data
            if test_data is not None:

                # compute losses
                test_weight_res = estimate_flatness(model, loss_fn, test_data, noise=weight_noise, iters=weight_iters,
                                               out=test_weight_res, epoch=t, val_frequency=val_frequency, chkpt=chkpt)
                test_input_res = estimate_denoising(model, loss_fn, test_data, noise=input_noise, iters=input_iters,
                                               out=test_input_res, epoch=t, val_frequency=val_frequency)
                # compute metrics
                weight_dict = get_metrics(test_weight_res, weight_noise, iters=weight_iters, epoch=t,
                                          val_frequency=val_frequency, type='test_w')
                input_dict = get_metrics(test_input_res, input_noise, iters=input_iters, epoch=t,
                                         val_frequency=val_frequency, type='test_i')
                # log metrics to wandb
                if wb:
                    wandb.log(weight_dict, commit=False)
                    wandb.log(input_dict, commit=False)

        # print out
        if verbose:
            print(f"epoch {t + 1}")

        # test_loss
        if test_data is not None:
            model.eval()
            with torch.no_grad():
                test_loss = loss_fn(test_data, model(test_data))
            if wb:
                wandb.log({'test_RE': test_loss}, commit=False)

        # train for one epoch
        metrics = train_ae(dataloader, model, loss_fn, optimizer, device=device,
                         stddev=stddev, verbose=flag, log_frequency=log_frequency, labels=labels)
        loss, l2 = metrics[0], metrics[1]

        # update loss lists
        losses.append(loss)
        l2s.append(l2)

        # log losses to wandb
        if wb:
            wandb.log({'RE': loss, 'l2': l2, 'epoch': t+1})

        # print out
        if verbose:
            print(f"avg RE: {loss}\n")

        # check if reconstruction error hit 0
        if loss < zero:
            if verbose:
                print("Stop training because RE hit zero")
            break

        # check early stopping
        if patience is not None:
            if loss < best - improvement:
                best = loss
                count = 0
            else:
                count += 1

            # stopping condition
            if count == patience:
                if verbose:
                    print("Stop training because learning plateau-ed")
                break

    # estimate flatness and denoisinge RE at the end of training
    if flatness:

        # test flatness
        if test_data is not None:

            # compute losses
            test_weight_res = estimate_flatness(model, loss_fn, test_data, noise=weight_noise, iters=weight_iters,
                                                out=test_weight_res, epoch=epochs, val_frequency=val_frequency, chkpt=chkpt)
            test_input_res = estimate_denoising(model, loss_fn, test_data, noise=input_noise, iters=input_iters,
                                                out=test_input_res, epoch=epochs, val_frequency=val_frequency)
            # compute metrics
            weight_dict = get_metrics(test_weight_res, weight_noise, iters=weight_iters, epoch=epochs,
                                      val_frequency=val_frequency, type='test_w')
            input_dict = get_metrics(test_input_res, input_noise, iters=input_iters, epoch=epochs,
                                     val_frequency=val_frequency, type='test_i')
            # log metrics to wandb
            if wb:
                wandb.log(weight_dict, commit=False)
                wandb.log(input_dict, commit=False)

        # train flatness
        weight_res = estimate_flatness(model, loss_fn, val_data, noise=weight_noise, iters=weight_iters,
                                       out=weight_res, epoch=epochs, val_frequency=val_frequency, chkpt=chkpt)
        input_res = estimate_denoising(model, loss_fn, val_data, noise=input_noise, iters=input_iters,
                                       out=input_res, epoch=epochs, val_frequency=val_frequency)
        # compute metrics
        weight_dict = get_metrics(weight_res, weight_noise, iters=weight_iters, epoch=epochs,
                                  val_frequency=val_frequency, type='w')
        input_dict = get_metrics(input_res, input_noise, iters=input_iters, epoch=epochs,
                             val_frequency=val_frequency, type='i')
        # log metrics to wandb
        if wb:
            wandb.log(weight_dict, commit=False)
            wandb.log(input_dict)

        # create tables with flatness and denoising during training. columns:
        weight_noise = list(map(lambda x: str(x), weight_noise))
        input_noise = list(map(lambda x: str(x), input_noise))
        # epoch column
        input_noise.append('epoch')
        weight_noise.append('epoch')

        # train tables
        weight_df = pd.DataFrame(weight_res, columns=weight_noise)
        input_df = pd.DataFrame(input_res, columns=input_noise)
        # average flatness grouping by epoch
        avg_weight_df = weight_df.groupby(['epoch']).mean()
        avg_input_df = input_df.groupby(['epoch']).mean()

        # test tables
        if test_data is not None:
            test_weight_df = pd.DataFrame(test_weight_res, columns=weight_noise)
            test_input_df = pd.DataFrame(test_input_res, columns=input_noise)
            # average flatness grouping by epoch
            test_avg_weight_df = test_weight_df.groupby(['epoch']).mean()
            test_avg_input_df = test_input_df.groupby(['epoch']).mean()

        # log tables to wandb
        if wb:
            weight_tbl = wandb.Table(data=weight_df, columns=weight_noise)
            input_tbl = wandb.Table(data=input_df, columns=input_noise)
            avg_weight_tbl = wandb.Table(data=avg_weight_df, columns=weight_noise[:-1])
            avg_input_tbl = wandb.Table(data=avg_input_df, columns=input_noise[:-1])
            wandb.log({'flatness': weight_tbl, 'denoising': input_tbl,
                       'avg_flatness': avg_weight_tbl, 'avg_denoising': avg_input_tbl})
            if test_data is not None:
                test_weight_tbl = wandb.Table(data=test_weight_df, columns=weight_noise)
                test_input_tbl = wandb.Table(data=test_input_df, columns=input_noise)
                test_avg_weight_tbl = wandb.Table(data=test_avg_weight_df, columns=weight_noise[:-1])
                test_avg_input_tbl = wandb.Table(data=test_avg_input_df, columns=input_noise[:-1])
                wandb.log({'test_flatness': test_weight_tbl, 'test_denoising': test_input_tbl,
                           'test_avg_flatness': test_avg_weight_tbl, 'test_avg_denoising': test_avg_input_tbl})

    # return
    if not flatness:
        return (losses, l2s)
    if test_data is None:
        return (losses, l2s, weight_df, input_df)
    return (losses, l2s, weight_df, input_df, test_weight_df, test_input_df)


# super simple class that wraps a torch.Tensor inside a Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Sigmoid activation with arbitrary steepness. Can be used in a network.

class Sigmoid(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return torch.sigmoid(self.k * x)


# load mnist dataset and return a pytorch tensor containing train and test digits, without labels,
# with pixel intensities between 0 and 1.

def load_mnist():

    # load data
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # extract digits
    train = training_data.data
    test = test_data.data

    # flatten digits
    train = train.flatten(start_dim=1)
    test = test.flatten(start_dim=1)

    # merge train and test
    data = torch.cat((train, test), 0)

    # squeeze entries between 0.0 and 1.0
    data = (data / 255.0).to(torch.float32)

    return data


def hidden_manifold(
        N,
        D,
        P,
        sigma=Sigmoid(30),
        p=0.5,
        device='cpu',
        test=None,
):
    F = torch.randn((D, N), device=device)              # feature matrix
    probas = p * torch.ones((P, D), device=device)      # matrix of probabilities for binary patterns
    csi = torch.bernoulli(probas).to(device)            # random binary patterns
    data = torch.matmul(csi, F) / np.sqrt(D)            # compute activations for nonlinearity
    data = sigma(data)                                  # pass projected data through nonlinearity

    if test is None:
        return data

    probas = p * torch.ones((test, D), device=device)  # matrix of probabilities for binary patterns
    csi = torch.bernoulli(probas).to(device)  # random binary patterns
    test_data = torch.matmul(csi, F) / np.sqrt(D)  # compute activations for nonlinearity
    test_data = sigma(test_data)  # pass projected data through nonlinearity
    
    return data, test_data


def train_replicated(
        dataloader,
        replicas,       # list of replicas
        baricenter,     # state_dict (OrderedDict) - same architecture as replicas
        loss_fn,
        optimizers,     # list of optimizer, one for each replica
        device,         # same for all models
        gamma,          # interaction parameter. follows exponential schedule (handled by train_loop)
        kappa=1,        # frequency of elastic updates.
        stddev=0.0,
        verbose=1,
        log_frequency=25,
        labels=False,
):
    size = len(dataloader.dataset)  # number of datapoints
    batches = len(dataloader)       # number of batches
    y = len(replicas)               # number of replicas

    average_losses = torch.zeros(y, device=device)
    total_params = sum(p.numel() for p in replicas[0].parameters())

    # enter training mode
    for replica in replicas:
        replica.train()

    for batch, x in enumerate(dataloader):

        # update baricenter
        if batch % kappa == 0:
            with torch.no_grad():
                # spit pairs (key, tup), where tup has one element per replica, all the elements corresponding to key
                for key, tup in zip(baricenter.keys(), zip(*[replica.parameters() for replica in replicas])):
                    baricenter[key] = sum(tup) / len(tup)

        # if labels are present, throw them away
        if labels:
            x = x[0]  # assume data has the form (datapoint, label)

        # add Gaussian noise to the inputs for noisy reconstruction (if stddev=0, normal autoencoder)
        with torch.no_grad():
            noisy = x + stddev * torch.randn_like(x)

        # copy the batch on the appropriate device (same as for model!)
        x = x.to(device)
        noisy = noisy.to(device)

        # WARNING: for loops are inefficient. Consider parallelizable code in the future (vectorized ops)
        for i in range(y):

            # train i-th replica on current batch
            replica = replicas[i]
            optimizer = optimizers[i]

            # Compute reconstruction error.
            error = loss_fn(replica(noisy), x.flatten(start_dim=1))
            loss = error   # let the optimizer handle weight decay

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update average reconstruction error of current replica
            with torch.no_grad():
                average_losses[i] += error

        # drift towards baricenter in weight space
        if batch % kappa == 0:
            for replica in replicas:
                for key, p in zip(baricenter.keys(), replica.parameters()):
                    p.data = p.data + kappa * gamma * (baricenter[key] - p.data)

        # print out regularly
        if verbose and batch % log_frequency == 0:
            with torch.no_grad():
                # avg l2 norm (normalized by number of weights) of replicas
                avg_l2_norm = sum([torch.sqrt(sum([(p ** 2).sum() for p in replica.parameters()])) / total_params for replica in replicas]) / y
                # avg reconstruction error of replicas on current batch
                avg_error = sum([loss_fn(x.flatten(start_dim=1), replica(x)) for replica in replicas]) / y
                # avg distance (normalized by number of weights) of replicas from baricenter
                avg_distance = sum([torch.sqrt(sum([((baricenter[key] - p)**2).sum() for (key, p) in zip(baricenter.keys(), replica.parameters())])) / total_params for replica in replicas]) / y
            # print out (no need to call .item() on metrics since they are floats here)
            current = batch * len(x)
            print(f"avg error: {avg_error:>7f}   avg l2 norm: {avg_l2_norm:>7f}   avg distance: {avg_distance:>7f}   [{current:>5d}/{size:>5d}]")

    with torch.no_grad():

        # normalize losses by number of batches (since loss_fn already gives average error in the batch by default).
        # average over replicas
        average_losses = average_losses / batches
        avg_loss = average_losses.sum() / y

        # compute l2 norm of weights, normalized by the sqrt of the number of weights
        # WARNING: ALL weights are counted
        avg_l2_norm = sum([torch.sqrt(sum([(p ** 2).sum() for p in replica.parameters()])) / np.sqrt(total_params) for replica in replicas]) / y

        # avg l2 distance (normalized by sqrt of the number of weights) of replicas from baricenter
        avg_distance = sum([torch.sqrt(sum([((baricenter[key] - p) ** 2).sum() for (key, p) in
                                            zip(baricenter.keys(), replica.parameters())])) / np.sqrt(total_params) for replica in replicas]) / y

    # N.B.: before 18/03, l2_norm and l2_distance were normalized by total_params, not by its root.

    return avg_loss.cpu(), avg_l2_norm.cpu(), avg_distance.cpu()


# TODO: test!

def train_loop_replicated(
        dataloader,        # dataloader object
        replicas,
        tester,            # instance of AE class. baricenter OrderedDict will be extracted from it
        loss_fn,           # pytorch loss function - expects reduction='mean'
        optimizers,        # list of pytorch optimizers - one per replica
        device,
        epochs,  # max number of epochs
        gamma0,
        gamma1,
        stddev=0.0,  # intensity of Gaussian noise
        val_data=None,     # use to estimate flatness and denoising capabilities
        flatness=True,    # whether to estimate flatness and denoising
        weight_iters=3,
        input_iters=3,
        weight_noise=None, # list - use to perturb weights
        input_noise=None,  # list - use to perturb inputs
        chkpt=None,        # path to save model checkpoint to estimate flatness (.pt extension)
        patience=None,     # early stopping
        improvement=None,   # what is an improvement in early stopping?
        zero=0.0,          # stop training if RE goes below this value
        verbose=1,         # 0, 1, 2
        log_frequency=25,      # number of batches before printing losses
        val_frequency=10,  # number of epochs before assessing flatness
        labels=False,      # if dataloader spits pairs (input, label), pass True
        wb=False,          # whether to log to wandb
        gamma_freq=1,      # update gamma every gamma_freq epochs
        test_data=None,    # test data
):
    losses = []                       # store average reconstruction errors across replicas
    l2s = []                          # store average l2 norms across replicas
    distances = []                    # store average distances from baricenter
    best, count = np.inf, 0           # for early stopping
    flag = 1 if verbose >= 2 else 0     # verbose argument for train_replicated
    gamma = gamma0                      # attraction parameter

    # move models to correct device
    tester.to(device)
    for replica in replicas:
        replica.to(device)

    # extract state_dict of the baricenter
    baricenter = tester.state_dict()

    # set up flatness estimation
    if flatness:

        # default stddevs
        if weight_noise is None:
            weight_noise = [0.0, 0.0001, 0.0005] + list(np.linspace(0.001, 0.01, 10)) + \
                           list(np.linspace(0.015, 0.05, 8)) + list(np.linspace(0.06, 0.1, 5))
        if input_noise is None:
            input_noise = list(np.linspace(0.0, 0.5, 21))

        # default val_data. N.B.: Better to pass same data for all runs (default does not!)
        if val_data is None:
            n = len(dataloader.dataset)
            val_data = dataloader.dataset[:n // 10]

        # default checkpoint path
        if chkpt is None:
            chkpt = 'perturbation_checkpoint.pt'

        # store results of landscape flatness
        h = (ceil(epochs / val_frequency) + 1) * weight_iters  # one row per flatness profile
        w = len(weight_noise) + 1  # stddevs, current epoch
        weight_res = torch.zeros((h, w))

        # store results of denoising
        h = (ceil(epochs / val_frequency) + 1) * input_iters
        w = len(input_noise) + 1
        input_res = torch.zeros((h, w))

        if test_data is not None:

            # store results of landscape flatness
            h = (ceil(epochs / val_frequency) + 1) * weight_iters  # one row per flatness profile
            w = len(weight_noise) + 1  # stddevs, current epoch
            test_weight_res = torch.zeros((h, w))

            # store results of denoising
            h = (ceil(epochs / val_frequency) + 1) * input_iters
            w = len(input_noise) + 1
            test_input_res = torch.zeros((h, w))

    # initial metrics (prior to training)
    if wb:
        total_params = sum(p.numel() for p in replicas[0].parameters())
        y = len(replicas)

        with torch.no_grad():

            # loss
            replicas[0].eval()
            loss = loss_fn(replicas[0](dataloader.dataset.data), dataloader.dataset.data)

            # l2 norm
            avg_l2_norm = sum([torch.sqrt(sum([(p ** 2).sum() for p in replica.parameters()])) / np.sqrt(total_params)
                               for replica in replicas]) / y
            # l2 distance
            avg_distance = sum([torch.sqrt(sum([((baricenter[key] - p) ** 2).sum() for (key, p) in
                                zip(baricenter.keys(), replica.parameters())])) / np.sqrt(total_params) for replica in replicas]) / y
        # log to wandb
        wandb.log({'RE': loss, 'l2': avg_l2_norm, 'distance': avg_distance, 'epoch': 0})

    for t in range(epochs):

        # estimate flatness and denosing RE for one of the replicas
        if flatness and t % val_frequency == 0:

            # # load current baricenter weights into shell
            # tester.load_state_dict(baricenter)

            # compute losses
            weight_res = estimate_flatness(replicas[0], loss_fn, val_data, noise=weight_noise, iters=weight_iters,
                                           out=weight_res, epoch=t, val_frequency=val_frequency, chkpt=chkpt)
            input_res = estimate_denoising(replicas[0], loss_fn, val_data, noise=input_noise, iters=input_iters,
                                           out=input_res, epoch=t, val_frequency=val_frequency)
            # compute metrics
            weight_dict = get_metrics(weight_res, weight_noise, iters=weight_iters, epoch=t,
                                      val_frequency=val_frequency, type='w')
            input_dict = get_metrics(input_res, input_noise, iters=input_iters, epoch=t,
                                     val_frequency=val_frequency, type='i')
            # log metrics to wandb
            if wb:
                wandb.log(weight_dict, commit=False)
                wandb.log(input_dict, commit=False)

            # do the same for the test data
            if test_data is not None:

                # compute losses
                test_weight_res = estimate_flatness(replicas[0], loss_fn, test_data, noise=weight_noise, iters=weight_iters,
                                                    out=test_weight_res, epoch=t, val_frequency=val_frequency, chkpt=chkpt)
                test_input_res = estimate_denoising(replicas[0], loss_fn, test_data, noise=input_noise, iters=input_iters,
                                                    out=test_input_res, epoch=t, val_frequency=val_frequency)
                # compute metrics
                weight_dict = get_metrics(test_weight_res, weight_noise, iters=weight_iters, epoch=t,
                                          val_frequency=val_frequency, type='test_w')
                input_dict = get_metrics(test_input_res, input_noise, iters=input_iters, epoch=t,
                                         val_frequency=val_frequency, type='test_i')
                # log metrics to wandb
                if wb:
                    wandb.log(weight_dict, commit=False)
                    wandb.log(input_dict, commit=False)

        # print out
        if verbose:
            print(f"epoch {t + 1}")

        # test_loss
        if test_data is not None:
            replicas[0].eval()
            with torch.no_grad():
                test_loss = loss_fn(test_data, replicas[0](test_data))
            if wb:
                wandb.log({'test_RE': test_loss}, commit=False)

        # train for one epoch
        metrics = train_replicated(dataloader, replicas, baricenter, loss_fn, optimizers, device, gamma=gamma, kappa=1,
                                 stddev=stddev, verbose=flag, log_frequency=log_frequency, labels=labels)
        loss, l2, distance = metrics[0], metrics[1], metrics[2]

        # update loss lists
        losses.append(loss)
        l2s.append(l2)
        distances.append(distance)

        # log losses to wandb
        if wb:
            wandb.log({'RE': loss, 'l2': l2, 'distance': distance, 'epoch': t+1})

        # print out
        if verbose:
            print(f"avg RE: {loss}   avg distance: {distance}\n")

        # check if reconstruction error hit 0
        if loss < zero:
            if verbose:
                print("Stop training because RE hit zero")
            break

        # check early stopping
        if patience is not None:
            if loss < best - improvement:
                best = loss
                count = 0
            else:
                count += 1

            # stopping condition
            if count == patience:
                if verbose:
                    print("Stop training because learning plateau-ed")
                break

        if t % gamma_freq == 0 and gamma < 0.75:
            gamma = min(gamma * (1 + gamma1), 0.75)

    # at the end of training, estimate flatness and denoising RE of one of the replicas
    if flatness:

        # # load current baricenter weights into shell
        # tester.load_state_dict(baricenter)

        # test flatness
        if test_data is not None:

            # compute losses
            test_weight_res = estimate_flatness(replicas[0], loss_fn, test_data, noise=weight_noise, iters=weight_iters,
                                                out=test_weight_res, epoch=epochs, val_frequency=val_frequency,
                                                chkpt=chkpt)
            test_input_res = estimate_denoising(replicas[0], loss_fn, test_data, noise=input_noise, iters=input_iters,
                                                out=test_input_res, epoch=epochs, val_frequency=val_frequency)
            # compute metrics
            weight_dict = get_metrics(test_weight_res, weight_noise, iters=weight_iters, epoch=epochs,
                                      val_frequency=val_frequency, type='test_w')
            input_dict = get_metrics(test_input_res, input_noise, iters=input_iters, epoch=epochs,
                                     val_frequency=val_frequency, type='test_i')
            # log metrics to wandb
            if wb:
                wandb.log(weight_dict, commit=False)
                wandb.log(input_dict, commit=False)

        # train flatness

        # compute losses
        weight_res = estimate_flatness(replicas[0], loss_fn, val_data, noise=weight_noise, iters=weight_iters,
                                       out=weight_res, epoch=epochs, val_frequency=val_frequency, chkpt=chkpt)
        input_res = estimate_denoising(replicas[0], loss_fn, val_data, noise=input_noise, iters=input_iters,
                                       out=input_res, epoch=epochs, val_frequency=val_frequency)
        # compute metrics
        weight_dict = get_metrics(weight_res, weight_noise, iters=weight_iters, epoch=epochs,
                                  val_frequency=val_frequency, type='w')
        input_dict = get_metrics(input_res, input_noise, iters=input_iters, epoch=epochs,
                                 val_frequency=val_frequency, type='i')
        # log metrics to wandb
        if wb:
            wandb.log(weight_dict, commit=False)
            wandb.log(input_dict)

        # create tables with flatness and denoising during training. columns:
        weight_noise = list(map(lambda x: str(x), weight_noise))
        input_noise = list(map(lambda x: str(x), input_noise))
        # epoch column
        input_noise.append('epoch')
        weight_noise.append('epoch')

        # train tables
        weight_df = pd.DataFrame(weight_res, columns=weight_noise)
        input_df = pd.DataFrame(input_res, columns=input_noise)
        # average flatness grouping by epoch
        avg_weight_df = weight_df.groupby(['epoch']).mean()
        avg_input_df = input_df.groupby(['epoch']).mean()

        # test tables
        if test_data is not None:
            test_weight_df = pd.DataFrame(test_weight_res, columns=weight_noise)
            test_input_df = pd.DataFrame(test_input_res, columns=input_noise)
            # average flatness grouping by epoch
            test_avg_weight_df = test_weight_df.groupby(['epoch']).mean()
            test_avg_input_df = test_input_df.groupby(['epoch']).mean()

        # log tables to wandb
        if wb:
            weight_tbl = wandb.Table(data=weight_df, columns=weight_noise)
            input_tbl = wandb.Table(data=input_df, columns=input_noise)
            avg_weight_tbl = wandb.Table(data=avg_weight_df, columns=weight_noise[:-1])
            avg_input_tbl = wandb.Table(data=avg_input_df, columns=input_noise[:-1])
            wandb.log({'flatness': weight_tbl, 'denoising': input_tbl,
                       'avg_flatness': avg_weight_tbl, 'avg_denoising': avg_input_tbl})
            if test_data is not None:
                test_weight_tbl = wandb.Table(data=test_weight_df, columns=weight_noise)
                test_input_tbl = wandb.Table(data=test_input_df, columns=input_noise)
                test_avg_weight_tbl = wandb.Table(data=test_avg_weight_df, columns=weight_noise[:-1])
                test_avg_input_tbl = wandb.Table(data=test_avg_input_df, columns=input_noise[:-1])
                wandb.log({'test_flatness': test_weight_tbl, 'test_denoising': test_input_tbl,
                           'test_avg_flatness': test_avg_weight_tbl, 'test_avg_denoising': test_avg_input_tbl})

    # load final baricenter weights into shell
    tester.load_state_dict(baricenter)

    # return
    if not flatness:
        return (losses, l2s, distances)
    if test_data is None:
        return (losses, l2s, distances, weight_df, input_df)
    return (losses, l2s, distances, weight_df, input_df, test_weight_df, test_input_df)
