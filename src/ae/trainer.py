import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Union
from .logger import WandbLogger
from .model import AutoEncoder
from .constants import PROJECT, ENTITY
from .evaluation import Eval
from .sam import SAM
from .utils import set_seed
import random
import os
import json
import datetime
import socket
import platform


class Trainer:
    def __init__(
            self,
            model: AutoEncoder,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            train_loader: DataLoader,
            test_loader: Optional[DataLoader] = None,
            dataset_metadata: Optional[dict] = None,
            max_epochs: int = 1,
            device: str = "cpu",
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            log_to_wandb: bool = True,
            log_interval: int = 10,
            log_images: bool = False,
            checkpoint_interval: Optional[int] = None,
            checkpoint_root_dir: str = "checkpoints",
            flatness_interval: Optional[int] = None,
            train_set_percentage_for_flatness: Union[float, str] = 1.0,
            flatness_iters: int = 5,
            denoising_iters: int = 1,
            target_loss: Optional[float] = None,
            seed: int = 42,
            compile_model: bool = True,
            is_sweep: bool = False,
            wandb_project: Optional[str] = None,
        ):
        """
        :param model: autoencoder model
        :param optimizer: optimizer
        :param criterion: loss function
        :param train_loader: DataLoader for training set
        :param test_loader: DataLoader for test set
        :param dataset_metadata: metadata about the dataset
        :param max_epochs: number of epochs to train for
        :param device: "cpu" or "cuda"
        :param scheduler: learning rate scheduler
        :param log_to_wandb: whether to log to wandb
        :param log_interval: log training loss every log_interval steps
        :param log_images: whether to log input-output image pairs to wandb.
        Only works with MNIST-like images for now.
        :param checkpoint_interval: save a checkpoint every checkpoint_interval epochs.
        If None, no checkpoints are saved. To save only at the end of training, set it to epochs.
        :param checkpoint_root_dir: directory to save checkpoints
        :param flatness_interval: compute flatness every flatness_interval epochs. If None,
        flatness is not computed. To compute only at the end of training, set it to epochs.
        :param train_set_percentage_for_flatness: percentage of the training set to use for
        computing flatness. If "auto", use as many samples as there are in the test set.
        :param flatness_iters: number of repetitions to compute flatness. Higher values
        give more accurate results, but take longer to compute.
        :param target_loss: if not None, training stops when the train loss is below this value.
        :param seed: random seed set at the beginning of training.
        :param compile_model: if True, call torch.compile(model) at the end of __init__.
        """
        if checkpoint_interval is None:
            checkpoint_interval = max_epochs + 1 # no checkpoints
        else:
            os.makedirs(checkpoint_root_dir, exist_ok=True)
        if flatness_interval is None:
            flatness_interval = max_epochs + 1 # no flatness
        if train_set_percentage_for_flatness == 'auto':
            assert test_loader is not None, "If train_set_percentage_for_flatness is 'auto', test_loader must be provided."
            train_set_percentage_for_flatness = min(len(test_loader.dataset) / len(train_loader.dataset), 1.0)
        if target_loss is None:
            target_loss = 0.0
        if wandb_project is None:
            wandb_project = PROJECT
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dataset_metadata = dataset_metadata
        self.max_epochs = max_epochs
        self.device = device
        self.scheduler = scheduler
        self.log_to_wandb = log_to_wandb
        self.log_interval = log_interval
        self.log_images = log_images
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_root_dir = checkpoint_root_dir
        self.flatness_interval = flatness_interval
        self.train_set_percentage_for_flatness = train_set_percentage_for_flatness
        self.flatness_iters = flatness_iters
        self.denoising_iters = denoising_iters
        self.target_loss = target_loss
        self.seed = seed
        self.compile_model = compile_model
        self.is_sweep = is_sweep
        self.wandb_project = wandb_project
        self.step = 0
        self.epoch = 0
        self.logger = WandbLogger(project=wandb_project, entity=ENTITY)
        self.model.to(self.device)
        torch.compile(self.model)
    
    def get_training_hyperparameters(self):
        """
        Return a dictionary with the hyperparameters used for training.
        Does not include the model architecture, nor dataset metadata.
        """
        hyperparameters = {
            "lr": self.optimizer.param_groups[0]['lr'],
            "rho": self.optimizer.defaults['rho'] if isinstance(self.optimizer, SAM) else None,
            "batch_size": self.train_loader.batch_size,
            "max_epochs": self.max_epochs,
            "weight_decay": self.optimizer.param_groups[0]['weight_decay'],
            "optimizer": type(self.optimizer).__name__,
            "criterion": type(self.criterion).__name__,
            "scheduler": type(self.scheduler).__name__ if self.scheduler else None,
            "train_size": len(self.train_loader.dataset),
            "test_size": len(self.test_loader.dataset) if self.test_loader else None,
            "train_set_percentage_for_flatness": self.train_set_percentage_for_flatness,
            "target_loss": self.target_loss,
            "seed": self.seed,
            "compile_model": self.compile_model,
        }
        return hyperparameters
    
    def get_optimizer_hyperparameters(self):
        """
        Return a dictionary with the hyperparameters used for the optimizer.
        """
        return self.optimizer.defaults

    def get_scheduler_hyperparameters(self):
        """
        Return a dictionary with the hyperparameters used for the scheduler.
        """
        settings = {}
        if not self.scheduler:
            return settings
        for key, value in self.scheduler.__dict__.items():
            if not key.startswith("_") and key not in ['optimizer', 'best', 'num_bad_epochs', 'last_epoch']:
                settings[key] = value
        return settings
    
    def get_device_info(self):
        """
        Return a dictionary with information about the device used for training.
        """
        device_info = {
        "hostname": socket.gethostname(),
        "cpu": platform.processor(),
        "pytorch_version": torch.__version__,
        "device": self.device,
        "cuda_version": torch.version.cuda,
        }
        return device_info

    def train_step(self, x):
        loss = self.criterion(self.model(x), x)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step += 1
        if self.log_to_wandb:
            self.log_on_train_step(loss)
        return loss.item()
    
    def train_step_sam(self, x):
        # first forward-backward pass; use original weights w.
        loss = self.criterion(self.model(x), x)
        self.optimizer.zero_grad()
        loss.backward()
        # move to w + e(w)
        self.optimizer.first_step()
        # second forward-backward pass; use w + e(w)
        self.optimizer.zero_grad()
        self.criterion(self.model(x), x).backward()
        # move back to w and use base optimizer to update weights.
        self.optimizer.second_step()
        self.step += 1
        if self.log_to_wandb:
            self.log_on_train_step(loss)
        return loss.item()
    
    def train_epoch(self):
        self.model.train()
        loss = 0.0
        for x in self.train_loader:
            x = x.to(self.device)
            if isinstance(self.optimizer, SAM):
                loss += self.train_step_sam(x)
            else:
                loss += self.train_step(x)
        self.epoch += 1
        self.logger.log_metric(self.epoch, "train/epoch", self.step)
        self.logger.log_metric(self.epoch, "val/epoch", self.step) # redundant, but useful in the dashboard.
        if self.epoch % self.checkpoint_interval == 0:
            self.make_checkpoint()
        return loss / len(self.train_loader)
    
    def make_checkpoint(self):
        chkpt_dir = os.path.join(self.checkpoint_root_dir, self.dataset_metadata["id"])
        os.makedirs(chkpt_dir, exist_ok=True)
        chkpt_metadata = {
            "step": self.step,
            "epoch": self.epoch,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "optimizer": self.get_optimizer_hyperparameters(),
            "architecture": self.model.get_architecture(),
            "hyperparameters": self.get_training_hyperparameters(),
            "device": self.get_device_info(),
            "dataset": self.dataset_metadata,
        }
        with open(os.path.join(chkpt_dir, "metadata.json"), "w") as f:
            json.dump(chkpt_metadata, f)
        chkpt_path = os.path.join(chkpt_dir, "weights.pt")
        torch.save(self.model.state_dict(), chkpt_path)
        if self.log_to_wandb:
            artifact_name = f"chkpt-{self.dataset_metadata['id']}"
            self.logger.log_checkpoint(chkpt_dir, artifact_name)
    
    def log_on_train_step(self, loss):
        if self.step % self.log_interval != 0:
            return
        self.logger.log_metric(loss.item(), "train/loss", self.step)
        weight_norm, bias_norm = self.model.compute_parameter_norm()
        self.logger.log_metric(weight_norm, "train/weight_norm", self.step)
        self.logger.log_metric(bias_norm, "train/bias_norm", self.step)
    
    def test_step(self, x):
        x_hat = self.model(x)
        loss = self.criterion(x_hat, x)
        return loss.item()
    
    @torch.no_grad()
    def test_epoch(self):
        self.model.eval()
        loss = 0.0
        for x in self.test_loader:
            x = x.to(self.device)
            loss += self.test_step(x)
        loss /= len(self.test_loader)
        if self.log_to_wandb:
            self.logger.log_metric(loss, "val/loss", self.step)
            if self.log_images:
                self.log_images_to_wandb(split="val")
                self.log_images_to_wandb(split="train")
        if self.epoch % self.flatness_interval == 0:
            self.handle_flatness()
            self.handle_denoising()
        return loss
    
    def handle_flatness(self):
        sigmas = np.linspace(0, 0.15, 10)
        n_iters = self.flatness_iters
        flatness_train = Eval.flatness_profile(self.model, self.train_loader, sigmas, n_iters, criterion=self.criterion, data_percentage=self.train_set_percentage_for_flatness)
        self.log_flatness(flatness_train, split="train")
        flatness_val = Eval.flatness_profile(self.model, self.test_loader, sigmas, n_iters, criterion=self.criterion)
        self.log_flatness(flatness_val, split="val")
    
    def log_flatness(self, losses: dict, split: str):
        """
        :param losses: dictionary with noise_strengths as keys and lists of losses as values
        :param split: "train" or "val"
        """
        df = pd.DataFrame(losses)
        df.columns = [f"{sigma:.3f}" for sigma in df.columns]
        for sigma in df.columns:
            mean = df[sigma].mean()
            std = df[sigma].std()
            self.logger.log_metric(mean, f"flatness/{split}/mean_{sigma}", self.step)
            self.logger.log_metric(std, f"flatness/{split}/std_{sigma}", self.step)
        avg_diff = df.mean().mean() - df.iloc[0, 0]
        self.logger.log_metric(avg_diff, f"flatness/{split}/avg_diff", self.step)
        self.logger.log_metric(avg_diff, f"{split}/flatness_avg_diff", self.step) # redundant, but useful in the dashboard.
        # self.logger.log_table(df, f"tables/{split}_flatness", self.step)
        self.plot_and_log(losses, split=split, plot_type="flatness")

    def plot_and_log(self, losses: dict, split: str, plot_type: str):
        """
        If run is within a sweep, do nothing (cannot use matplotlib GUI from a non-main thread).
        :param losses: dictionary with noise_strengths as keys and lists of losses as values
        :param split: "train" or "val"
        :param plot_type: "flatness" or "denoising"
        """
        if self.is_sweep:
            return
        plt.figure(figsize=(10, 5))
        plt.title(f"{plot_type} profile")
        Eval.plot_profile(losses, color="blue", label=split)
        plt.legend()
        self.logger.log_plot(f"{split}/{plot_type}", self.step)
        plt.close()

    def handle_denoising(self):
        sigmas = np.linspace(0, 1.0, 10)
        n_iters = self.denoising_iters
        denoising_train = Eval.denoising_profile(self.model, self.train_loader, sigmas, n_iters, criterion=self.criterion, data_percentage=self.train_set_percentage_for_flatness)
        self.log_denoising(denoising_train, split="train")
        denoising_val = Eval.denoising_profile(self.model, self.test_loader, sigmas, n_iters, criterion=self.criterion)
        self.log_denoising(denoising_val, split="val")

    def log_denoising(self, losses: dict, split: str):
        """
        :param losses: dictionary with noise_strengths as keys and lists of losses as values
        :param split: "train" or "val"
        """
        df = pd.DataFrame(losses)
        df.columns = [f"{sigma:.3f}" for sigma in df.columns]
        for sigma in df.columns:
            mean = df[sigma].mean()
            std = df[sigma].std()
            self.logger.log_metric(mean, f"denoising/{split}/mean_{sigma}", self.step)
            self.logger.log_metric(std, f"denoising/{split}/std_{sigma}", self.step)
        avg_diff = df.mean().mean() - df.iloc[0, 0]
        self.logger.log_metric(avg_diff, f"denoising/{split}/avg_diff", self.step)
        self.logger.log_metric(avg_diff, f"{split}/denoising_avg_diff", self.step) # redundant, but useful in the dashboard.
        # self.logger.log_table(df, f"tables/{split}_denoising", self.step)
        self.plot_and_log(losses, split=split, plot_type="denoising")
    
    @torch.no_grad()
    def log_images_to_wandb(self, split: str = "val", n_images: int = 1):
        """
        Log some pairs of input and output images to wandb.
        For now, only works with MNIST-like images.
        :param split: "val" or "train"
        :param n_images: number of distinct images to try
        """
        ds = self.test_loader.dataset if split == "val" else self.train_loader.dataset
        strengths = [0.0, 0.0, 0.5, 1.0, 0.25, 0.5, 0.25, 0.5]
        types = [
            "identity",
            "identity",
            "gaussian-additive",
            "gaussian-additive",
            "salt-and-pepper",
            "salt-and-pepper",
            "dropout",
            "dropout",
        ]
        images = []
        for _ in range(n_images):
            idx = random.randint(0, len(ds) - 1)
            for strength, noise_type in zip(strengths, types):
                x = ds[idx].to(self.device)
                noisy = Eval.corrupt_data(x, strength, noise_type=noise_type)
                x_hat = self.model(noisy)
                noisy = self.reassemble_image(noisy)
                x_hat = self.reassemble_image(x_hat)
                images.extend([noisy, x_hat])
        self.logger.log_tensor_as_image(images, f"{split}/images", self.step)

    def reassemble_image(self, x):
        """
        :param x: a single image tensor (C*H*W,)
        :return: a reshaped image tensor (C, H, W). dimensions
        are inferred from the dataset metadata.
        """
        name = self.dataset_metadata["id"]
        if "cifar" in name.lower():
            x = x.reshape(32, 32, 3)
            x = x.permute(2, 0, 1)
        elif "mnist" in name.lower():
            x = x.reshape(1, 28, 28)
        else:
            raise ValueError(f"Dataset {name} not recognized.")
        return x
    
    def train(self):
        if self.log_to_wandb:
            self.setup_wandb()
        set_seed(self.seed)
        for epoch in range(self.max_epochs):
            train_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}/{self.max_epochs}, train_loss: {train_loss:.4f}", end="")
            if self.test_loader:
                val_loss = self.test_epoch()
                print(f", val_loss: {val_loss:.4f}", end="")
            print()
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()
                print(f"lr: {self.scheduler.get_last_lr()[0]}")
            if train_loss < self.target_loss:
                print(f"Train loss hit target {self.target_loss}. Stopping training.")
                break
        self.end_training()

    def end_training(self):
        if self.checkpoint_interval <= self.max_epochs and self.epoch % self.checkpoint_interval != 0:
            self.make_checkpoint()
        if self.flatness_interval <= self.max_epochs and self.epoch % self.flatness_interval != 0:
            self.handle_flatness()
            self.handle_denoising()
        if self.log_to_wandb:
            self.logger.end_run()

    def setup_wandb(self):
        self.logger.init_run(self.model, is_sweep=self.is_sweep)
        self.logger.use_dataset(self.dataset_metadata)
        self.logger.add_to_config(self.get_training_hyperparameters())
        self.logger.add_to_config(self.get_optimizer_hyperparameters())
        self.logger.add_to_config(self.get_scheduler_hyperparameters())
        self.logger.add_to_config(self.model.get_architecture())
        self.logger.add_to_config(self.get_device_info())
        if self.dataset_metadata:
            self.logger.add_to_config(self.dataset_metadata)
