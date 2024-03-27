import torch
from torch.utils.data import DataLoader
from ae.utils import set_seed, clean_wandb
from ae.constants import PROJECT
from ae.dataset import HiddenManifold, load_mnist, load_cifar
from ae.model import AutoEncoder
from ae.trainer import Trainer
from ae.sam import SAM
import wandb
import yaml
import argparse


def get_mnist(config=None):
    root = "../data"
    dataset, dataset_metadata = load_mnist(log_to_wandb=False, root=root)
    dataset = dataset[:config.train_size + config.test_size]
    train_loader = DataLoader(dataset[:config.train_size], batch_size=config.batch_size)
    test_loader = DataLoader(dataset[config.train_size:config.train_size+config.test_size], batch_size=128)
    return train_loader, test_loader, dataset_metadata

def get_cifar10(config=None):
    dataset, dataset_metadata = load_cifar(log_to_wandb=False, num_classes=10)
    dataset = dataset[:config.train_size + config.test_size]
    train_loader = DataLoader(dataset[:config.train_size], batch_size=config.batch_size)
    test_loader = DataLoader(dataset[config.train_size:config.train_size+config.test_size], batch_size=128)
    return train_loader, test_loader, dataset_metadata

def get_optimizer(config, model):
    if config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer.lower() == "sam":
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, lr=config.lr, weight_decay=config.weight_decay, rho=config.rho)
    scheduler = None
    return optimizer, scheduler


def train(config=None):
    """
    Pass this function to wandb.agent to train the model using
    sweep hyperparameters.
    :param config: this will be overridden by the sweep config.
    You can use it to pass default values.
    """
    with wandb.init(config=config):
        config = wandb.config # this, apparently, will have been set by wandb.agent
        get_dataset = get_mnist if config.dataset.lower() == "mnist" else get_cifar10
        train_loader, test_loader, dataset_metadata = get_dataset(config)
        model = AutoEncoder(input_dim=config.N, encoder_hidden=config.hidden_layers, activation=config.activation, seed=config.seed)
        optimizer, scheduler = get_optimizer(config, model)
        criterion = torch.nn.MSELoss()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        train_config = {
            "model": model,
            "optimizer": optimizer,
            "criterion": criterion,    
            "train_loader": train_loader,
            "test_loader": test_loader,
            "dataset_metadata": dataset_metadata,
            "max_epochs": config.max_epochs,
            "device": device,
            "scheduler": scheduler,
            "log_to_wandb": True,
            "log_interval": config.log_interval,
            "log_images": config.log_images,
            "checkpoint_interval": config.checkpoint_interval,
            "checkpoint_root_dir": "../checkpoints",
            "flatness_interval": config.flatness_interval,
            "train_set_percentage_for_flatness": 'auto',
            "flatness_iters": config.flatness_iters,
            "target_loss": config.target_loss,
            "seed": config.seed,
            "compile_model": config.compile_model,
            "is_sweep": True,
        }
        
        trainer = Trainer(**train_config)
        trainer.train()


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='sweep.yaml', help='Path to the config file')
parser.add_argument('--project', type=str, default=PROJECT, help='Wandb project name')
parser.add_argument('--count', type=int, default=1, help='Number of runs')
args = parser.parse_args()

with open(args.config, "r") as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)
sweep_id = wandb.sweep(sweep_config, project=args.project)
agent = wandb.agent(sweep_id, function=train, count=args.count)
