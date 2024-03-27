import torch
from torch.utils.data import DataLoader
from ae.utils import set_seed, clean_wandb
from ae.dataset import HiddenManifold, load_mnist
from ae.model import AutoEncoder
from ae.trainer import Trainer
from ae.sam import SAM

# mnist, B = 15.
# shallow: bs 64, lr 0.001, wd 0.1, rho 0.15
# deep: bs 64, wd 0.1, lr 0.0005, rho 0.1


def generate_configs():
    for seed in [12, 34]:
        for optimizer_class in ["adamw", "sam"]:
                yield {
                        "optimizer_class": optimizer_class,
                        "seed": seed,
                }

log_dataset_to_wandb = False # i do it here so that it's logged only once

for config in generate_configs():
    optimizer_class = config["optimizer_class"]
    seed = config["seed"]

    # fixed hyperparams
    N = 784
    B = 15
    hidden_layers = [N, N, B]  # from first hidden to bottleneck, extrema included
    train_size = 256
    test_size = 512
    batch_size = 64
    lr = 0.0005
    weight_decay = 0.1
    max_epochs = 3000
    device = "cuda"
    compile_model = True
    base_optimizer = torch.optim.AdamW
    rho = 0.1
    target_loss = 0.0
    print("Using device: ", device)

    # logging
    log_to_wandb = True
    log_images = True
    log_interval = 10 # batches
    checkpoint_interval = None # epochs
    flatness_interval = 200 # epochs
    flatness_iters = 5
    denoising_iters = 1
    wandb_project = "ae-prove"


    # mnist dataset
    root = "../data"
    dataset, dataset_metadata = load_mnist(log_to_wandb=log_dataset_to_wandb, project=wandb_project, root=root)
    train_loader = DataLoader(dataset[:train_size], batch_size=batch_size)
    test_loader = DataLoader(dataset[-test_size:], batch_size=test_size)  # be mindful of the size
    if log_dataset_to_wandb:
        log_dataset_to_wandb = False


    # model
    model = AutoEncoder(input_dim=N, encoder_hidden=hidden_layers, activation="ReLU", seed=seed)
    if optimizer_class.lower() == "sam":
        optimizer = SAM(model.parameters(), base_optimizer, lr=lr, weight_decay=weight_decay, rho=rho)
    elif optimizer_class.lower() == "adamw":
        optimizer = base_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    scheduler = None


    train_config = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,    
        "train_loader": train_loader,
        "test_loader": test_loader,
        "dataset_metadata": dataset_metadata,
        "max_epochs": max_epochs,
        "device": device,
        "scheduler": scheduler,
        "log_to_wandb": log_to_wandb,
        "log_interval": log_interval,
        "log_images": log_images,
        "checkpoint_interval": checkpoint_interval,
        "checkpoint_root_dir": "../checkpoints",
        "flatness_interval": flatness_interval,
        "train_set_percentage_for_flatness": 'auto',
        "flatness_iters": flatness_iters,
        "denoising_iters": denoising_iters,
        "target_loss": target_loss,
        "seed": seed,
        "compile_model": compile_model,
        "wandb_project": wandb_project,
    }

    trainer = Trainer(**train_config)
    trainer.train()
