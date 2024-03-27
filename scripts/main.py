import torch
from torch.utils.data import DataLoader
from ae.utils import set_seed, clean_wandb
from ae.dataset import HiddenManifold, load_mnist, load_cifar
from ae.model import AutoEncoder
from ae.trainer import Trainer
from ae.sam import SAM

# mnist, B = 15.
# shallow: bs 64, lr 0.001, wd 0.1, rho 0.15
# deep: bs 64, wd 0.1, lr 0.0005, rho 0.1

# cifar, B = 30
# shallow: bs 64, lr 0.0005, wd 0.1, rho 0.15


def generate_configs():
    for seed in range(71, 81):
        for train_size in [256, 512, 1024, 2048, 4096, 8192, 16384]:
            for optimizer_class in ["adamw", "sam"]:
                    yield {
                        "train_size": train_size,
                        "seed": seed,
                        "optimizer_class": optimizer_class,
                    }

# mnist
# mean + 3*std for 10 seeds (except 256 which is arbitrary)
# targets = {
#      (256, 1): 0.015,
#      (256, 2): 0.002,
#      (512, 1): 0.015,
#      (512, 2): 0.0085,
#      (1024, 1): 0.035,
#      (1024, 2): 0.017,
#      (2048, 1): 0.045,
#      (2048, 2): 0.032,
#      (4096, 1): 0.056,
#      (4096, 2): 0.043,
#      (8192, 1): 0.050,
#      (8192, 2): 0.038,
#      (16384, 1): 0.050,
#      (16384, 2): 0.043,
#      (32768, 1): 0.052,
#      (32768, 2): 0.046,
# }

targets = {
    256: 0.01,
    512: 0.015,
    1024: 0.02,
    2048: 0.025,
    4096: 0.035,
    8192: 0.05,
    16384: 0.055,
}

log_dataset_to_wandb = True # i do it here so that it's logged only once

for config in generate_configs():
    # config
    train_size = config["train_size"]
    seed = config["seed"]
    optimizer_class = config["optimizer_class"]

    # fixed hyperparams
    N = 3072
    B = 30
    hidden_layers = [N, B]  # from first hidden to bottleneck, extrema included
    test_size = 30000
    batch_size = 64
    lr = 0.0005
    weight_decay = 0.15
    max_epochs = 3000 * 512 // train_size
    device = "cuda"
    compile_model = True
    base_optimizer = torch.optim.AdamW
    rho = 0.05
    print("Using device: ", device)

    # target loss
    L = len(hidden_layers) - 1
    target_loss = targets[train_size]

    # logging
    log_to_wandb = True
    log_images = True
    log_interval = 10 # batches
    checkpoint_interval = None # epochs
    flatness_interval = max(1, 40 * 512 // train_size) # epochs
    # flatness_interval = max_epochs # epochs
    flatness_iters = 10
    denoising_iters = 3
    wandb_project = "cifar-target-loss"


    # dataset
    root = "../data"
    dataset, dataset_metadata = load_cifar(log_to_wandb=log_dataset_to_wandb, project=wandb_project, root=root)
    set_seed(0)
    dataset = dataset[torch.randperm(len(dataset))]  # shuffle
    train_loader = DataLoader(dataset[:train_size], batch_size=batch_size, shuffle=True)
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
