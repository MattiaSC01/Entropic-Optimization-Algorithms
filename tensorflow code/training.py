import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib as mpl
from matplotlib import pyplot as plt
import os

from callbacks import StopTrainingCallback, ExponentialRateCallback
from models import ReplicatedAutoEncoder


# set up path to save figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)


# function to save figures (from Hands On textbook)
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# function to plot MNIST digits (from Hands On textbook)
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # This is equivalent to n_rows = ceil(len(instances) / images_per_row):
    n_rows = (len(instances) - 1) // images_per_row + 1

    # Append empty images to fill the end of the grid, if needed:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # Reshape the array so it's organized as a grid containing 28×28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we
    # want to combine next to each other, using transpose(), and only then we
    # can reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # Now that we have a big image, we just need to show it:
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


# subsample the data to have a smaller training size
def subsample(data, n_samples, stratify=None, seed=42):

    x, _ = train_test_split(data, train_size=n_samples, stratify=stratify, random_state=seed)
    return x


# configure callbacks and returns a list to be passed as the callbacks argument to fit.
# Not meant to be called independently. Called by train.

def _make_callbacks(
    cls,   # one of AutoEncoder, DenoisingAutoEncoder, ReplicatedAutoEncoder
    model_path,   # path where to save the model checkpoint
    zero_error,   # stop training when monitored loss hits this value
    early_stopping,   # boolean. If True, use early stopping callback
    patience,   # patience for early stopping (ignored if early_stopping == False)
    initial_rate,   # initial rate for regularization schedule (ignored if cls != ReplicatedAutoEncoder)
    update_coeff,   # update coefficient for regularization schedule (ignored if cls != ReplicatedAutoEncoder)
):
    
    MONITOR = 'loss'   # loss to monitor
    ZERO_IMPROVEMENT = 0   # zero improvement for early stopping
    
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor=MONITOR,
            save_weights_only=True,
        )

    stop_training_callback = StopTrainingCallback(
        epsilon=zero_error,
        monitor=MONITOR,
    )
    
    callbacks = [
            checkpoint_callback, 
            stop_training_callback,
        ]
    
    if early_stopping:
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor=MONITOR,
            min_delta=ZERO_IMPROVEMENT,
            patience=patience,
        )
        callbacks += [early_stopping_callback]
    
    if cls == ReplicatedAutoEncoder:
        exponential_rate_callback = ExponentialRateCallback(initial_rate, update_coeff)
        callbacks += [exponential_rate_callback]
    
    return callbacks


# train a model with a given configuration many times, and saves the weights through checkpoints.

def train(
    cls,   # one of AutoEncoder, DenoisingAutoEncoder, ReplicatedAutoEncoder
    config,   # configuration of the model (architecture, etc.)
    compile_config,   # optimizer, loss, etc.
    data,   # dataset to train on
    stratify=None,   # for subsampling, ignored if n_samples is None
    n_samples=None,   # number of points to keep for subsampling. Keep all if None
    n_iter=1,   # how many times to train
    epochs=20,   
    zero_error=1e-4,   # stop training when monitored loss hits this value
    early_stopping=False,   # whether to use early stopping
    patience=10,   # patience for early stopping (ignored if early_stopping == False)
    initial_rate=1e-6,   # initial rate for regularization schedule (ignored if cls != ReplicatedAutoEncoder)
    update_coeff=0.2,   # update coefficient for regularization schedule (ignored if cls != ReplicatedAutoEncoder)
    name="test", 
    seed=42,
):

    # subsample if required
    if n_samples is not None:
        data = subsample(data, n_samples, stratify, seed)
    
    # instantiate model with given configuration
    model = cls(**config)

    # some useful constants
    PATH = f"models/{name}"
    MONITOR = 'loss'
    ZERO_IMPROVEMENT = 0

    # train the models
    for i in range(n_iter):

        # path for checkpointing
        MODEL_PATH = os.path.join(PATH, str(i), "weights")

        # reset weights to train models after the first (to avoid retracing)
        if i > 0:
            model.reset_weights()
        model.compile(**compile_config)

        print(f"\nbeginning training of the {i}-th model\n")

        # configure callbacks
        callbacks = _make_callbacks(cls, MODEL_PATH, zero_error, early_stopping, patience, initial_rate, update_coeff)

        # fit the model
        history = model.fit(
            x=data, 
            y=data, 
            epochs=epochs,
            callbacks=callbacks,
        )

        # save history in memory
        HISTORY_PATH = os.path.join(PATH, str(i), "history")
        df = pd.DataFrame(history.history)
        df.to_csv(HISTORY_PATH)
