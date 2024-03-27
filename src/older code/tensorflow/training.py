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
