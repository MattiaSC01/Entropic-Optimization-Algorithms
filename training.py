import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib as mpl
from matplotlib import pyplot as plt
import os

from models import *
from layers import *
from callbacks import *

# files needed for image elastic distortion
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

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


# load the state of a model and of its optimizer, as saved by train
def load_model(cls, config, compile_config, path, batch):

    loaded = cls(**config)
    loaded.compile(**compile_config)

    loaded.train_on_batch(batch, batch)
    loaded.load_weights(path)

    return loaded


# add gaussian noise to the data
def add_noise(data, stddev=0.1, seed=42):
    noise = tf.random.normal(shape=data.shape, mean=0.0, stddev=stddev, seed=seed)
    return data + noise


def input_robustness(data, model, stddevs, n_iter):
    
    dict = {}
    
    for stddev in stddevs:
        errors = []
        mse = keras.losses.MeanSquaredError()
        for i in range(n_iter):
            data_noisy = add_noise(data, stddev)
            error = mse(data, model.predict(data_noisy))
            errors.append(error)
        errors = tf.convert_to_tensor(errors)
        dict[stddev] = tf.math.reduce_mean(errors).numpy(), tf.math.reduce_std(errors).numpy()
    
    return pd.DataFrame(dict, index=['mean', 'stddev'])


def weight_robustness(data_noisy, data, model, stddevs, n_iter):

    dict = {}
    original_weights = model.get_weights()

    for stddev in stddevs:
        errors = []
        mse = keras.losses.MeanSquaredError()
        for i in range(n_iter):
            perturbed_weights = []
            for tensor in original_weights:
                perturbed_weights.append(add_noise(tensor, stddev))
            
            model.set_weights(perturbed_weights)
            error = mse(data, model.predict(data))
            errors.append(error)
        
        errors = tf.convert_to_tensor(errors)
        dict[stddev] = tf.math.reduce_mean(errors).numpy(), tf.math.reduce_std(errors).numpy()
    
    model.set_weights(original_weights)
    return pd.DataFrame(dict, index=['mean', 'stddev'])


def plot_errors(errors, x_axis, alpha=0.2, title=''):
    """
    :param errors: an errors dataframe containing the mean error
    and the std of the errors at each value.
    :param x_axis: the values used (in our case the stds).
    :param alpha: sets the opacity of the error bars.
    :return: a plot of the errors.
    """
    # Set the styling theme
    plt.style.use('default')
    # Set the plot size
    fig = plt.figure(figsize=(10, 6))

    # get means and stds
    means = errors.iloc[0, :]
    stds = errors.iloc[1, :]

    # plot means
    plt.plot(x_axis, means, 'or')
    plt.plot(x_axis, means, '-', color='gray')

    # plot stds
    plt.fill_between(x_axis, means - stds, means + stds,
                     color='gray', alpha=alpha)

    # other plot specs
    plt.grid(True)
    plt.xlim(x_axis[0], x_axis[-1])
    plt.title(title)
    plt.xlabel('Std Deviation')
    plt.ylabel('MSE error')


# for a given class and configuration, it loads the weights of the models that have been trained and it
# calls input_robustness and weights_robustness. It averages the resulting dataframes and returns the two averages.
# is this what we want? is averaging standard deviations (with respect to the different means) desirable? should
# we make input_robustness and weight_robustness return all the metrics, and average them inside test directly?

def test(cls, config, compile_config, data, n_copies, name, stddevs_inputs, stddevs_weights, n_iter):
    
    # root of the path of the model
    PATH = f"models/{name}"

    # initialize model
    model = cls(**config)
    model.compile(**compile_config)

    model.train_on_batch(data[:1], data[:1])   # to initialize optimizer's weights
    
    for i in range(n_copies):
        
        # load the model
        MODEL_PATH = os.path.join(PATH, str(i), "weights")
        model.load_weights(MODEL_PATH)
        
        # compute metrics
        df_inputs = input_robustness(data, model, stddevs_inputs, n_iter)
        df_weights = weight_robustness(data, data, model, stddevs_weights, n_iter)

        # update average dataframe (create it if i == 0)
        if i == 0:
            inputs_data = df_inputs.to_numpy()
            weights_data = df_weights.to_numpy()
        else:
            inputs_data += df_inputs.to_numpy()
            weights_data += df_weights.to_numpy()
    
    inputs_data /= n_copies
    weights_data /= n_copies

    return (pd.DataFrame(inputs_data, index=df_inputs.index, columns=df_inputs.columns), 
        pd.DataFrame(weights_data, index=df_weights.index, columns=df_weights.columns))


def elastic_transform(image, alpha_range, sigma, random_state=None):
    """
    :image: Numpy array with shape (height, width, channels).
    :alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
           Controls intensity of deformation.
    :sigma: Float, sigma of gaussian filter that smooths the displacement fields.
    :random_state: `numpy.random.RandomState` object for generating displacement fields.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def weight_dist(weights, title='Weight Distribution'):
    """

    :param weights: weights of a Keras model.
    :param title: Title of the plot.
    :return: histogram of all weights of the model.
    """
    flat_weights = list(map(np.ndarray.flatten, weights[0::2]))
    # you can get the weights by layer using the list above
    # for now we plot all the weights
    all_weights = np.concatenate(flat_weights)
    print(all_weights.max())
    plt.hist(all_weights)
    plt.title(title)
    plt.show()