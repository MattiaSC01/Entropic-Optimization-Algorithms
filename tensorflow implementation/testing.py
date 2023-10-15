import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
import os

from models import *
from layers import *
from callbacks import *

from utils import add_gaussian_noise


# evaluate the ability of a model to reconstruct corrupted inputs, using Mean Squared Error as a measure. 
# A dataframe with columns 'mean' and 'stddev' is returned. Each row corresponds to a noise intensity, 
# and reports the mean and stddev of the reconstruction errors the model incurred when denoising the 
# provided data corrupted with noise of that intensity.
# This function should not be called directly on a replicated autoencoder model, but rather on its 
# baricenter (extracted with extract_baricenter).

def input_robustness(
    data,               # data to use for the test
    model,              # model to test
    stddevs,            # list of floats. Inputs are perturbed by gaussian noises generated with these stddevs. adviced to use np.linspace().
    n_iter,             # for each stddev in stddevs, repeat the corruption and reconstruction this number of times
    percentage=False,   # if True, consider increase in loss relative to 0 noise loss
    seed=42,            # for reproducibility
):
    
    # initialize empty dictionary to store results. keys are the elements of stddevs, while entries are 
    # mean and stddev of the reconstruction error when data is perturbed with that noise intensity
    dict = {}
    
    # use MSE as reconstruction error
    mse = keras.losses.MeanSquaredError()
    
    # iterate through values in stddevs
    for stddev in stddevs:
        
        # initialize an empty list to store the reconstruction errors at the current noise intensity
        errors = []
        
        # compute the reconstruction error many times
        for i in range(n_iter):
            
            # apply gaussian noise
            data_noisy = add_gaussian_noise(data, stddev, seed=seed)

            # use the model to denoise corrupted data and compute reconstruction error
            error = mse(data, model.predict(data_noisy))
            
            # add reconstruction error to the list
            errors.append(error)
        
        # compute mean ans stddev of the reconstruction errors at the current noise intensity, and store them
        errors = tf.convert_to_tensor(errors)
        dict[stddev] = tf.math.reduce_mean(errors).numpy(), tf.math.reduce_std(errors).numpy()
    
    # create a dataframe summarizing the results
    df = pd.DataFrame(dict, index=['mean', 'stddev']).T

    # if percentage, compute the percentage increase in loss wrt 0 noise
    if percentage:
        df.loc[:, 'stddev'] = df.loc[:, 'stddev'] / df.iloc[0, 0]
        df.loc[:, 'mean'] = (df.loc[:, 'mean'] - df.iloc[0, 0]) / df.iloc[0, 0]

    # return 
    return df


# similar to the previous functions, but it perturbs weights rather than inputs, 
# and attempts to reconstruct non-noisy data. Like before, don't call on replicated
# autoencoder (but on the extracted baricenter).

def weight_robustness(
    data,               # data fot the test
    model,              # model to test
    stddevs,            # list of floats. Weights are perturbed by gaussian noises generated with these stddevs. adviced to use np.linspace().
    n_iter,             # for each stddev in stddevs, repeat the corruption and reconstruction this number of times
    percentage=False,   # if True, consider increase in loss relative to 0 noise loss
    seed=42,            # for reproducibility
):

    # initialize empty dictionary to store results. keys are the elements of stddevs, while entries are 
    # mean and stddev of the reconstruction error when weights are perturbed with that noise intensity
    dict = {}

    # use MSE as reconstruction error
    mse = keras.losses.MeanSquaredError()

    # save model weights
    original_weights = model.get_weights()

    # iterate through values in stddevs
    for stddev in stddevs:
        
        # initialize an empty list to store the reconstruction errors at the current noise intensity
        errors = []

        # compute the reconstruction error many times
        for i in range(n_iter):

            # apply gaussian noise to the weights
            perturbed_weights = []
            for tensor in original_weights:
                perturbed_tensor = add_gaussian_noise(tensor, stddev, seed=seed)
                perturbed_weights.append(perturbed_tensor)
            
            # compute reconstruction error using the model with corrupted weights
            model.set_weights(perturbed_weights)
            error = mse(data, model.predict(data))
            
            # add reconstruction error to the list
            errors.append(error)
        
        # compute mean ans stddev of the reconstruction errors at the current noise intensity, and store them
        errors = tf.convert_to_tensor(errors)
        dict[stddev] = tf.math.reduce_mean(errors).numpy(), tf.math.reduce_std(errors).numpy()
    
    # reset the original weights
    model.set_weights(original_weights)

    # create a dataframe summarizing the results
    df = pd.DataFrame(dict, index=['mean', 'stddev']).T
    
    # if percentage, compute the percentage increase in loss wrt 0 noise
    if percentage:
        df.loc[:, 'stddev'] = df.loc[:, 'stddev'] / df.iloc[0, 0]
        df.loc[:, 'mean'] = (df.loc[:, 'mean'] - df.iloc[0, 0]) / df.iloc[0, 0]

    # return 
    return df


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
