import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from models import ReplicatedAutoEncoder, FullReplicatedAutoEncoder
from testing import input_robustness, weight_robustness
from utils import extract_baricenter


# The following class, inheriting from keras.callbacks.Callback, is a callback
# that takes as input to the constructor a schedule for the regularization rate
# in the form of two lists. It then applies the schedule by updating the rate,
# if necessary, at the beginning of every epoch.

class RateCallback(keras.callbacks.Callback):
    
    def __init__(self, times=[], values=[], **kwargs):
        super(RateCallback, self).__init__(**kwargs)
        self.times = times
        self.values = values
        self.i = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.times:
            print(f"\n\n Setting rate to {self.values[self.i]} \n\n")
            self.model.rate.assign(tf.constant(self.values[self.i], dtype=tf.float32))
            self.i += 1


# The following is a less flexible callback handling the regularization rate.
# Instead of accepting any schedule, it uses an exponentially increasing schedule
# and only takes as input an initial value and a rate of increase.

class ExponentialRateCallback(keras.callbacks.Callback):
    def __init__(self, initial_value=None, update_coeff=0.1, verbose=1, **kwargs):
        super(ExponentialRateCallback, self).__init__(**kwargs)
        self.initial_value = initial_value
        self.update_coeff = update_coeff
        self.verbose = verbose
    
    def on_train_begin(self, logs=None):
        if self.initial_value is not None:
            value = self.initial_value / (1 + self.update_coeff)
            self.model.rate.assign(tf.constant(value, dtype=tf.float32))
    
    def on_epoch_begin(self, epoch, logs=None):
        current = self.model.rate.read_value()
        new = current * (1 + self.update_coeff)
        if self.verbose:
            print(f"\nSetting rate to {new} \n")
        self.model.rate.assign(tf.constant(new, dtype=tf.float32))


# the following stops training when the monitored loss is below the value
# of the input epsilon.

class StopTrainingCallback(keras.callbacks.Callback):
    def __init__(self, epsilon=1e-3, monitor='loss', **kwargs):
        super(StopTrainingCallback, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.monitor = monitor
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs[self.monitor]
        if current < self.epsilon:
            print("   Stopping Training!")
            self.model.stop_training = True


# The following Callback is meant to calculate
# generalization statistics after each epoch

class GeneralizableCallback(keras.callbacks.Callback):
    def __init__(self, X, y, std=0.1):
        super(GeneralizableCallback, self).__init__()
        self.gen_results = [] # results of each epoch
        self.std = std
        self.X = X
        self.y = y

    # add comment for commit
    def on_train_start(self):
        pass

    def on_epoch_end(self, epoch, logs=None):

        print('Perturbing the weights...')

        # save the initial score

        initial_result = self.model.evaluate(self.X, self.y, verbose=0)

        # placeholder for weights
        perturbed_weights = []
        original_weights = self.model.get_weights()

        # create the perturbed weights list
        for weights in original_weights:
            perturb = tf.random.normal(stddev=self.std, shape=weights.shape)
            perturbed_weights.append(weights+perturb)

        # set perturbed weights
        self.model.set_weights(perturbed_weights)

        # evaluate on perturbed weights
        final_result = self.model.evaluate(self.X, self.y, verbose=0)

        result = tf.math.subtract(final_result, initial_result)
        print('The error from the current model is:', initial_result)
        print('The error made from the perturbed model is', final_result)

        # store result
        self.gen_results.append(result)

        # reset original weights
        self.model.set_weights(original_weights)


# measure robustness of the model to perturbation of inputs and weights during training,
# at regular intervals. Results are accessible through self.input_dfs and self.weight_dfs.

class FlatnessCallback(keras.callbacks.Callback):

    def __init__(
        self, 
        data,           # data to use to compute the metrics
        frequency,      # integer number of epochs. Frequency at which to compute the metrics
        n_iter=3,       # n_iter passed to input_robustness and weight robustness
        stddevs=None,   # tuple of two elements, stddevs passed respectively to input_robustness and weight_robustness 
        **kwargs,       # arguments for parent constructor
    ):
        super(FlatnessCallback, self).__init__(**kwargs)
        
        # default stddev ranges
        if stddevs is None:
            stddevs = np.linspace(0, 3.0, 31), np.linspace(0, 1.0, 21)
        
        # unpacks stddev ranges
        self.input_std, self.weight_std = stddevs

        # save parameters
        self.data = data
        self.frequency = frequency
        self.n_iter = n_iter
        
        # initialize empty dictionaries to store results
        self.input_dfs = {}
        self.weight_dfs = {}
    
    def on_epoch_end(self, epoch, logs=None):
        
        # every self.frequency epochs
        if epoch > 0 and epoch % self.frequency == 0:
            
            original_weights = self.model.get_weights()

            # extract baricenter if model is a replicated autoencoder, to speed up computations
            if isinstance(self.model, (ReplicatedAutoEncoder, FullReplicatedAutoEncoder)):
                ae = extract_baricenter(self.model)
            else:
                ae = self.model
            
            # compute metrics
            input_df = input_robustness(self.data, ae, self.input_std, self.n_iter)
            weight_df = weight_robustness(self.data, ae, self.weight_std, self.n_iter)

            # save results
            self.input_dfs[epoch] = input_df
            self.weight_dfs[epoch] = weight_df
