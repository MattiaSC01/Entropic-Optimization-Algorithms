import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras


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
# generalizability statistics after each epoch

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

