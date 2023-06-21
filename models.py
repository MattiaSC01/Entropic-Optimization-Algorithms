# TODO: where possible, replace native python functions with Tensorflow equivalent functions, to improve performance.


import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from layers import Encoder, Decoder


# A vanilla AutoEncoder implemented as a Multi-Layer Perceptron stacking
# an Encoder and a Decoder from the script layers.py.

class AutoEncoder(keras.Model):
    
    def __init__(
        self,
        neurons_encoder,   # list of integers
        neurons_decoder,   # list of integers
        activations_encoder=None,   # list of activations
        activations_decoder=None,   # list of activations
        name='autoencoder',
        l2_rate=None,               # rate for l2 regularization
        **kwargs,
    ):
        
        # call parent constructor
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        
        # initialize Encoder and Decoder
        self.encoder = Encoder(neurons_encoder, activations_encoder, name='encoder', l2_rate=l2_rate)
        self.decoder = Decoder(neurons_decoder, activations_decoder, name='decoder', l2_rate=l2_rate)
    
    # define forward pass
    def call(self, inputs):
        return self.decoder(self.encoder(inputs))
    
    # initialize weights of encoder and decoder
    def build(self, input_shape):
        self.encoder._build(input_shape)
        self.decoder._build(self.encoder.bottleneck)
        if not hasattr(self.build, '_is_default'):
            self._build_input_shape = input_shape
        self.built = True

    # returns the configuration of the instance
    def get_config(self):
        config = {
            "neurons_encoder": self.encoder.neurons,
            "neurons_decoder": self.decoder.neurons,
            "activations_encoder": self.encoder.activations,
            "activations_decoder": self.decoder.activations,
        }
        return config
    
    # re-initialize randomly the weights of the encoder and of the decoder
    def reset_weights(self):
        print(f"\nRe-initializing weights of model {self.name}\n")
        self.encoder.reset_weights()
        self.decoder.reset_weights()


# a Denoising AutoEncoder implemented stacking an Encoder and a Decoder,
# and adding Gaussian Noise to the batches during training.

class DenoisingAutoEncoder(keras.Model):

    def __init__(
        self,
        neurons_encoder,   # list of integers
        neurons_decoder,   # list of integers
        stddev=0.1,   # std of the noise added during training
        activations_encoder=None,   # list of activations
        activations_decoder=None,   # list of activations
        name='denoising_autoencoder',
        **kwargs,
    ):
        
        # call parent constructor
        super(DenoisingAutoEncoder, self).__init__(name=name, **kwargs)
        
        # initialize encoder and decoder, save stddev of the noise
        self.encoder = Encoder(neurons_encoder, activations_encoder, name='encoder')
        self.decoder = Decoder(neurons_decoder, activations_decoder, name='decoder')
        self.stddev = tf.Variable(stddev, trainable=False, name='stddev', dtype=tf.float32)
    
    # define forward pass. Different behaviour during training and during inference:
    # when training, gaussian noise is added to the data that must be reconstructed.
    def call(self, inputs, training=False):
        x = inputs
        if training:
            noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=self.stddev)
            x = x + noise
        return self.decoder(self.encoder(x))
    
    # initialize weights
    def build(self, input_shape):
        self.encoder._build(input_shape)
        self.decoder._build(self.encoder.bottleneck)
        if not hasattr(self.build, '_is_default'):
            self._build_input_shape = input_shape
        self.built = True

    # return configuration of the instance
    def get_config(self):
        config = {
            "neurons_encoder": self.encoder.neurons,
            "neurons_decoder": self.decoder.neurons,
            "stddev": self.stddev.numpy(),
            "activations_encoder": self.encoder.activations,
            "activations_decoder": self.decoder.activations,
        }
        return config
    
    # re-initialize randomly the weights
    def reset_weights(self):
        print(f"\nRe-initializing weights of model {self.name}\n")
        self.encoder.reset_weights()
        self.decoder.reset_weights()


# a Replicated AutoEncoder, implemented as a number of replicas of the Encoder,
# a common Decoder, and a 'baricenter' Encoder that is used for inference.

class ReplicatedAutoEncoder(keras.Model):
    
    def __init__(
        self,
        neurons_encoder,   # list of integers
        neurons_decoder,   # list of integers
        n_replicas=5,   # number of replicas of the Encoder
        rate=0.001,   # rate of regularization
        activations_encoder=None,   # list of activations
        activations_decoder=None,   # list of activations
        name='replicated_autoencoder',
        **kwargs,
    ):
        
        # call parent constructor
        super(ReplicatedAutoEncoder, self).__init__(name=name, **kwargs)

        self.n_replicas = n_replicas

        # initialize replicas of the encoder
        replicas = []
        for i in range(n_replicas):
            replica = Encoder(neurons_encoder, activations_encoder, name=f"encoder_{i}")
            replicas.append(replica)
        self.replicas = replicas

        # initialize baricenter and decoder
        self.baricenter = Encoder(neurons_encoder, activations_encoder, name="baricenter")
        self.decoder = Decoder(neurons_decoder, activations_decoder, name="decoder")

        # set up the losses
        self.rate = tf.Variable(rate, trainable=False, name='rate', dtype=tf.float32)
        self.total_loss_tracker = keras.metrics.Mean(name='loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.elastic_loss_tracker = keras.metrics.Mean(name='distance_loss')
    
    # define the forward pass. During training, compute one reconstruction per replica of the 
    # encoder. During inference, use only the baricenter for reconstruction.
    def call(self, inputs, training=False):
        if training:
            return [self.decoder(replica(inputs)) for replica in self.replicas]
        else:
            return self.decoder(self.baricenter(inputs))
    
    # initialize weights
    def build(self, input_shape):
        for replica in self.replicas:
            replica._build(input_shape)
        self.baricenter._build(input_shape)
        self.decoder._build(self.baricenter.bottleneck)
        if not hasattr(self.build, '_is_default'):
            self._build_input_shape = input_shape
        self.built = True

    # list losses and metrics. Their state will be reset at the beginning of each epoch
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.elastic_loss_tracker,
        ]
    
    # The following method computes the average reconstruction error of the reconstructions
    # in y_pred (one per replica) with respect to y, as defined by the argument loss_fn.
    def loss_fn(self, y, y_pred, loss_fn_, sample_weight=None):
        loss = tf.constant(0, dtype=tf.float32)
        for reconstruction in y_pred:
            loss_replica = loss_fn_(y, reconstruction, sample_weight=sample_weight)
            loss += loss_replica
        return loss / self.n_replicas
    
    # The following method defines the distance function between the weights of two Encoders.
    # w1, w2 are lists of tensors containing the kernels and biases of the layers composing the
    # two encoders.x
    def dist(self, w1, w2):
        d = 0
        for a, b in zip(w1, w2):
            d += tf.norm(a-b, ord='euclidean') ** 2
        return d
    
    # customize the behaviour during training at the level of batches. fit repeatedly calls this method.
    def train_step(self, data):
        
        # get the batch data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        
        # Forward pass and loss computation
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            reconstruction_loss = self.loss_fn(y, y_pred, self.compiled_loss, sample_weight=sample_weight)
            elastic_loss = 0
            for replica in self.replicas:
                elastic_loss += self.dist(replica.weights, self.baricenter.weights)
            elastic_loss /= self.n_replicas
            total_loss = reconstruction_loss + self.rate * elastic_loss
        
        # weight update
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # loss update
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.elastic_loss_tracker.update_state(elastic_loss)
        
        return {m.name: m.result() for m in self.metrics}
    
    # define the behaviour during inference. model.evaluate gives wrong results!!!
    def test_step(self, data):
        x, y = data
        y_pred_baricenter = self(x, training=False)
        baricenter_loss = self.compiled_loss(y, y_pred_baricenter)
        return {'baricenter': baricenter_loss}
        
    # return configuration of the instance
    def get_config(self):
        config = {
            "neurons_encoder": self.baricenter.neurons,
            "neurons_decoder": self.decoder.neurons,
            "activations_encoder": self.baricenter.activations,
            "activations_decoder": self.decoder.activations,
            "n_replicas": self.n_replicas,
            "rate": self.rate.numpy(),
        }
        return config
    
    # re-initialize randomly the weights
    def reset_weights(self):
        print(f"\nRe-initializing weights of model {self.name}\n")
        for replica in self.replicas:
            replica.reset_weights()
        self.baricenter.reset_weights()
        self.decoder.reset_weights()



class AE(keras.Model):

    def __init__(self, architecture, activations=None, reg_lambda=0, **kwargs):
        super(AE, self).__init__()

        if activations is None:
            activations = ['relu']*len(architecture)

        layers = []
        for neurons, activation in zip(architecture, activations):
            layer = keras.layers.Dense(neurons, activation=activation,
                                kernel_regularizer=regularizers.L2(reg_lambda))
            layers.append(layer)
        self.own_layers = layers

    def call(self, inputs):
        x = inputs
        for layer in self.own_layers:
            x = layer(x)
        return x





