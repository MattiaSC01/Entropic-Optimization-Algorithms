from tensorflow import keras
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import os

# files needed for image elastic distortion
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from sklearn.model_selection import train_test_split


# A Multi-Layer Perceptron implemented as a stack of keras.layers.Dense layers.
# It implements the methods __init__, call, get_config.

class MLP(keras.layers.Layer):

    def __init__(
        self, 
        neurons,            # list of integers, each being the number of neurons of a layer
        activations=None,   # list of activations, one per layer.
        l2_rate=None,       # intensity of l2 regularization
        name='mlp', 
        **kwargs,           # keyword arguments for the parent class constructor
    ):
        
        # call parent constructor
        super(MLP, self).__init__(name=name, **kwargs)
        
        # default regularization rate
        if l2_rate is None:
            l2_rate = 0
        
        # default activations
        if activations is None:
            activations = [None] * len(neurons)
        
        # store initialization parameters
        self.neurons = neurons
        self.l2_rate = l2_rate
        self.activations = activations
        
        # initialize densely connected layers in a loop, and save them inside a list
        layers = []
        for n_neurons, activation in zip(neurons, activations):
            layer = keras.layers.Dense(
                n_neurons, 
                activation=activation, 
                kernel_regularizer=keras.regularizers.L2(l2_rate), 
                bias_regularizer=keras.regularizers.L2(l2_rate)
            )
            layers.append(layer)

        # make the layer list an attribute.
        self.layers = layers


    # define the forward pass
    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    # return the layer configuration
    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            "neurons": self.neurons,
            "activations": self.activations,
            "l2_rate": self.l2_rate,
        })
        return config


# A vanilla AutoEncoder implemented as a Multi-Layer Perceptron by stacking
# an Encoder and a Decoder, both instances of the MLP class.
# It implements the methods __init__, call, get_config.

class AutoEncoder(keras.Model):
    
    def __init__(
        self,
        neurons_encoder,            # list of integers, each being the number of neurons of a layer of the encoder
        neurons_decoder,            # list of integers, each being the number of neurons of a layer of the decoder
        activations_encoder=None,   # list of activations, one per layer of the encoder
        activations_decoder=None,   # list of activations, one per layer of the decoder
        l2_rate=0,                  # intensity of l2 regularization
        name='autoencoder',
        **kwargs,                   # keyword arguments for the parent class constructor
    ):
        
        # call parent constructor
        super(AutoEncoder, self).__init__(name=name, **kwargs)

        # default encoder activations
        if activations_encoder is None:
            activations_encoder = ['relu'] * len(neurons_encoder)
        
        # default decoder activations
        if activations_decoder is None:
            activations_decoder = ['relu'] * (len(neurons_decoder) - 1) + ['sigmoid']
        
        # store initialization parameters
        self.neurons_encoder = neurons_encoder
        self.neurons_decoder = neurons_decoder
        self.activations_encoder = activations_encoder
        self.activations_decoder = activations_decoder
        self.l2_rate = l2_rate

        # initialize the encoder
        self.encoder = MLP(
            neurons_encoder, 
            activations_encoder, 
            l2_rate, 
            name='encoder', 
        )

        # initialize the decoder
        self.decoder = MLP(
            neurons_decoder, 
            activations_decoder,
            l2_rate,  
            name='decoder', 
        )

    # define the forward pass
    def call(self, inputs):
        x = inputs
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    # return the model configuration, to enable saving of the model
    def get_config(self):
        config = {
            "neurons_encoder": self.neurons_encoder, 
            "neurons_decoder": self.neurons_decoder, 
            "activations_encoder": self.activations_encoder, 
            "activations_decoder": self.activations_decoder, 
            "l2_rate": self.l2_rate, 
        }
        return config


# A denoising AutoEncoder implemented as a Multi-Layer Perceptron by stacking
# a keras.layers.GaussianNoise layer, an Encoder and a Decoder, the last two being
# instances of the MLP class.
# It implements the methods __init__, call, get_config.

class DenoisingAutoEncoder(keras.Model):

    def __init__(
        self,
        neurons_encoder,                # list of integers, each being the number of neurons of a layer of the encoder
        neurons_decoder,                # list of integers, each being the number of neurons of a layer of the decoder
        activations_encoder=None,       # list of activations, one per layer of the encoder
        activations_decoder=None,       # list of activations, one per layer of the decoder
        stddev=0.1,                     # stddev of the gaussian noise applied during training
        l2_rate=0,                      # intensity of l2 regularization
        name='denoising_autoencoder',
        **kwargs,                       # keyword arguments for the parent class constructor
    ):
        
        # call parent constructor
        super(DenoisingAutoEncoder, self).__init__(name=name, **kwargs)

        # default encoder activations
        if activations_encoder is None:
            activations_encoder = ['relu'] * len(neurons_encoder)
        
        # default decoder activations
        if activations_decoder is None:
            activations_decoder = ['relu'] * (len(neurons_decoder) - 1) + ['sigmoid']
        
        # store initialization parameters
        self.neurons_encoder = neurons_encoder
        self.neurons_decoder = neurons_decoder
        self.activations_encoder = activations_encoder
        self.activations_decoder = activations_decoder
        self.stddev = stddev
        self.l2_rate = l2_rate
        
        # initialize noise layer
        self.gaussian_noise = keras.layers.GaussianNoise(stddev)

        # initialize the encoder
        self.encoder = MLP(
            neurons_encoder, 
            activations_encoder, 
            l2_rate, 
            name='encoder', 
        )

        # initialize the decoder
        self.decoder = MLP(
            neurons_decoder, 
            activations_decoder,
            l2_rate,  
            name='decoder', 
        )

    # define the forward pass. During training, apply Gaussian noise to the inputs before feeding them to the autoencoder.
    # During inference, do not add noise.
    def call(self, inputs, training=False):
        x = inputs
        x = self.gaussian_noise(x, training=training)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    # return the model configuration, to enable saving of the model
    def get_config(self):
        config = {
            "neurons_encoder": self.neurons_encoder,
            "neurons_decoder": self.neurons_decoder,
            "activations_encoder": self.activations_encoder,
            "activations_decoder": self.activations_decoder,
            "stddev": self.stddev, 
            "l2_rate": self.l2_rate, 
        }
        return config


# a Replicated AutoEncoder with replicas of the full autoencoder, each being an instance
# of the AutoEncoder class.
# It implements the methods __init__, call, get_config, train_step, test_step.

class FullReplicatedAutoEncoder(keras.Model):

    def __init__(
        self,
        neurons_encoder,                      # list of integers, each being the number of neurons of a layer of the encoder
        neurons_decoder,                      # list of integers, each being the number of neurons of a layer of the decoder
        activations_encoder=None,             # list of activations, one per layer of the encoder
        activations_decoder=None,             # list of activations, one per layer of the decoder
        n_replicas=5,                         # number of replicas of the autoencoder
        distance_rate=1e-6,                   # control how strong an incentive the replicas have to stay close together
        name='full_replicated_autoencoder',   
        **kwargs,                             # keyword arguments for the parent class constructor
    ):
        
        # call parent constructor
        super(FullReplicatedAutoEncoder, self).__init__(name=name, **kwargs)

        # default encoder activations
        if activations_encoder is None:
            activations_encoder = ['relu'] * len(neurons_encoder)
        
        # default decoder activations
        if activations_decoder is None:
            activations_decoder = ['relu'] * (len(neurons_decoder) - 1) + ['sigmoid']

        # store initialization parameters
        self.neurons_encoder = neurons_encoder
        self.neurons_decoder = neurons_decoder
        self.activations_encoder = activations_encoder
        self.activations_decoder = activations_decoder
        self.n_replicas = n_replicas

        # initialize a variable storing the distance regularization rate
        self.distance_rate = tf.Variable(distance_rate, trainable=False, name='rate', dtype=tf.float32)

        # store input dimension
        self.input_dim = neurons_decoder[-1]

        # initialize replicas of the autoencoder in a loop, and save them inside a list
        replicas = []
        for i in range(n_replicas):
            replica = AutoEncoder(
                neurons_encoder, 
                neurons_decoder, 
                activations_encoder, 
                activations_decoder, 
                name=f"autoencoder_{i}"
            )
            replicas.append(replica)
        
        # save the list of replicas as an attribute
        self.replicas = replicas

        # initialize the baricenter
        self.baricenter = AutoEncoder(
                neurons_encoder, 
                neurons_decoder, 
                activations_encoder, 
                activations_decoder, 
                name=f"baricenter"
            )

        # set up the losses
        self.total_loss_tracker = keras.metrics.Mean(name='loss')                           # total loss, to be minimized
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')   # average reconstruction loss of replicas
        self.baricenter_loss_tracker = keras.metrics.Mean(name='baricenter_loss')           # reconstruction loss of the baricenter
        self.distance_loss_tracker = keras.metrics.Mean(name='distance_loss')               # average distance of replicas from baricenter
    
    # define the forward pass. It returns a list of reconstructions, the first being obtained by the baricenter,
    # and the following ones by the replicas.
    def call(self, inputs):
        return [self.baricenter(inputs)] + [replica(inputs) for replica in self.replicas]
    
    # return the model configuration to enable saving
    def get_config(self):
        config = {
            "neurons_encoder": self.neurons_encoder,
            "neurons_decoder": self.neurons_decoder,
            "activations_encoder": self.activations_encoder,
            "activations_decoder": self.activations_decoder,
            "n_replicas": self.n_replicas,
            "distance_rate": self.distance_rate.numpy(),
        }
        return config
    
    # compute the distance of each replica from the baricenter, and return the distances in a list
    def distances(self):
        distances = []
        for replica in self.replicas:
            d = 0
            for w1, w2 in zip(replica.weights, self.baricenter.weights):
                d += tf.norm(w1 - w2) ** 2   # since we compute the squared euclidean distance, we can do it piece-wise
            distances.append(d)
        return distances
    
    # define the training behaviour when the model is presented with a batch of data
    def train_step(self, data):

        # unpack the batch data
        x, y = data

        # to compute reconstruction error
        mse = keras.losses.MeanSquaredError()

        # forward pass and computation of losses
        with tf.GradientTape() as tape:
            
            # compute the reconstructions given by the replicas and by the baricenter
            reconstructions = self(x)
            reconstructions_replicas = reconstructions[0:]
            reconstruction_baricenter = reconstructions[0]

            # compute the average reconstruction error of the replicas
            reconstruction_loss = 0
            for reconstruction in reconstructions_replicas:
                loss = mse(y, reconstruction)
                reconstruction_loss += loss
            reconstruction_loss /= self.n_replicas

            # compute the average squared euclidean distance between the weights of the baricenter and those of the replicas.
            distance_loss = 0
            for replica in self.replicas:
                d = 0
                for w1, w2 in zip(replica.weights, self.baricenter.weights):
                    d += tf.norm(w1 - w2) ** 2   # since we compute the squared euclidean distance, we can do it piece-wise
                distance_loss += d
            distance_loss /= self.n_replicas

            # compute the total loss
            total_loss = reconstruction_loss + self.distance_rate * distance_loss
        
        # compute the reconstruction error of the baricenter
        baricenter_loss = mse(y, reconstruction_baricenter)

        # update trainable weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # update the state of the loss trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.baricenter_loss_tracker.update_state(baricenter_loss)
        self.distance_loss_tracker.update_state(distance_loss)

        # return the current state of the losses
        return {m.name: m.result() for m in self.metrics}
    
    # define the testing behaviour when the model is presented with a batch of data
    def test_step(self, data):
        
        # unpack the batch data
        x, y = data

        # to compute reconstruction error
        mse = keras.losses.MeanSquaredError()

        # compute the reconstructions given by the replicas and by the baricenter
        reconstructions = self(x)
        reconstructions_replicas = reconstructions[0:]
        reconstruction_baricenter = reconstructions[0]

        # compute the average reconstruction error of the replicas
        reconstruction_loss = 0
        for reconstruction in reconstructions_replicas:
            loss = mse(y, reconstruction)
            reconstruction_loss += loss
        reconstruction_loss /= self.n_replicas

        # compute the reconstruction error of the baricenter
        baricenter_loss = mse(y, reconstruction_baricenter)

        # compute the average squared euclidean distance between the weights of the baricenter and those of the replicas.
        distance_loss = 0
        for replica in self.replicas:
            d = 0
            for w1, w2 in zip(replica.weights, self.baricenter.weights):
                d += tf.norm(w1 - w2) ** 2   # since we compute the squared euclidean distance, we can do it piece-wise
            distance_loss += d
        distance_loss /= self.n_replicas

        # compute the total loss
        total_loss = reconstruction_loss + self.distance_rate * distance_loss

        # update the state of the loss trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.baricenter_loss_tracker.update_state(baricenter_loss)
        self.distance_loss_tracker.update_state(distance_loss)

        # return the current state of the losses
        return {m.name: m.result() for m in self.metrics}

    
    # metrics listed here have their state reset at the beginning of every epoch when training with fit, and
    # at the beginning of every call to evaluate
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.baricenter_loss_tracker, 
            self.distance_loss_tracker,
        ]


# a Replicated AutoEncoder with replicas of the encoder only, each being an instance
# of the MLP class, and a common decoder, also of the MLP class.
# It implements the methods __init__, call, get_config, train_step, test_step.

class ReplicatedAutoEncoder(keras.Model):

    def __init__(
        self,
        neurons_encoder,                      # list of integers, each being the number of neurons of a layer of the encoder
        neurons_decoder,                      # list of integers, each being the number of neurons of a layer of the decoder
        activations_encoder=None,             # list of activations, one per layer of the encoder
        activations_decoder=None,             # list of activations, one per layer of the decoder
        n_replicas=5,                         # number of replicas of the autoencoder
        distance_rate=1e-6,                   # control how strong an incentive the replicas have to stay close together
        name='full_replicated_autoencoder',   
        **kwargs,                             # keyword arguments for the parent class constructor
    ):
        
        # call parent constructor
        super(ReplicatedAutoEncoder, self).__init__(name=name, **kwargs)

        # default encoder activations
        if activations_encoder is None:
            activations_encoder = ['relu'] * len(neurons_encoder)
        
        # default decoder activations
        if activations_decoder is None:
            activations_decoder = ['relu'] * (len(neurons_decoder) - 1) + ['sigmoid']

        # store initialization parameters
        self.neurons_encoder = neurons_encoder
        self.neurons_decoder = neurons_decoder
        self.activations_encoder = activations_encoder
        self.activations_decoder = activations_decoder
        self.n_replicas = n_replicas
        
        # initialize a variable storing the distance regularization rate
        self.distance_rate = tf.Variable(distance_rate, trainable=False, name='distance_rate', dtype=tf.float32)

        # store input dimension
        self.input_dim = neurons_decoder[-1]
        
        # initialize replicas of the autoencoder in a loop, and save them inside a list
        replicas = []
        for i in range(n_replicas):
            replica = MLP(
                neurons_encoder, 
                activations_encoder, 
                name=f"encoder_{i}"
            )
            replicas.append(replica)
        
        # save the list of replicas as an attribute
        self.replicas = replicas

        # initialize the baricenter
        self.baricenter = MLP(
            neurons_encoder, 
            activations_encoder, 
            name=f"baricenter", 
        )

        # initialize the common decoder
        self.decoder = MLP(
            neurons_decoder, 
            activations_decoder, 
            name='decoder', 
        )

        # set up the losses
        self.total_loss_tracker = keras.metrics.Mean(name='loss')                           # total loss, to be minimized
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')   # average reconstruction loss of replicas
        self.baricenter_loss_tracker = keras.metrics.Mean(name='baricenter_loss')           # reconstruction loss of the baricenter
        self.distance_loss_tracker = keras.metrics.Mean(name='distance_loss')               # average distance of replicas from baricenter
    
    # define the forward pass. It returns a list of reconstructions, the first being obtained by the baricenter,
    # and the following ones by the replicas.
    def call(self, inputs):
        return [self.decoder(self.baricenter(inputs))] + [self.decoder(replica(inputs)) for replica in self.replicas]
    
    # return the model configuration to enable saving
    def get_config(self):
        config = {
            "neurons_encoder": self.neurons_encoder,
            "neurons_decoder": self.neurons_decoder,
            "activations_encoder": self.activations_encoder,
            "activations_decoder": self.activations_decoder,
            "n_replicas": self.n_replicas,
            "distance_rate": self.distance_rate.numpy(),
        }
        return config
    
    # compute the distance of each replica from the baricenter, and return the distances in a list
    def distances(self):
        distances = []
        for replica in self.replicas:
            d = 0
            for w1, w2 in zip(replica.weights, self.baricenter.weights):
                d += tf.norm(w1 - w2) ** 2   # since we compute the squared euclidean distance, we can do it piece-wise
            distances.append(d)
        return distances

    # define the training behaviour when the model is presented with a batch of data
    def train_step(self, data):

        # unpack the batch data
        x, y = data

        # to compute reconstruction error
        mse = keras.losses.MeanSquaredError()

        # forward pass and computation of losses
        with tf.GradientTape() as tape:
            
            # compute the reconstructions given by the replicas and by the baricenter
            reconstructions = self(x, training=True)
            reconstructions_replicas = reconstructions[0:]
            reconstruction_baricenter = reconstructions[0]

            # compute the average reconstruction error of the replicas
            reconstruction_loss = 0
            for reconstruction in reconstructions_replicas:
                loss = mse(y, reconstruction)
                reconstruction_loss += loss
            reconstruction_loss /= self.n_replicas

            # compute the average squared euclidean distance between the weights of the baricenter and those of the replicas.
            distance_loss = 0
            for replica in self.replicas:
                d = 0
                for w1, w2 in zip(replica.weights, self.baricenter.weights):
                    d += tf.norm(w1 - w2) ** 2   # since we compute the squared euclidean distance, we can do it piece-wise
                distance_loss += d
            distance_loss /= self.n_replicas

            # compute the total loss
            total_loss = reconstruction_loss + self.distance_rate * distance_loss
        
        # compute the reconstruction error of the baricenter
        baricenter_loss = mse(y, reconstruction_baricenter)

        # update trainable weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # update the state of the loss trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.baricenter_loss_tracker.update_state(baricenter_loss)
        self.distance_loss_tracker.update_state(distance_loss)

        # return the current state of the losses
        return {m.name: m.result() for m in self.metrics}
    
    # define the testing behaviour when the model is presented with a batch of data
    def test_step(self, data):
        
        # unpack the batch data
        x, y = data

        # to compute reconstruction error
        mse = keras.losses.MeanSquaredError()

        # compute the reconstructions given by the replicas and by the baricenter
        reconstructions = self(x, training=True)
        reconstructions_replicas = reconstructions[0:]
        reconstruction_baricenter = reconstructions[0]

        # compute the average reconstruction error of the replicas
        reconstruction_loss = 0
        for reconstruction in reconstructions_replicas:
            loss = mse(y, reconstruction)
            reconstruction_loss += loss
        reconstruction_loss /= self.n_replicas

        # compute the reconstruction error of the baricenter
        baricenter_loss = mse(y, reconstruction_baricenter)

        # compute the average squared euclidean distance between the weights of the baricenter and those of the replicas.
        distance_loss = 0
        for replica in self.replicas:
            d = 0
            for w1, w2 in zip(replica.weights, self.baricenter.weights):
                d += tf.norm(w1 - w2) ** 2   # since we compute the squared euclidean distance, we can do it piece-wise
            distance_loss += d
        distance_loss /= self.n_replicas

        # compute the total loss
        total_loss = reconstruction_loss + self.distance_rate * distance_loss

        # update the state of the loss trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.baricenter_loss_tracker.update_state(baricenter_loss)
        self.distance_loss_tracker.update_state(distance_loss)

        # return the current state of the losses
        return {m.name: m.result() for m in self.metrics}

    
    # metrics listed here have their state reset at the beginning of every epoch when training with fit, and
    # at the beginning of every call to evaluate
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.baricenter_loss_tracker, 
            self.distance_loss_tracker,
        ]


# load the state of a model and of its optimizer (as saved by model.save_weights())

def load_model(
    path,                 # path where model's weights were saved (as by model.save_weights())
    cls,                  # class to which the model belong (e.g. AutoEncoder)
    config,               # model configuration (as returned by model.get_config())
    batch,                # a batch from the training data, as a numpy array
    compile_config=None   # compile configuration of the model
):

    # default compile configuration
    if compile_config is None:
        compile_config = {"optimizer": 'adam', "loss": 'mse'}
    
    # instantiate and compile the model
    loaded = cls(**config)
    loaded.compile(**compile_config)

    # train the model on a batch to initialize its weights and those of the optimizer
    loaded.train_on_batch(batch, batch)

    # load weights into the model
    loaded.load_weights(path)

    return loaded


# subsample the data to have a smaller training size and return the small dataset.

def subsample(
    data,             # the data to subsample, as a numpy array
    n_samples=None,   # an integer representing the number of samples to keep. It must be strictly less than the total number of samples. If None, return the full data.
    stratify=None,    # stratify argument passed to sklearn's train_test_split
    seed=42,          # sklearn's train_test_split random_state, for reproducibility
):

    if n_samples is None:
        return data
    
    x, _ = train_test_split(data, train_size=n_samples, stratify=stratify, random_state=seed)
    return x


# add 0-centered gaussian noise to the data, and return the corrupted version

def add_gaussian_noise(
    data,         # data to be corrupted
    stddev=0.1,   # stddev of the gaussian noise
    seed=42       # for reproducibility
):
    noise = tf.random.normal(shape=data.shape, mean=0.0, stddev=stddev, seed=seed)
    return data + noise


# from Kaggle. Applies elastic deformation to MNIST digit

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


# takes as input a replicated autoencoder model and returns its baricenter as an independent
# AutoEncoder instance (if only the encoder is replicated, stack the decoder on top of it).

def extract_baricenter(model):

    # check if the full autoencoder is replicated or only the encoder
    full = isinstance(model, FullReplicatedAutoEncoder)

    # retrieve input dimension
    input_dim = model.input_dim

    # retrieve configuration and weights of the autoencoder that must be instantiated
    
    if full:
        # get configuration
        config = model.baricenter.get_config()
        # get weights
        weights = model.baricenter.get_weights()
        
    else:
        # get configuration
        config = model.get_config()
        del config['n_replicas']
        del config['distance_rate']
        
        # get weights
        weights = model.baricenter.get_weights() + model.decoder.get_weights()
    
    # instantiate a vanilla AutoEncoder and initialize its weights
    ae = AutoEncoder(**config)
    ae.build((None, input_dim))

    # set the weights of the baricenter
    ae.set_weights(weights)
    
    return ae


# plot dataframe as returned by input_robustness and weight_robustness

def plot_errors(errors, x_axis, alpha=0.2, title='', percentage=False):
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
    means = errors.iloc[:, 0]
    stds = errors.iloc[:, 1]

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
    
    # ylabel
    ylabel = 'MSE error'
    if percentage:
        ylabel = 'Percentage increase in ' + ylabel
    plt.ylabel(ylabel)


# set up path to save figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

# save a pyplot figure in memory. It must be called before plt.show() (from Hands On textbook)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# plot MNIST digits in a grid. It does not call plt.show() to enable saving (from Hands On textbook)

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


# handle the regularization rate during training of a replicated autoencoder
# by applying an exponential schedule.

class ExponentialRateCallback(keras.callbacks.Callback):
    
    def __init__(
        self, 
        update_coeff=0.1,     # rate is updated multiplying it by 1 + update_coeff
        frequency=None,       # if None, update rate every epoch. if frequency is an integer, update every frequency batches.
        initial_value=None,   # if not None, the rate is set to that value at the beginning of training
        verbose=0,            # pass 1 to have a message printed when the rate is updated
        **kwargs,             # arguments for parent constructor
    ):
        # call parent constructor
        super(ExponentialRateCallback, self).__init__(**kwargs)
                
        # initialize batch counter
        self.counter = 0
        
        # store parameters
        self.frequency = frequency
        self.initial_value = initial_value
        self.update_coeff = update_coeff
        self.verbose = verbose
    
    # if an initial value is provided, set it
    def on_train_begin(self, logs=None):
        if self.initial_value is not None:
            value = self.initial_value
            self.model.distance_rate.assign(tf.constant(value, dtype=tf.float32))
    
    def on_train_batch_begin(self, batch, logs=None):
        
        # update batch counter
        self.counter += 1

        # if the time is ripe, update rate
        if self.frequency is not None and self.counter % self.frequency == 0:
            current = self.model.distance_rate.read_value()
            new = current * (1 + self.update_coeff)
            if self.verbose:
                print(f"\ndistance rate: {new}")
            self.model.distance_rate.assign(tf.constant(new, dtype=tf.float32))
    
    # if self.frequency is None, update rate
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0 and self.frequency is None:
            current = self.model.distance_rate.read_value()
            new = current * (1 + self.update_coeff)
            if self.verbose:
                print(f"\ndistance rate: {new}")
            self.model.distance_rate.assign(tf.constant(new, dtype=tf.float32))


# stop training of a model when the monitored loss gets below epsilon

class StopTrainingCallback(keras.callbacks.Callback):
    
    def __init__(
        self, 
        epsilon=1e-3, 
        monitor='loss', 
        patience=0,
        **kwargs
    ):
        
        # call parent constructor
        super(StopTrainingCallback, self).__init__(**kwargs)
        
        # store parameters
        self.epsilon = epsilon
        self.monitor = monitor
        self.patience = patience
        
        self.flag = False   # True if epsilon has been reached by monitored loss
        self.count = 0      # epochs since epsilon has been reached
    
    def on_epoch_end(self, epoch, logs=None):
        
        # get current value of loss
        current = logs[self.monitor]
        
        # if epsilon has been reached, increase count
        if self.flag:
            self.count += 1
        # else, check if epsilon has been reached
        elif current < self.epsilon:
            print(f"\n\n{self.monitor} hit {self.epsilon}!")
            self.flag = True
        
        # if epsilon has been reached since long enough, stop training
        if self.flag and self.count >= self.patience:
            self.model.stop_training = True


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
