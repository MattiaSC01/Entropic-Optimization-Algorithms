from tensorflow import keras
import tensorflow as tf


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
