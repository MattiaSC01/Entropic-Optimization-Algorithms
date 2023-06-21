import tensorflow as tf
from tensorflow import keras


# A Multi-Layer Perceptron implemented as a stack of keras.layers.Dense layers.

class MLP(keras.layers.Layer):

    def __init__(
        self, 
        neurons,       # list of integers
        activations,   # list of activations (supports strings)
        name='mlp', 
        l2_rate=None,       # whether to use l2 regularization on the weights
        **kwargs,
    ):
        
        assert len(neurons) == len(activations)   # check input
        super(MLP, self).__init__(name=name, **kwargs)   # call parent constructor
        
        # save configuration
        self.neurons = neurons
        self.activations = activations

        # build the layers and store them in a list
        activations = map(keras.activations.get, activations)
        layers = []
        for n_neurons, activation in zip(neurons, activations):
            if l2_rate is not None:
                layer = keras.layers.Dense(
                    n_neurons, 
                    activation=activation, 
                    kernel_regularizer=keras.regularizers.L2(l2_rate), 
                    bias_regularizer=keras.regularizers.L2(l2_rate)
                )
            else:
                layer = keras.layers.Dense(n_neurons, activation=activation)
            layers.append(layer)
        self.layers = layers
    
    # define the forward pass
    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
    
    # initialize weights based on input shape
    def _build(self, input_shape):
        last = input_shape
        for layer in self.layers:
            layer.build(last)
            last = layer.units

            # experimental
            self.built = True
    
    # returns the configuration of the layer, to reproduce it
    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            "neurons": self.neurons,
            "activations": self.activations,
        })
        return config
    
    # re-initialize the weights of all the layers, randomly.
    # weights attribute does not get updated (why??). get_weights() returns the updated weights instead.
    def reset_weights(self):
        for layer in self.layers:
            new_kernel = layer.kernel_initializer(layer.kernel.shape)
            new_bias = layer.bias_initializer(layer.bias.shape)
            layer.set_weights([new_kernel, new_bias])


# Essentially a MLP, it represents an Encoder

class Encoder(MLP):
    
    def __init__(
        self, 
        neurons,   # list of integers
        activations=None,   # list of activations (supports strings) 
        name='encoder', 
        **kwargs,
    ):
        
        # default activation functions
        if activations is None:
            activations = ['relu'] * len(neurons)
        
        # call parent constructor
        super(Encoder, self).__init__(neurons, activations, name, **kwargs)

        self.bottleneck = neurons[-1]
    
    # for portability, not really needed with tf.keras
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.bottleneck])


# Essentially a MLP, it represents a Decoder

class Decoder(MLP):

    def __init__(
        self, 
        neurons,   # list of integers
        activations=None,   # list of activations (supports strings)
        name='decoder', 
        **kwargs,
    ):
        
        # default activation functions
        if activations is None:
            activations = ['relu'] * (len(neurons) - 1) + [None]
        
        # call parent constructor
        super(Decoder, self).__init__(neurons, activations, name, **kwargs)

        self.input_dim = neurons[-1]
    
    # for portability, not really needed with tf.keras
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.input_dim])
