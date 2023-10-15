from torch import nn


# basic AutoEncoder shell. Tolerates different architectures for encoder and decoder.
# l2 regularization and injection of noise on inputs are decided at train time.

class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


# create a sequence of alternating linear layers and activation functions. Initialize weights (biases use
# default init). return a list of layers. Used to instantiate encoder and decoder.

def makelist(
        neurons,                # layer widths, including input and output (e.g. [784, 784, 14])
        activations,            # activations (e.g. [nn.ReLU(), nn.Sigmoid()])
        initializations=None,   # weight initializers in-place (e.g. [nn.init.xavier_uniform_, nn.init.xavier_uniform_])
        bias=True,              # whether to include bias in linear layers
):

    # default initialization
    if initializations is None:
        initializations = len(activations)*[nn.init.xavier_uniform_]

    last = neurons[0]
    layers = []

    for n, act, init in zip(neurons[1:], activations, initializations):

        # instantiate linear layer
        linear = nn.Linear(last, n, bias=bias)

        # initialize weights (does this initialize bias?) - notice initialization in-place
        if init == nn.init.kaiming_uniform_:
            init(linear.weight, nonlinearity='relu')   # only support relu (not leaky_relu) for simplicity
        else:
            init(linear.weight)

        # add layer
        layers.append(linear)
        last = n

        # add activation function
        if act is not None:
            layers.append(act)

    return layers