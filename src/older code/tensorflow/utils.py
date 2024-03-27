import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# files needed for image elastic distortion
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from sklearn.model_selection import train_test_split

from models import FullReplicatedAutoEncoder, AutoEncoder


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


# load the state of a model and of its optimizer, as saved by train
def load_model(cls, config, compile_config, path, batch):

    loaded = cls(**config)
    loaded.compile(**compile_config)

    loaded.train_on_batch(batch, batch)
    loaded.load_weights(path)

    return loaded
