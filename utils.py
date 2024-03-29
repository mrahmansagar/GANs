# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:15:27 2023

@author: mrahm
"""

import os
from tkinter import Tcl
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import tifffile

from keras.utils import load_img, img_to_array
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K


# load images/data in shape for training data generation 
def load_images_in_shape(img_dir, **kwargs):
    """
    Load images from a directory and convert them to a NumPy array with specified shape.

    Args:
        img_dir (str): Path to the directory containing the images.
        **kwargs: Additional keyword arguments to be passed to `load_img()` function.

    Returns:
        numpy.ndarray: Array containing the loaded images in the specified shape.
    """
    img_data_in_shape = []
    
    list_of_images = Tcl().call('lsort', '-dict', os.listdir(img_dir))
    print('Found', len(list_of_images), 'files in the directory')
    
    for im in tqdm(list_of_images):
        im = load_img(os.path.join(img_dir, im), **kwargs)
        imarray = img_to_array(im)
        img_data_in_shape.append(imarray)
        
    img_data_in_shape = np.asarray(img_data_in_shape)
        
    print('Loaded', img_data_in_shape.shape, 'images')
    
    return img_data_in_shape
        
# Data scaling
def scale_data(data, min_pix_val=0, max_pix_val=255, final_activation='tanh'):
    """
    Scale input data to a specific range according to final activation function.

    Args:
        data (numpy.ndarray): Input data to be scaled.
        min_pix_val (int, optional): Minimum pixel value of the desired range. Default is 0.
        max_pix_val (int, optional): Maximum pixel value of the desired range. Default is 255.
        final_activation (str, optional): Activation function for the final scaled data. Default is 'tanh'.

    Returns:
        numpy.ndarray: Scaled data.
    """
    try:
        if final_activation == 'tanh':
            scale_coef = (max_pix_val - min_pix_val)/2.0

        scaled_data = (data - scale_coef) / scale_coef

        return scaled_data
    
    except:
        print('Error: Activation function type is not defined.')   


#saving the volume as series of slices 
def saveSlices(volume, folderName):
    depth = volume.shape[0]
    for aSlice in range(depth):
        img = volume[aSlice, :, :]
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        fName = os.path.join(folderName, f'slice_{aSlice}.tif') 
        tifffile.imwrite(fName, img)


#determine the number of conv block
def calculate_conv_block(init_dim, final_dim):

    found = False
    count = 0
    for i in range(int(final_dim/init_dim)):
        init_dim = init_dim*2
        count += 1
        if init_dim == final_dim:
            found = True
            return count
            break
        
    if found == False:
        print('Error: cannot match final dimension. Check init_dim, final_dim to match the output shape. Otherwise manipulate stride, padding, or add layers, etc.')


# generate random noise (latent points)
def generate_latent_points(latent_dim, batch_size):
    """
    Generate random noise (latent points) for use in a generative model.

    Parameters:
        latent_dim (int): The dimensionality of the latent space.
        batch_size (int): The number of latent points to generate in a single batch.

    Returns:
        numpy.ndarray: A 2D array of shape (batch_size, latent_dim) containing
                       random noise (latent points).
    """
    x = np.random.randn(latent_dim*batch_size)
    x = x.reshape(batch_size, latent_dim)
    return x


def generate_fake_samples(generator, latent_dim, batch_size):
    """
    Generates fake samples using a generator model.

    Args:
        generator (tf.keras.Model): The generator model used to generate fake samples.
        latent_dim (int): The dimension of the latent space.
        batch_size (int): The number of fake samples to generate in a batch.

    Returns:
        tuple: A tuple containing generated fake samples (x_f) and corresponding labels (y_f).
            - x_f (numpy.ndarray): An array of generated fake samples with shape (batch_size, sample_shape).
            - y_f (numpy.ndarray): An array of corresponding labels with shape (batch_size, 1).
    """
    latent_input = generate_latent_points(latent_dim, batch_size)
    x_f = generator.predict(latent_input)
    y_f = np.zeros(shape=(batch_size, 1))
    return x_f, y_f


def evaluate_model_performance(gen_model, latent_dim, iteration, name='gen'):
    """
    Evaluate and save the performance of a GAN-generated model.

    This function generates images using the provided generator model and evaluates 
    its performance.
    The generated images are saved in a 5x5 grid, and the trained generator 
    model is saved to disk.

    Parameters:
        gen_model (tf.keras.Model): The generator model to generate images.
        latent_dim (int): The dimension of the latent space for generating random points.
        iteration (int): The current iteration/epoch of the model training.
        name (str): A name or identifier for the generated images and model files.

    Returns:
        None
    """
    
    # generate images
    latent_points = generate_latent_points(latent_dim, 25)  #Latent dim and n_samples
    # generate images
    X = gen_model.predict(latent_points)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    
    import numpy as np
    X = (X*255).astype(np.uint8)
    
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5, 1 + i)
        plt.axis('off')
        plt.imshow(X[i, :, :, :])

    
    plt_name = os.path.join(name, f'output_after_{iteration}.png')
    plt.savefig(plt_name)
    plt.close()
    #saving the model at the iteration 
    model_name = os.path.join(name, f'model_after_{iteration}.h5')
    gen_model.save(model_name)


def training_log(log_dir=None, fileName=None):
    
    if not log_dir is None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    else:
        log_dir = os.getcwd()
    
    if fileName is None:
        logFile = os.path.join(log_dir, 'training_log.txt')
    else:
        logFile = os.path.join(log_dir, fileName)
    
    log_train = open(logFile, 'w')
    
    return log_train
    

"""
This is a helper(Normalization) fuction that is needed for Cycle-GANs. This 
is part of keras-contrib library. You can either install the keras-contrib library 
or use from here. 
The below code is copied from 
https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py

"""

class InstanceNormalization(Layer):
    """Instance normalization layer.

    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.

    # Output shape
        Same shape as input.

    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    


