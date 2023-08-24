# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:55:41 2023

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany
"""

# importing necessary libraries 
import os 
import numpy as np


# importing tensorflow and keras
import tensorflow as tf

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.layers import LeakyReLU, Dropout


# defining a function to build discriminator model
def build_discriminator(input_shape, optimizer=Adam, lr=0.0002, beta1=0.5, 
                        loss='binary_crossentropy', metrics=['accuracy']):
    
    """
    Build and compile a discriminator model.

    This function constructs a convolutional neural network based discriminator 
    model for an adversarial network.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        optimizer (class, optional): Optimizer class to use for compiling the model. 
        Default is Adam.
        lr (float, optional): Learning rate for the optimizer. 
        Default is 0.0002.
        beta1 (float, optional): Exponential decay rate for the first moment 
        estimates in Adam optimizer. Default is 0.5.
        loss (str, optional): Loss function to use. 
        Default is 'binary_crossentropy'.
        metrics (list, optional): List of metrics to monitor during training. 
        Default is ['accuracy'].

    Returns:
        tensorflow.keras.models.Model: Compiled discriminator model.
    """
    
    
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = optimizer(learning_rate=lr, beta_1=beta1)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    return model



def build_generator(latent_dim, n_channel=3):
    
    """
    Build a generator model for generating images using a deep neural network.

    Args:
        latent_dim (int): Size of the input random noise vector.
        n_channel (int, optional): Number of channels in the generated images.
        Defaults to 3.

    Returns:
        keras.models.Sequential: A Keras Sequential model representing the generator.
    """
    
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer
    model.add(Conv2D(n_channel, (3,3), activation='tanh', padding='same'))
    return model


def build_gan(generator, discriminator, optimizer=Adam, lr=0.0002, beta1=0.5, 
                        loss='binary_crossentropy'):
    
    """
    Build a GAN (Generative Adversarial Network) model.

    This function constructs a GAN model by connecting a generator and a discriminator.
    The discriminator is set to be non-trainable, as it will be trained separately.

    Args:
        generator (tensorflow.keras.Model): The generator model.
        discriminator (tensorflow.keras.Model): The discriminator model.
        optimizer (tensorflow.keras.optimizers.Optimizer): The optimizer for 
        the GAN model. Default is Adam.
        lr (float): The learning rate for the optimizer. Default is 0.0002.
        beta1 (float): The beta1 parameter for the optimizer. Default is 0.5.
        loss (str): The loss function to use. Default is 'binary_crossentropy'.

    Returns:
        tensorflow.keras.Model: The compiled GAN model.

    """
    
    #Discriminator is trained separately. So set to not trainable.
    discriminator.trainable = False
    # connect generator and discriminator
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    # compile model
    opt = optimizer(learning_rate=lr, beta_1=beta1)
    model.compile(loss=loss, optimizer=opt)
    return model


#todo:
#def train_gan(generator, discriminator, gan, ):
    
 





































    
