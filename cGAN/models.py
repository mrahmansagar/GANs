# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 09:39:44 2023

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany

Conditional GAN implementation
"""

# importing necessary libraries 
import os 
import numpy as np


from sklearn.utils import shuffle

# importing tensorflow and keras
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Embedding, Concatenate, BatchNormalization, Dropout
from keras.optimizers import Adam

from .. import utils

# building the discriminator model
def build_discriminator(in_shape, n_class, noise_dim=50, optimizer=Adam, 
                        lr=0.0002, beta1=0.5, loss='binary_crossentropy', metrics=['accuracy']):
    """
    Build a discriminator model for a conditional GAN.

    This function constructs a discriminator model for a conditional GAN 
    (Generative Adversarial Network). The discriminator takes both image and 
    class label inputs and predicts whether the input image is real or fake.

    Args:
        in_shape (tuple): The shape of the input images (height, width, channels).
        n_class (int): The number of classes or categories for conditioning.
        noise_dim (int, optional): The dimension of noise used for conditioning (default is 50).
        optimizer (class, optional): The optimizer to use when compiling the model (default is Adam).
        lr (float, optional): The learning rate for the optimizer (default is 0.0002).
        beta1 (float, optional): The exponential decay rate for the first moment estimate in Adam (default is 0.5).
        loss (str, optional): The loss function to use during training (default is 'binary_crossentropy').
        metrics (list of str, optional): The evaluation metrics to use during training (default is ['accuracy']).

    Returns:
        keras.Model: The compiled discriminator model.
    """
    
    # label input 
    input_label = Input(shape=(1,))
    li = Embedding(n_class, noise_dim)(input_label)
    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)
    li = Reshape((in_shape[0], in_shape[1], 1))(li)
    
    # image input 
    input_img = Input(shape=in_shape)
    
    # merge
    dis = Concatenate()([input_img, li])
    dis = Conv2D(128, (3,3), strides=(2,2), padding='same')(dis)
    dis = LeakyReLU(alpha=0.2)(dis)
    dis = Conv2D(128, (3,3), strides=(2,2), padding='same')(dis) 
    dis = LeakyReLU(alpha=0.2)(dis)
    dis = Flatten()(dis)
    dis = Dropout(0.4)(dis)
    # output
    out_layer = Dense(1, activation='sigmoid')(dis)  #Shape=1
    
    dis_model = Model([input_img, input_label], out_layer)
    opt = optimizer(learning_rate=lr, beta_1=beta1)
    dis_model.compile(loss=loss, optimizer=opt, metrics=metrics)
    
    return dis_model



# building the generator model
def build_generator(noise_dim, n_class, init_dim, final_dim, n_channel):
    """
    Build a generator model for generating images.This function creates a 
    generator model that generates images from random noise and class labels. 
    It combines the noise vector and class label vector to produce an image of 
    the desired size

    Args:
        noise_dim (int, optional): The dimensionality of the noise vector. Default is 50.
        n_class (int, optional): The number of classes or labels for conditioning. Default is 10.

    Returns:
        tf.keras.Model: The generator model that takes noise and class labels
        as input and generates images as output.

    This function creates a generator model that generates images from random noise
    and class labels. It combines the noise vector and class label vector to produce
    an image of the desired size (8x8 in this case).
    """
    num_conv_block = utils.calculate_conv_block(init_dim, final_dim)
    
    if not num_conv_block == None:
    
        # Class Lebel vector 
        in_label = Input(shape=(1,))
        li = Embedding(n_class, noise_dim)(in_label)
        n_nodes = init_dim * init_dim 
        li = Dense(n_nodes)(li)
        li = Reshape((init_dim, init_dim, 1))(li)
    
        # Noise vector 
        in_noise = Input(noise_dim)
        n_nodes = 128 * init_dim * init_dim
        ni = Dense(n_nodes)(in_noise)
        ni = LeakyReLU(alpha=0.2)(ni)
        ni = Reshape((init_dim, init_dim, 128))(ni)
        
        # Now merging the noise vector and class lebel vector to generate image of desired size
        gen = Concatenate()([ni, li]) 
        
        for i in range(num_conv_block):
            gen = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(gen)
            gen = LeakyReLU(alpha=0.2)(gen)
            gen = BatchNormalization()(gen)

    out_layer = Conv2D((n_channel), (3,3), activation='tanh', padding='same')(gen)
    gen_model = Model([in_noise, in_label], out_layer)
    
    return gen_model




def build_cgan(generator, discriminator, optimizer=Adam, lr=0.0002, beta1=0.5, 
                        loss='binary_crossentropy'):
    """
  Build a Conditional Generative Adversarial Network (CGAN) model.

  This function creates a CGAN model by combining a generator and a discriminator.
  The discriminator is set to be not trainable during training.

  Args:
      generator (tf.keras.Model): The generator model.
      discriminator (tf.keras.Model): The discriminator model.
      optimizer (tf.keras.optimizers.Optimizer, optional): The optimizer to use for training.
          Defaults to Adam optimizer.
      lr (float, optional): The learning rate for the optimizer. Defaults to 0.0002.
      beta1 (float, optional): The exponential decay rate for the first moment estimates
          in the optimizer. Defaults to 0.5.
      loss (str, optional): The loss function to use for training. Defaults to 'binary_crossentropy'.

  Returns:
      tf.keras.Model: The CGAN model.

  Note:
      - The discriminator is set to be not trainable in the CGAN model.
      - The `generator` and `discriminator` models should be properly configured
        before calling this function.

  Example:
      >>> cgan_model = build_cgan(generator_model, discriminator_model)
  """
    
    #make discriminator model not trainable
    discriminator.trainable = False
    
    # getting the inputs of generator
    gen_noise, gen_label = generator.input
    # getting the output of generator model
    gen_output = generator.output
    # getting the discriminator output using the output of generator and lebel as input  
    dis_out = discriminator([gen_output, gen_label])
    # connecting inputs and output of generator and discriminator to make new cgan model
    cgan_model = Model([gen_noise, gen_label], dis_out)
    
    opt = optimizer(learning_rate=lr, beta_1=beta1)
    cgan_model.compile(loss=loss, optimizer=opt)
    
    return cgan_model
    