# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:02:22 2023

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany

"""

# importing necessary libraries 
import os 
import numpy as np


# importing tensorflow and keras
import tensorflow as tf

from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate
from keras.layers import Activation, LeakyReLU
from keras.layers import BatchNormalization, Dropout


# building the discriminator model 
def build_discriminator(input_shape, opt=Adam, lr=0.0002, beta1=0.5, 
                        loss='binary_crossentropy', loss_weights=[0.5]):
    """
    Build and compile a discriminator model for use in image-to-image translation
    using the PatchGAN architecture.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        opt (Optimizer, optional): The optimizer class to use for compiling the model. 
        Default is Adam.
        lr (float, optional): Learning rate for the optimizer. Default is 0.0002.
        beta1 (float, optional): Exponential decay rate for the first moment 
        estimates in Adam optimizer. Default is 0.5.
        loss (str, optional): Loss function to be used for training.
        Default is 'binary_crossentropy'.
        loss_weights (list, optional): List of loss weights. Default is [0.5].

    Returns:
        keras.models.Model: Compiled discriminator model.
    """
    
    
    # weight initialization according to the original pix2pix paper
    init = RandomNormal(stddev=0.02)
    
    src_input = Input(shape=input_shape)
    tar_input = Input(shape=input_shape)
    
    #concatenate
    merge = Concatenate()([src_input, tar_input])
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', 
               kernel_initializer=init)(merge)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', 
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', 
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', 
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(512, (4,4), padding='same', 
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(1, (4,4), padding='same', 
               kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # the reason there is no flatten and Dense layer at the end is because the original paper uses the 
    # patchGAN loss. So there is not a single error rather an array of error for each patch. 
    
    # define model
    model = Model([src_input, tar_input], patch_out)
    
    # compile model
    opt = Adam(learning_rate=lr, beta_1=beta1)
    model.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    return model




# Generator model 
def build_generator(input_shape):
    """
    Build a generator model for image-to-image translation using the U-Net architecture.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).

    Returns:
        keras.models.Model: Generator model.
    """
    init = RandomNormal(stddev=0.02)
    
    input_img = Input(shape=input_shape)
    
    # encoder block
    e1 = Conv2D(64, (4,4), strides=(2,2), padding='same', 
                kernel_initializer=init)(input_img)
    e1 = LeakyReLU(alpha=0.2)(e1)
    
    e2 = Conv2D(128, (4,4), strides=(2,2), padding='same', 
                kernel_initializer=init)(e1)
    e2 = BatchNormalization()(e2, training=True)
    e2 = LeakyReLU(alpha=0.2)(e2)
    
    e3 = Conv2D(256, (4,4), strides=(2,2), padding='same', 
                kernel_initializer=init)(e2)
    e3 = BatchNormalization()(e3, training=True)
    e3 = LeakyReLU(alpha=0.2)(e3)
    
    e4 = Conv2D(512, (4,4), strides=(2,2), padding='same', 
                kernel_initializer=init)(e3)
    e4 = BatchNormalization()(e4, training=True)
    e4 = LeakyReLU(alpha=0.2)(e4)
    
    e5 = Conv2D(512, (4,4), strides=(2,2), padding='same', 
                kernel_initializer=init)(e4)
    e5 = BatchNormalization()(e5, training=True)
    e5 = LeakyReLU(alpha=0.2)(e5)
    
    e6 = Conv2D(512, (4,4), strides=(2,2), padding='same', 
                kernel_initializer=init)(e5)
    e6 = BatchNormalization()(e6, training=True)
    e6 = LeakyReLU(alpha=0.2)(e6)
    
    e7 = Conv2D(512, (4,4), strides=(2,2), padding='same', 
                kernel_initializer=init)(e6)
    e7 = BatchNormalization()(e7, training=True)
    e7 = LeakyReLU(alpha=0.2)(e7)
    
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', 
               kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    
    # decoder block
    d1 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', 
                         kernel_initializer=init)(b)
    d1 = BatchNormalization()(d1, training=True)
    d1 = Dropout(0.5)(d1, training=True)
    d1 = Concatenate()([d1, e7])
    d1 = Activation('relu')(d1)
    
    d2 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', 
                         kernel_initializer=init)(d1)
    d2 = BatchNormalization()(d2, training=True)
    d2 = Dropout(0.5)(d2, training=True)
    d2 = Concatenate()([d2, e6])
    d2 = Activation('relu')(d2)
    
    d3 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', 
                         kernel_initializer=init)(d2)
    d3 = BatchNormalization()(d3, training=True)
    d3 = Dropout(0.5)(d3, training=True)
    d3 = Concatenate()([d3, e5])
    d3 = Activation('relu')(d3)
    
    d4 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', 
                         kernel_initializer=init)(d3)
    d4 = BatchNormalization()(d4, training=True)
    d4 = Concatenate()([d4, e4])
    d4 = Activation('relu')(d4)
    
    d5 = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', 
                         kernel_initializer=init)(d4)
    d5 = BatchNormalization()(d5, training=True)
    d5 = Concatenate()([d5, e3])
    d5 = Activation('relu')(d5)
    
    d6 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', 
                         kernel_initializer=init)(d5)
    d6 = BatchNormalization()(d6, training=True)
    d6 = Concatenate()([d6, e2])
    d6 = Activation('relu')(d6)
    
    d7 = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', 
                         kernel_initializer=init)(d6)
    d7 = BatchNormalization()(d7, training=True)
    d7 = Concatenate()([d7, e1])
    d7 = Activation('relu')(d7)
    
    
    g = Conv2DTranspose(input_shape[2], (4,4), strides=(2,2), padding='same', 
                        kernel_initializer=init)(d7) #Modified 
    out_image = Activation('tanh')(g)  #Generates images in the range -1 to 1. So change inputs also to -1 to 1
    # define model
    model = Model(input_img, out_image)
    return model

# building combined model
def build_pix2pix(generator, discriminator, input_shape, opt=Adam, lr=0.0002, beta1=0.5,
                   loss=['binary_crossentropy', 'mae'], loss_weights=[1,100]):
    """
    Build a combined Pix2Pix model that consists of a generator and discriminator 
    for image-to-image translation.

    Args:
        generator (keras.models.Model): The generator model.
        discriminator (keras.models.Model): The discriminator model.
        input_shape (tuple): Shape of the input images (height, width, channels).
        opt (Optimizer, optional): The optimizer class to use for compiling 
        the model. Default is Adam.
        lr (float, optional): Learning rate for the optimizer. 
        Default is 0.0002.
        beta1 (float, optional): Exponential decay rate for the first moment 
        estimates in Adam optimizer. Default is 0.5.
        loss (list, optional): List of loss functions to be used for training. 
        Default is ['binary_crossentropy', 'mae'].
        loss_weights (list, optional): List of loss weights. 
        Default is [1, 100].

    Returns:
        keras.models.Model: Combined Pix2Pix model.
    """
    
    for layer in discriminator.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
            
    # source image
    src_img = Input(shape=input_shape)
    
    # connect the source input to the generator input
    gen_out = generator(src_img)
    
    # connect the source input and generator output to the discriminator input
    dis_out = discriminator([src_img, gen_out])
    
    # definge the model
    model = Model(src_img, [dis_out, gen_out])
    
    opt = Adam(learning_rate=lr, beta_1=beta1)
    model.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    
    return model