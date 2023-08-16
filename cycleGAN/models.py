# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:51:56 2023

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany

"""

# importing necessary libraries 
import os 
import numpy as np


# importing tensorflow and keras
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate
from keras.optimizers import Adam 
from keras.initializers import RandomNormal

from .. import utils


# building the discriminator model 
def build_discriminator(input_shape, loss='mse', opt=Adam, lr=0.0002, beta1=0.5, loss_weights=[0.5]):
    """
    Build and compile a discriminator model for a Generative Adversarial Network (GAN).
    
    Args:
        input_shape (tuple): The shape of the input image data (height, width, channels).
        loss (str, optional): The loss function to be used for training. Default is 'mse' (Mean Squared Error).
        opt (Optimizer, optional): The optimizer class to use for compiling the model. Default is Adam.
        lr (float, optional): Learning rate for the optimizer. Default is 0.0002.
        beta1 (float, optional): Exponential decay rate for the first moment estimates in Adam optimizer. Default is 0.5.
        loss_weights (list, optional): List of loss weights. Default is [0.5].
        
    Returns:
        keras.models.Model: Compiled discriminator model.
    """
    # initialization of weights
    init_weights = RandomNormal(stddev=0.02)
    
    input_image = Input(shape=input_shape)
    cnv = Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=init_weights)(input_image)
    cnv = LeakyReLU(alpha=0.2)(cnv)
    
    cnv = Conv2D(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=init_weights)(cnv)
    cnv = utils.InstanceNormalization(axis=-1)(cnv)
    cnv = LeakyReLU(alpha=0.2)(cnv)
    
    cnv = Conv2D(filters=256, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=init_weights)(cnv)
    cnv = utils.InstanceNormalization(axis=-1)(cnv)
    cnv = LeakyReLU(alpha=0.2)(cnv)
    
    # added an extra layer which was not in the original paper.
    cnv = Conv2D(filters=512, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=init_weights)(cnv)
    cnv = utils.InstanceNormalization(axis=-1)(cnv)
    cnv = LeakyReLU(alpha=0.2)(cnv)
    
    cnv = Conv2D(filters=512, kernel_size=(4,4), padding='same', kernel_initializer=init_weights)(cnv)
    cnv = utils.InstanceNormalization(axis=-1)(cnv)
    cnv = LeakyReLU(alpha=0.2)(cnv)
    
    output_patch = Conv2D(filters=1, kernel_size=(4,4), padding='same')(cnv)
    
    dis_model = Model(input_image, output_patch)
    
    # compling the model with optimizers and loss
    dis_model.compile(optimizer=opt(learning_rate=lr, beta_1=beta1), loss=loss, loss_weights=loss_weights)
    
    return dis_model


# defining a resnet block that will be used in the generator model according to 
# the original paper

def resnet_conv_block(filters, input_layer):
    """
    Create a residual block using convolutional layers for a ResNet-like architecture.
    
    Args:
        filters (int): Number of filters in the convolutional layers.
        input_layer (keras.layers.Layer): Input layer to the block.
        
    Returns:
        keras.layers.Layer: Output layer of the residual block.
    """
    # initialization of weights
    init_weights = RandomNormal(stddev=0.02)
    
    # First convolutional layer
    res = Conv2D(filters=filters, kernel_size=(3,3), padding='same', kernel_initializer=init_weights)(input_layer)
    res = utils.InstanceNormalization(axis=-1)(res)
    res = Activation('relu')(res)
    
    # Second convolutional layer
    res = Conv2D(filters=filters, kernel_size=(3,3), padding='same', kernel_initializer=init_weights)(res)
    res = utils.InstanceNormalization(axis=-1)(res)
    
    # Concatenate with the input layer
    res = Concatenate()([res, input_layer])
    
    return res

# building generator model 
def build_generator(input_shape, sizeof_resnet_block=9):
    """
    Build and compile a generator model for an image-to-image translation task, using a U-Net architecture with ResNet blocks.

    Args:
        input_shape (tuple): The shape of the input image data (height, width, channels).
        sizeof_resnet_block (int, optional): Number of ResNet blocks to include. Default is 9.

    Returns:
        keras.models.Model: Generator model.
    """
    # initialization of weights
    init_weights = RandomNormal(stddev=0.02)
    
    input_image = Input(shape=input_shape)
    
    # encoder part of the UNet
    encode = Conv2D(filters=64, kernel_size=(7,7), padding='same', kernel_initializer=init_weights)(input_image)
    encode = utils.InstanceNormalization(axis=-1)(encode)
    encode = Activation('relu')(encode)
    
    encode = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer=init_weights)(encode)
    encode = utils.InstanceNormalization(axis=-1)(encode)
    encode = Activation('relu')(encode)
    
    encode = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer=init_weights)(encode)
    encode = utils.InstanceNormalization(axis=-1)(encode)
    encode = Activation('relu')(encode)
    
    # resnet blocks
    for i in range(sizeof_resnet_block):
        encode = resnet_conv_block(filters=256, input_layer=encode)
        
    # decoder part of the Unet/autoencoder
    decode = Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer=init_weights)(encode)
    decode = utils.InstanceNormalization(axis=-1)(decode)
    decode = Activation('relu')(decode)
    
    decode = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer=init_weights)(decode)
    decode = utils.InstanceNormalization(axis=-1)(decode)
    decode = Activation('relu')(decode)
    
    decode = Conv2DTranspose(filters=3, kernel_size=(3,3), padding='same', kernel_initializer=init_weights)(decode)
    decode = utils.InstanceNormalization(axis=-1)(decode)
    output_image = Activation('tanh')(decode)
    
    gen_model = Model(input_image, output_image)
    
    return gen_model
    
    
   
# building combined model using 2opasdjfopusiopdfl
def build_cycleGAN(gen1, dis, gen2, input_shape, loss=['mse', 'mae', 'mae', 'mae'], opt=Adam, lr=0.0002, beta1=0.5, loss_weights=[1, 5, 10, 10]):
    
    #set the parameters to be trainable for one generator model at a time. 
    # others are kept constant. discriminator model is updated while training 
    gen1.trainable = True
    #
    dis.trainable = False
    gen2.trainable = False 
    
    #loss-1 adversarial loss
    input_gen1 = Input(shape=input_shape)
    output_gen1 = gen1(input_gen1)
    output_dis = dis(output_gen1)
    
    #loss-2: identity loss
    input_identity = Input(shape=input_shape)
    output_identity = gen1(input_identity)
    
    #loss-3: cycle loss :forward
    output_forward = gen2(output_gen1)
    
    #loss-4: cycle loss :backward
    output_gen2 = gen2(input_identity)
    output_backward = gen1(output_gen2)
    
    # model with input and output connected
    cycleGAN = Model([input_gen1, input_identity], [output_dis, output_identity, output_forward, output_backward])
    
    cycleGAN.compile(optimizer=opt(learning_rate=lr, beta_1=beta1), loss=loss, loss_weights=loss_weights)
    
    return cycleGAN
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    