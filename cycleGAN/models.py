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
    
    
    