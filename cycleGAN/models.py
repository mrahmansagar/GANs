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
from . import model_utils as mu

# building the discriminator model 
def build_discriminator(input_shape, opt=Adam, lr=0.0002, beta1=0.5, 
                        loss='mse', loss_weights=[0.5]):
    """
    Build and compile a discriminator model for a Generative Adversarial 
    Network (GAN).
    
    Args:
        input_shape (tuple): The shape of the input image data (height, width, channels).
        loss (str, optional): The loss function to be used for training. 
        Default is 'mse' (Mean Squared Error).
        opt (Optimizer, optional): The optimizer class to use for compiling the model. 
        Default is Adam.
        lr (float, optional): Learning rate for the optimizer. Default is 0.0002.
        beta1 (float, optional): Exponential decay rate for the first moment 
        estimates in Adam optimizer. Default is 0.5.
        loss_weights (list, optional): List of loss weights. Default is [0.5].
        
    Returns:
        keras.models.Model: Compiled discriminator model.
    """
    # initialization of weights
    init_weights = RandomNormal(stddev=0.02)
    
    input_image = Input(shape=input_shape)
    cnv = Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding='same', 
                 kernel_initializer=init_weights)(input_image)
    cnv = LeakyReLU(alpha=0.2)(cnv)
    
    cnv = Conv2D(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', 
                 kernel_initializer=init_weights)(cnv)
    cnv = utils.InstanceNormalization(axis=-1)(cnv)
    cnv = LeakyReLU(alpha=0.2)(cnv)
    
    cnv = Conv2D(filters=256, kernel_size=(4,4), strides=(2,2), padding='same', 
                 kernel_initializer=init_weights)(cnv)
    cnv = utils.InstanceNormalization(axis=-1)(cnv)
    cnv = LeakyReLU(alpha=0.2)(cnv)
    
    # added an extra layer which was not in the original paper.
    cnv = Conv2D(filters=512, kernel_size=(4,4), strides=(2,2), padding='same', 
                 kernel_initializer=init_weights)(cnv)
    cnv = utils.InstanceNormalization(axis=-1)(cnv)
    cnv = LeakyReLU(alpha=0.2)(cnv)
    
    cnv = Conv2D(filters=512, kernel_size=(4,4), padding='same', 
                 kernel_initializer=init_weights)(cnv)
    cnv = utils.InstanceNormalization(axis=-1)(cnv)
    cnv = LeakyReLU(alpha=0.2)(cnv)
    
    output_patch = Conv2D(filters=1, kernel_size=(4,4), padding='same')(cnv)
    
    dis_model = Model(input_image, output_patch)
    
    # compling the model with optimizers and loss
    dis_model.compile(optimizer=opt(learning_rate=lr, beta_1=beta1), 
                      loss=loss, loss_weights=loss_weights)
    
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
    res = Conv2D(filters=filters, kernel_size=(3,3), padding='same', 
                 kernel_initializer=init_weights)(input_layer)
    res = utils.InstanceNormalization(axis=-1)(res)
    res = Activation('relu')(res)
    
    # Second convolutional layer
    res = Conv2D(filters=filters, kernel_size=(3,3), padding='same', 
                 kernel_initializer=init_weights)(res)
    res = utils.InstanceNormalization(axis=-1)(res)
    
    # Concatenate with the input layer
    res = Concatenate()([res, input_layer])
    
    return res

# building generator model 
def build_generator(input_shape, sizeof_resnet_block=9):
    """
    Build and compile a generator model for an image-to-image translation task, 
    using a U-Net architecture with ResNet blocks.

    Args:
        input_shape (tuple): The shape of the input image data (height, width, channels).
        sizeof_resnet_block (int, optional): Number of ResNet blocks to include. 
        Default is 9.

    Returns:
        keras.models.Model: Generator model.
    """
    # initialization of weights
    init_weights = RandomNormal(stddev=0.02)
    
    input_image = Input(shape=input_shape)
    
    # encoder part of the UNet
    encode = Conv2D(filters=64, kernel_size=(7,7), padding='same', 
                    kernel_initializer=init_weights)(input_image)
    encode = utils.InstanceNormalization(axis=-1)(encode)
    encode = Activation('relu')(encode)
    
    encode = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', 
                    kernel_initializer=init_weights)(encode)
    encode = utils.InstanceNormalization(axis=-1)(encode)
    encode = Activation('relu')(encode)
    
    encode = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='same', 
                    kernel_initializer=init_weights)(encode)
    encode = utils.InstanceNormalization(axis=-1)(encode)
    encode = Activation('relu')(encode)
    
    # resnet blocks
    for i in range(sizeof_resnet_block):
        encode = resnet_conv_block(filters=256, input_layer=encode)
        
    # decoder part of the Unet/autoencoder
    decode = Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', 
                             kernel_initializer=init_weights)(encode)
    decode = utils.InstanceNormalization(axis=-1)(decode)
    decode = Activation('relu')(decode)
    
    decode = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), padding='same', 
                             kernel_initializer=init_weights)(decode)
    decode = utils.InstanceNormalization(axis=-1)(decode)
    decode = Activation('relu')(decode)
    
    decode = Conv2DTranspose(filters=input_shape[-1], kernel_size=(3,3), padding='same', 
                             kernel_initializer=init_weights)(decode)
    decode = utils.InstanceNormalization(axis=-1)(decode)
    output_image = Activation('tanh')(decode)
    
    gen_model = Model(input_image, output_image)
    
    return gen_model
    
    
   
# building combined model using 2opasdjfopusiopdfl
def build_cycleGAN(gen1, dis, gen2, input_shape, opt=Adam, lr=0.0002, beta1=0.5,
                   loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10]):
    """
    Build and compile a CycleGAN model for image-to-image translation with two 
    generators and one discriminator.

    Args:
        gen1 (keras.models.Model): The first generator model.
        dis (keras.models.Model): The discriminator model.
        gen2 (keras.models.Model): The second generator model.
        input_shape (tuple): The shape of the input image data (height, width, channels).
        loss (list, optional): List of loss functions to be used for training. 
        Default is ['mse', 'mae', 'mae', 'mae'].
        opt (Optimizer, optional): The optimizer class to use for compiling the model. 
        Default is Adam.
        lr (float, optional): Learning rate for the optimizer. Default is 0.0002.
        beta1 (float, optional): Exponential decay rate for the first moment 
        estimates in Adam optimizer. Default is 0.5.
        loss_weights (list, optional): List of loss weights. 
        Default is [1, 5, 10, 10].

    Returns:
        keras.models.Model: Compiled CycleGAN model.
    """
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
    cycleGAN = Model([input_gen1, input_identity], 
                     [output_dis, output_identity, output_forward, output_backward])
    
    cycleGAN.compile(optimizer=opt(learning_rate=lr, beta_1=beta1), 
                     loss=loss, loss_weights=loss_weights)
    
    return cycleGAN



def train_cycleGAN(disA, disB, genA2B, genB2A, cganA2B, cganB2A, dataA, dataB, 
                   batch_size=1, epochs=10, summary_interval=10, nameA2B='GenA2B', nameB2A='GenB2A'):
    """
    Train a CycleGAN model for image-to-image translation.

    Args:
        disA (keras.models.Model): Discriminator model for domain A.
        disB (keras.models.Model): Discriminator model for domain B.
        genA2B (keras.models.Model): Generator model from domain A to B.
        genB2A (keras.models.Model): Generator model from domain B to A.
        cganA2B (keras.models.Model): CycleGAN model for domain A to B translation.
        cganB2A (keras.models.Model): CycleGAN model for domain B to A translation.
        dataA (numpy.ndarray): Training data for domain A.
        dataB (numpy.ndarray): Training data for domain B.
        batch_size (int, optional): Batch size for training. Default is 1.
        epochs (int, optional): Number of training epochs. Default is 10.
        summary_interval (int, optional): Interval (in terms of epochs) for saving 
        output plots and models. Default is 10.
        nameA2B (str, optional): The name of the gen model A2B (default is 'GenA2B').
        Relative path supported.
        nameB2A (str, optional): The name of the gen model B2A (default is 'GenB2A'). 
        Relative path supported.

    Returns:
        None
    """
    
    # output patch shape of the patchGAN discriminator
    patch_size = disA.output_shape[1]
    
    img_poolA = []
    img_poolB = []
    
    batch_per_epoch = int(len(dataA) / batch_size)
    
    train_iterations = batch_per_epoch * epochs
    
    for step in range(train_iterations):
        # randomly selecting a real sample from both domain 
        idx = np.random.randint(0, dataA.shape[0], batch_size)
        X_realA = dataA[idx]
        X_realB = dataB[idx]
        # creating labels for them. lebel is 1 for real samples 
        y_realA = np.ones(shape=(batch_size, patch_size, patch_size, 1))
        y_realB = np.ones(shape=(batch_size, patch_size, patch_size, 1))
        
        # using the batch of real samples generate fake samples with generator models
        X_fakeA, y_fakeA = mu.generate_fake_samples(genB2A, X_realB, patch_size)
        X_fakeB, y_fakeB = mu.generate_fake_samples(genA2B, X_realA, patch_size)
        
        # maintain the pool with the generated images
        X_fakeA = mu.update_generated_image_pool(img_poolA, X_fakeA)
        X_fakeB = mu.update_generated_image_pool(img_poolB, X_fakeB)
        
        # train the generator model to generate from domain A to B
        genA2B_loss, _, _, _, _ = cganA2B.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
        # train the discriminator model for B 
        disB_loss1 = disB.train_on_batch(X_realB, y_realB)
        disB_loss2 = disB.train_on_batch(X_realB, y_fakeB)
        
        
        # train the generator model to generate from domain B to A
        genB2A_loss, _, _, _, _ = cganB2A.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        # train the discriminator model for A
        disA_loss1 = disA.train_on_batch(X_realA, y_realA)
        disA_loss2 = disA.train_on_batch(X_fakeA, y_fakeA)
        
        # tracking the model train loss
        print(f'Epoch> {int(step/batch_per_epoch) +1}/{epochs} > Ite> {step+1} '
              f'disA[{disA_loss1:.3f}, {disA_loss2:.3f}] '
              f'disB[{disB_loss1:.3f}, {disB_loss2:.3f}] '
              f'gen[{genA2B_loss:.3f}, {genB2A_loss:.3f}]')
        
        #save the model and generated output after defined intervals
        if (step+1) % (batch_per_epoch*summary_interval) == 0:
            mu.evaluate_model_performance(genA2B, dataA, step, name=nameA2B)
            mu.evaluate_model_performance(genB2A, dataB, step, name=nameB2A)
            
        
        
        


        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    