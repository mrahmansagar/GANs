# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:55:41 2023

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany
"""

# importing necessary libraries 
import os 
import numpy as np


from sklearn.utils import shuffle

# importing tensorflow and keras
import tensorflow as tf

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.layers import LeakyReLU, Dropout


from . import utils



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




def build_generator(latent_dim, init_dim, final_dim, n_channel=3):
    
    """
    Build a generator model for generating images using a deep neural network.

    Args:
        latent_dim (int): Size of the input random noise vector.
        n_channel (int, optional): Number of channels in the generated images.
        Defaults to 3.

    Returns:
        keras.models.Sequential: A Keras Sequential model representing the generator.
    """
    num_conv_block = utils.calculate_conv_block(init_dim, final_dim)
    
    if not num_conv_block == None: 
        
        
        model = Sequential()
        # foundation for 4x4 image
        n_nodes = 256 * init_dim * init_dim
        model.add(Dense(n_nodes, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((init_dim, init_dim, 256)))
        
        for i in range(num_conv_block):
            # upsample to init_dim x 2
            model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
            model.add(LeakyReLU(alpha=0.2))
        
        # output layer
        model.add(Conv2D(n_channel, (3,3), activation='tanh', padding='same'))
        
        return model
    else:
        print('Modify the function')
        
        

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


# trainig of gan model 
# Each batch is selected randomly from the entire dataset while training 
def train_gan(generator, discriminator, gan, data, latent_dim, epochs=100, batch_size=128, 
              summary_interval=10):
    """
    Train a Generative Adversarial Network (GAN). 
    Each batch is selected randomly from the entire dataset while training   

    This function trains a GAN by alternating between training the discriminator
    and the generator. It iterates through the specified number of epochs and
    updates the models using the given data.

    Args:
        generator (tf.keras.Model): The generator model.
        discriminator (tf.keras.Model): The discriminator model.
        gan (tf.keras.Model): The combined GAN model.
        data (numpy.ndarray): The training data.
        latent_dim (int): The dimension of the generator's input noise.
        epochs (int, optional): Number of training epochs. Default is 100.
        batch_size (int, optional): Batch size for training. Default is 128.
        summary_interval (int, optional): Interval for summarizing progress. Default is 10.

    Returns:
        None
    """
    
    batch_per_epoch = int(len(data) / batch_size)
    
    for epoch in range(epochs):

        for batch in range(batch_per_epoch):
            idx = np.random.randint(0, len(data), batch_size)
            X_real = data[idx]
            #generating class lebels. 1 for real images 
            y_real = np.ones(shape=(batch_size, 1))
            
            d_loss_real, _ = discriminator.train_on_batch(X_real, y_real)
            X_fake, y_fake = utils.generate_fake_samples(generator, latent_dim, batch_size)
            d_loss_fake, _ = discriminator.train_on_batch(X_fake, y_fake)
            
            X_gan = utils.generate_latent_points(latent_dim, batch_size)
            y_gan = np.ones(shape=(batch_size, 1))
            
            g_loss = gan.train_on_batch(X_gan, y_gan)
            # tracking the model train loss
            print(f'Epoch> {epoch+1}/{epochs} > Ite> {batch+1} '
                  f'dis loss real[{d_loss_real:.3f}] '
                  f'dis loss fake[{d_loss_fake:.3f}] '
                  f'gen loss[{g_loss:.3f}]')
            
        #save the model and generated output after defined intervals
        if (epoch+1) % (summary_interval) == 0:
            utils.evaluate_model_performance(generator, latent_dim, epoch, 'gen')
    

# trainig of gan model 
# After each epoch dataset is shuffled and batch are extracted so that the
# entire dataset is used for training
def train_gan2(generator, discriminator, gan, data, latent_dim, epochs=100, batch_size=128, 
              summary_interval=10):
    """
    Train a Generative Adversarial Network (GAN). 
    After each epoch dataset is shuffled and batch are extracted so that the
    entire dataset is used for training

    This function trains a GAN by alternating between training the discriminator
    and the generator. It iterates through the specified number of epochs and
    updates the models using the given data.

    Args:
        generator (tf.keras.Model): The generator model.
        discriminator (tf.keras.Model): The discriminator model.
        gan (tf.keras.Model): The combined GAN model.
        data (numpy.ndarray): The training data.
        latent_dim (int): The dimension of the generator's input noise.
        epochs (int, optional): Number of training epochs. Default is 100.
        batch_size (int, optional): Batch size for training. Default is 128.
        summary_interval (int, optional): Interval for summarizing progress. Default is 10.

    Returns:
        None
    """
    
    batch_per_epoch = int(len(data) / batch_size)
    
    for epoch in range(epochs):
        shuffled_data = shuffle(data)
        for batch in range(batch_per_epoch):
            idx = list(range(batch * batch_size, (batch + 1) * batch_size))
            X_real = shuffled_data[idx]
            #generating class lebels. 1 for real images 
            y_real = np.ones(shape=(batch_size, 1))
            
            d_loss_real, _ = discriminator.train_on_batch(X_real, y_real)
            X_fake, y_fake = utils.generate_fake_samples(generator, latent_dim, batch_size)
            d_loss_fake, _ = discriminator.train_on_batch(X_fake, y_fake)
            
            X_gan = utils.generate_latent_points(latent_dim, batch_size)
            y_gan = np.ones(shape=(batch_size, 1))
            
            g_loss = gan.train_on_batch(X_gan, y_gan)
            # tracking the model train loss
            print(f'Epoch> {epoch+1}/{epochs} > Ite> {batch+1} '
                  f'dis loss real[{d_loss_real:.3f}] '
                  f'dis loss fake[{d_loss_fake:.3f}] '
                  f'gen loss[{g_loss:.3f}]')
            
        #save the model and generated output after defined intervals
        if (epoch+1) % (summary_interval) == 0:
            utils.evaluate_model_performance(generator, latent_dim, epoch, 'gen')

