# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:28:00 2026

@author: Mmr Sagar
"""
import os 
import numpy as np
from tqdm import tqdm
from datetime import datetime

import tensorflow as tf
from keras import Model, layers
from keras.models import Sequential
from keras.optimizers import Adam

from keras.layers import Conv2D, PReLU,BatchNormalization, Flatten
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add

from keras.applications import VGG19

from tfgans import utils
from tfgans.SRGAN import model_utils as mu


def build_discriminator(
        input_shape: tuple,
        loss: str = "binary_crossentropy",
        optimizer=Adam,
        metrics: list = ["accuracy"] 
) -> Model:
    """Builds the discriminator model for a Super-Resolution Generative Adversarial Network (SRGAN).

    Args:
        input_shape (tuple): The shape of the input images to the discriminator, typically (height, width, channels).
        loss (str): The loss function to use for training the discriminator. Default is "binary_crossentropy".
        optimizer (keras.optimizers.Optimizer): The optimizer to use for compiling the model. Default is Adam.
        metrics (list): A list of metrics to track during training. Default is ["accuracy"].

    Returns:
        keras.models.Model: The compiled discriminator model.
    """
    input = Input(shape=input_shape)
    d1 = Conv2D(64, (3,3), strides = 1, padding="same")(input)
    d1 = LeakyReLU(alpha=0.2)(d1)

    d2 = Conv2D(64, (3,3), strides = 2, padding="same")(d1)
    d2 = BatchNormalization(momentum=0.8)(d2)
    d2 = LeakyReLU(alpha=0.2)(d2)

    d3 = Conv2D(128, (3,3), strides = 1, padding="same")(d2)
    d3 = BatchNormalization(momentum=0.8)(d3)
    d3 = LeakyReLU(alpha=0.2)(d3)

    d4 = Conv2D(128, (3,3), strides = 2, padding="same")(d3)
    d4 = BatchNormalization(momentum=0.8)(d4)
    d4 = LeakyReLU(alpha=0.2)(d4)

    d5 = Conv2D(256, (3,3), strides = 1, padding="same")(d4)
    d5 = BatchNormalization(momentum=0.8)(d5)
    d5 = LeakyReLU(alpha=0.2)(d5)

    d6 = Conv2D(256, (3,3), strides = 2, padding="same")(d5)
    d6 = BatchNormalization(momentum=0.8)(d6)
    d6 = LeakyReLU(alpha=0.2)(d6)

    d7 = Conv2D(512, (3,3), strides = 1, padding="same")(d6)
    d7 = BatchNormalization(momentum=0.8)(d7)
    d7 = LeakyReLU(alpha=0.2)(d7)

    d8 = Conv2D(512, (3,3), strides = 2, padding="same")(d7)
    d8 = BatchNormalization(momentum=0.8)(d8)
    d8 = LeakyReLU(alpha=0.2)(d8)

    d9 = Flatten()(d8)
    d9 = Dense(1024)(d9)
    d9 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d9)

    model = Model(inputs=input, outputs=validity)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def build_generator(
        input_shape: tuple, 
        nb_residual_blocks: int = 16
) -> Model:
    """Builds the generator model for a Super-Resolution Generative Adversarial Network (SRGAN).

    Args:
        input_shape (tuple): The shape of the input images to the generator, typically (height, width, channels).
        nb_residual_blocks (int, optional): The number of residual blocks to include in the
        generator architecture. Default is 16.
    Returns:
        keras.models.Model: The compiled generator model.
    """

    input = Input(shape=input_shape)
    c1 = Conv2D(64, (9,9), strides=1, padding="same")(input)
    c1 = PReLU(shared_axes=[1, 2])(c1)

    r = mu.residual_block(c1)
    for _ in range(nb_residual_blocks):
        r = mu.residual_block(r)

    c2 = Conv2D(64, (3,3), strides=1, padding="same")(r)
    c2 = BatchNormalization(momentum=0.5)(c2)
    c2 = add([c1, c2])

    u1 = mu.upscale_block(c2)
    u2 = mu.upscale_block(u1)

    output = Conv2D(3, (9,9), strides=1, padding="same")(u2)

    model = Model(inputs=input, outputs=output)

    return model


def custom_vgg(
        input_shape: tuple
    ) -> Model:
    """Builds a VGG19 model for feature extraction in the SRGAN architecture.

    Args:
        input_shape (tuple): The shape of the input images to the VGG model, typically (height, width, channels).

    Returns:
        keras.models.Model: The VGG19 model with pre-trained weights, configured for feature extraction.
    """
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
    vgg.trainable = False
    model = Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)
    return model



def build_srgan(
        gen_model: Model,
        disc_model: Model,
        custom_vgg_model: Model,
        low_res_input_shape: tuple,
        high_res_input_shape: tuple,
        losses: list = ["binary_crossentropy", "mse"],
        loss_weights: list = [1e-3, 1],
        optimizer=Adam
) -> Model:
    """Builds the combined SRGAN model by connecting the generator and discriminator models.

    Args:
        gen_model (keras.models.Model): The generator model for the SRGAN.
        disc_model (keras.models.Model): The discriminator model for the SRGAN.
        custom_vgg_model (keras.models.Model): The custom VGG model for feature extraction.
        low_res_input_shape (tuple): The shape of the low-resolution input images, typically (height, width, channels).
        high_res_input_shape (tuple): The shape of the high-resolution input images, typically (height, width, channels).
        losses (list, optional): A list of loss functions to use for training the combined model. Default is ["binary_crossentropy", "mse"].
        loss_weights (list, optional): A list of weights for the loss functions. Default is
        [1e-3, 1].
        optimizer (keras.optimizers.Optimizer, optional): The optimizer to use for compiling the model. Default is Adam.

    Returns:
        keras.models.Model: The compiled combined SRGAN model.
    """
    
    
    disc_model.trainable = False

    low_res_input = Input(shape=low_res_input_shape)
    high_res_input = Input(shape=high_res_input_shape)

    generated_high_res = gen_model(low_res_input)
    
    generated_features = custom_vgg_model(generated_high_res)

    validity = disc_model(generated_high_res)

    model = Model(inputs=[low_res_input, high_res_input], outputs=[validity, generated_features])
    model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)

    return model



def train_srgan(
        gen_model: Model,
        disc_model: Model,
        cgan_model: Model,
        custom_vgg_model: Model,
        low_res_data: np.ndarray,
        high_res_data: np.ndarray,
        batch_size: int = 1,
        epochs: int = 10,
        summary_interval: int = 1,
        name: str = "SRGAN"
) -> None:
    """Trains the SRGAN model using the provided generator, discriminator, and combined models.

    Args:
        gen_model (keras.models.Model): The generator model for the SRGAN.
        disc_model (keras.models.Model): The discriminator model for the SRGAN.
        cgan_model (keras.models.Model): The combined SRGAN model for training.
        custom_vgg_model (keras.models.Model): The custom VGG model for feature extraction.
        low_res_data (numpy.ndarray): The low-resolution input data for training.
        high_res_data (numpy.ndarray): The high-resolution target data for training.
        batch_size (int, optional): The number of samples per batch during training. Default is 16.
        epochs (int, optional): The number of epochs to train the model. Default is 10000.
        summary_interval (int, optional): The interval (in epochs) at which to evaluate and summarize the model's performance. Default is 100.
        name (str, optional): The name to use for saving model checkpoints and summaries. Default is "SRGAN".
        

    Returns:
        None
    """
    curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # creating a folder to save models and traing logs and outputs
    output_folder =f"{name}_{curr_time}"

    if os.path.exists(output_folder):
        print("Saving to an existing folder")
    else:
        os.makedirs(output_folder)

    log_fileName = os.path.join(output_folder, "training_log.txt")
    log_file = utils.training_log(fileName=log_fileName)

    log_file.write(f"batch_size: {batch_size}\n")
    log_file.write(f"epochs: {epochs}\n")
    log_file.write(f"summary_interval: {summary_interval}\n")
    log_file.write(f"losses: {cgan_model.loss}\n")
    log_file.write(f"loss_weights: {cgan_model.loss_weights}\n")
    log_file.write(f"optimizer: {cgan_model.optimizer}\n")
    log_file.write(f"date size = {len(low_res_data)}\n")
    log_file.write('\n')

    train_low_res_batches =[]
    train_high_res_batches = []

    for index in range(int(high_res_data.shape[0] / batch_size)):
        start_idx = index * batch_size
        end_idx = start_idx + batch_size
        train_low_res_batches.append(low_res_data[start_idx:end_idx])
        train_high_res_batches.append(high_res_data[start_idx:end_idx])

    start_time = datetime.now()

    for epoch in range(epochs):
        fake_labels = np.zeros((batch_size, 1))
        real_labels = np.ones((batch_size, 1))

        for ite in tqdm(range(len(train_low_res_batches)), total=len(train_low_res_batches)):
            low_res_imgs = train_low_res_batches[ite]
            high_res_imgs = train_high_res_batches[ite]

            generated_high_res_imgs = gen_model.predict_on_batch(low_res_imgs)
            disc_model.trainable = True
            disc_loss_real = disc_model.train_on_batch(high_res_imgs, real_labels)
            disc_loss_fake = disc_model.train_on_batch(generated_high_res_imgs, fake_labels)
            
            disc_model.trainable = False
            disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

            generated_features = custom_vgg_model.predict(high_res_imgs)

            gen_loss, _, _ = cgan_model.train_on_batch([low_res_imgs, high_res_imgs], [real_labels, generated_features])

            log_message = (f"Epoch> {epoch+1}/{epochs} > Ite > {ite+1} >  "
                f"Discriminator Loss: {disc_loss}, Generator Loss: {gen_loss}\n")
            
            log_file.write(log_message)
            print(log_message)

        if (epoch + 1) % summary_interval == 0:
            mu.evaluate_model_performance(gen_model, low_res_data, epoch+1, output_folder)
    
    end_time = datetime.now()
    # Calculate the time difference
    time_diff = end_time - start_time

    # Extract days, seconds, and microseconds
    days = time_diff.days
    seconds = time_diff.seconds
    microseconds = time_diff.microseconds

    # Convert seconds to hours, minutes, and seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Print the time taken for the training 
    training_time = f'Training Time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {microseconds} microseconds\n'
    log_file.write('\n')
    log_file.write(training_time)    
    log_file.close()

    log_file.close()    
            

