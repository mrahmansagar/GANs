# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:01:07 2023

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany

Helper functions for cycleGAN model

"""
import os
import numpy as np
import matplotlib.pyplot as plt

def generate_fake_samples(gen_model, data, patchgan_output_size):
    """
    Generate fake samples using a generator model and create corresponding labels.
    
    Args:
        gen_model (keras.models.Model): The generator model used to produce fake/generated samples.
        data (numpy.ndarray): Input data used for generating fake samples.
        patchgan_output_size (int): Size of the PatchGAN output (e.g., 16x16).
        
    Returns:
        numpy.ndarray: Array of generated fake samples.
        numpy.ndarray: Array of labels for fake samples with shape (batch_size, patchgan_output_size, patchgan_output_size, 1).
    """
    #using the generator model to produce fake/generated samples 
    X_fake = gen_model.predict(data)
    
    #creating labels for fake samples of size pathgan_output_shape. labels for
    #fake/generated samples are 0
    y_fake = np.zeros(shape=(len(data), patchgan_output_size, patchgan_output_size, 1))
    
    return X_fake, y_fake


def update_generated_image_pool(maintained_pool, images, max_pool_size=50):
    """
    Update an image pool with newly generated images and select images for use based on a maximum pool size.

    Args:
        maintained_pool (list): List of previously generated images in the pool.
        images (list or numpy.ndarray): Newly generated images to update the pool.
        max_pool_size (int): Maximum size of the maintained image pool.

    Returns:
        numpy.ndarray: Array of selected images for use, considering the pool constraints.
    """
    # creating an empty list of selected image from pool to return after updating
    selected_image = []
    for image in images:
        #untill the max size of the pool stock all the images
        if len(maintained_pool) < 50:
            maintained_pool.append(image)
            selected_image.append(image)
        
        # when list is full with max pool size randomly decide if newly generated 
        # image to be used or use a randomly selected image from the pool and 
        # replace the used image in the pool with the newly generated image 
        elif np.random.random() < 0.5:
            # used the newly generated image but do not update the pool 
            selected_image.append(image)
        
        else:
            idx = np.random.randint(0, len(maintained_pool))
            selected_image.append(maintained_pool[idx])
            maintained_pool[idx] = image
        
    return np.asarray(selected_image)
        
            
# ToDo: optimize for 3d data    
def evaluate_model_performance(gen_model, data, iteration, name, sample_size=5):
    """
    Evaluate and visualize the performance of a generator model by generating and plotting images.

    Args:
        gen_model (keras.models.Model): The generator model to evaluate.
        data (numpy.ndarray): Input data used for generating images.
        iteration (int): The current iteration or step of training.
        name (str): Name used for saving the output files.
        sample_size (int, optional): Number of samples to evaluate and plot. 
        Default is 5.

    Returns:
        None
    """
    
    idx = np.random.randint(0, len(data), sample_size)
    X = data[idx]
    X_gen = gen_model.predict(X)
    
    # scaling the image to be ploted 
    X = (X + 1)/ 2.0
    X_gen = (X_gen + 1)/ 2.0
    
    
    # ploting the images 
    plt.figure(figsize=(sample_size*2, sample_size))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(sample_size):
        plt.subplot(2, sample_size, i+1)
        plt.axis('off')
        if X.shape[3] == 1:
            plt.imshow(X[i], cmap='gray')
        else:
            plt.imshow(X[i])
        
        plt.subplot(2, sample_size, sample_size+1+i)
        plt.axis('off')
        if X.shape[3] == 1:
            plt.imshow(X_gen[i], cmap='gray')
        else:
            plt.imshow(X_gen[i])
    
    plt_name = os.path.join(name, f'output_after_{iteration}.png')
    plt.savefig(plt_name)
    plt.close()
    model_name = os.path.join(name, f'model_after_{iteration}.h5')
    gen_model.save(model_name)
    
    