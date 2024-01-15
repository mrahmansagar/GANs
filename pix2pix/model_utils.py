# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:26:23 2023

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany

Helper functions for pix2pix model

"""
import os
import numpy as np
import matplotlib.pyplot as plt

from .. import utils

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
    y_fake = np.zeros(shape=(len(data), *patchgan_output_size))
    
    return X_fake, y_fake




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
    
    
    if len(data.shape) == 4:
        # Use 2D image data for plotting
        
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
        #saving the model at the iteration 
        model_name = os.path.join(name, f'model_after_{iteration}.h5')
        gen_model.save(model_name)
        
        
    elif len(data.shape) == 5:
        # Use 3D volumetric data for plotting
        
        #folder where the results will be stored after defined iteration 
        result_folder = f'{name}_after_{iteration}'
        if os.path.exists(result_folder):
            print('saving to a existing folder')
        else:
            os.makedirs(result_folder)
        
        for i in range(sample_size):
            src_data = np.squeeze(X[i])
            gen_data = np.squeeze(X_gen[i])
            space_btwn = 10
            combined_array = np.zeros(shape=(src_data.shape[0], src_data.shape[1], 2*src_data.shape[2]+space_btwn), dtype=src_data.dtype)
            combined_array[:, :, 0:src_data.shape[1]] = src_data
            combined_array[:, :, src_data.shape[1]+space_btwn:] = gen_data
            
            volume_name = os.path.join(name, f'sample_{i}')
            utils.saveSlices(combined_array, volume_name)
        
        model_name = os.path.join(name, f'model_after_{iteration}.h5')
        gen_model.save(model_name)
        
    else:
        raise ValueError("Data shape length should be 4 (2D image) or 5 (3D volumetric data).")
    












































