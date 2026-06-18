import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Layer, Conv2D, PReLU, BatchNormalization 
from keras.layers import add
from keras.layers import UpSampling2D



def residual_block(
        input_layer: Layer, 
        nbfilters: int = 64, 
        filer_size: tuple = (3,3), 
        padding: str = "same",
        batch_norm_momentum: float = 0.5
) -> Layer:
    """Defines a residual block for a neural network, which consists of 
    two convolutional layers with batch normalization and a skip connection.

    Args:
        input_layer (keras.layers.Layer): The input layer to the residual block.
        nbfilters (int, optional): The number of filters for the convolutional layers. Default is 64.
        filer_size (tuple, optional): The size of the convolutional filters. Default is (3, 3).
        padding (str, optional): The type of padding to use in the convolutional layers. Default is "same".
        batch_norm_momentum (float, optional): The momentum for the batch normalization layers. Default is 0.5. 
    Returns:
        keras.layers.Layer: The output layer of the residual block.
    """ 
    # first convolutional layer with batch normalization and PReLU activation
    x = Conv2D(filters=nbfilters, kernel_size=filer_size, padding=padding)(input_layer)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    
    # second convolutional layer with batch normalization
    x = Conv2D(filters=nbfilters, kernel_size=filer_size, padding=padding)(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    
    # adding the input layer to the output of the second convolutional layer (skip connection)
    x = add([input_layer, x])
    
    return x

def upscale_block(
        input_layer: Layer, 
        upscale_factor: int = 2,
        nbfilters: int = 256, 
        filer_size: tuple = (3,3),
        padding: str = "same" 
    ) -> Layer:
    """Defines an upscale block for a neural network, which consists of an upsampling layer followed by a convolutional layer with batch normalization and PReLU activation.

    Args:
        input_layer (keras.layers.Layer): The input layer to the upscale block.
        upscale_factor (int, optional): The factor by which to upscale the input layer. Default is 2.
        nbfilters (int, optional): The number of filters for the convolutional layer. Default is 256.
        filer_size (tuple, optional): The size of the convolutional filters. Default is (3, 3).
        padding (str, optional): The type of padding to use in the convolutional layer. Default is "same".

    Returns:
        keras.layers.Layer: The output layer of the upscale block.
    """
    # convolutional layer with batch normalization and PReLU activation
    x = Conv2D(filters=nbfilters, kernel_size=filer_size, padding=padding)(input_layer)
    # upsampling the input layer by a factor of 2
    x = UpSampling2D(size=upscale_factor)(input_layer)
    x = PReLU(shared_axes=[1, 2])(x)
    
    return x

def evaluate_model_performance(
        gen_model: tf.keras.models.Model,
        data: np.ndarray,
        iteration: int,
        name: str, 
        sample_size: int = 5
    ) -> None:
    """
    Evaluate and visualize the performance of a generator model by generating and plotting images.

    Args:
        gen_model (keras.models.Model): The generator model to evaluate.
        data (numpy.ndarray): Input data used for generating images.
        iteration (int): The current iteration or step of training.
        name (str): Name used for saving the output files.
        sample_size (int, optional): Number of samples to evaluate and plot. 
        Default is 5.
        """
    
    idx = np.random.randint(0, len(data), sample_size)
    X = data[idx]
    X_gen = gen_model.predict(X)

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
        if X_gen.shape[3] == 1:
            plt.imshow(X_gen[i], cmap='gray')
        else:
            plt.imshow(X_gen[i])
    
    plt_name = os.path.join(name, f'output_after_{iteration}.png')
    plt.savefig(plt_name)
    plt.close()
    #saving the model at the iteration
    model_name = os.path.join(name, f'model_after_{iteration}.h5')
    gen_model.save(model_name)  

