# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:15:27 2024

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

Modules to Quantitatively evaluate the GANs  
"""

import numpy as np

def relative_error(generated, real):
    """
    Calculate the relative error between two numpy arrays.

    The relative error is defined as the absolute difference between the sums
    of the two arrays divided by the sum of the 'real' array.

    Parameters:
    generated (numpy.ndarray): The generated array.
    real (numpy.ndarray): The real array to compare against.

    Returns:
    float: The relative error between the 'generated' and 'real' arrays.

    Raises:
    TypeError: If either input is not a numpy array or if the data types do not match.
    ValueError: If the dimensions of the input arrays do not match.
    """
    # Check if inputs are numpy arrays
    if not isinstance(generated, np.ndarray):
        raise TypeError("The 'generated' input must be a numpy array.")
    if not isinstance(real, np.ndarray):
        raise TypeError("The 'real' input must be a numpy array.")
    
    # Check if the dimensions of the inputs match
    if generated.shape != real.shape:
        raise ValueError("The 'generated' and 'real' inputs must have the same dimensions.")
    
    # Check if the data types of the inputs match
    if generated.dtype != real.dtype:
        raise TypeError("The 'generated' and 'real' inputs must have the same data type.")
    
    # Calculate the total pixel values
    total_pix_gen = np.sum(generated)
    total_pix_real = np.sum(real)
    
    # Calculate the relative error
    re = np.abs(total_pix_gen - total_pix_real) / total_pix_real
    
    return re




