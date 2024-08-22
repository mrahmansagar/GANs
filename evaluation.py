# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:15:27 2024

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

Modules to Quantitatively evaluate the GANs  
"""

import numpy as np

def relative_error(generated, real, epsilon=1e-10):
    """ 
   Calculate Relative Error (RE) between generated and ground truth,
   while avoiding division by zero by masking zero ground truth values.
   Args:
   - predicted: numpy array of shape (256, 256, 256) representing the predicted volume.
   - ground_truth: numpy array of shape (256, 256, 256) representing the ground truth volume.
   - epsilon: Small constant to avoid division by zero (default is 1e-10).
   Returns:
   - RE: Relative Error for the 3D volume.

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
    
    
    generated = generated.astype(np.float64)
    real = real.astype(np.float64)
    
    # Create a mask where ground_truth is not zero
    nonzero_mask = np.abs(real) > epsilon
    
    # Calculate absolute difference and relative error only where ground_truth is non-zero
    abs_diff = np.abs(generated - real)
    rel_error = np.zeros_like(abs_diff)
    rel_error[nonzero_mask] = abs_diff[nonzero_mask] / np.abs(real[nonzero_mask])
    
    # Compute mean relative error over all voxels
    mean_rel_error = np.mean(rel_error)
    return mean_rel_error




from scipy.ndimage import gaussian_filter

def calculate_ssim(image1, image2):
    """
    Calculate the Structural Similarity Index (SSIM) between two images or volumes.

    This function computes the SSIM between two n-dimensional images or volumes 
    (e.g., 2D grayscale images or 3D volumetric data such as MRI scans). 
    It applies a Gaussian filter to compute local means, variances, and covariances,
    and then calculates SSIM based on these statistics.

    Parameters
    ----------
    image1 : ndarray
        The first input image or volume. Can be 2D or 3D.
        
    image2 : ndarray
        The second input image or volume. Must have the same shape as `image1`.

    Returns
    -------
    float
        The mean SSIM value between the two input images or volumes.

    Raises
    ------
    ValueError
        If the input images or volumes do not have the same dimensions.

    Notes
    -----
    - The function assumes the images or volumes have pixel values in the range [0, 255].
    - The Gaussian filter's standard deviation (sigma) is set to 1.5 by default, which
      corresponds to an effective window size of approximately 9x9 pixels in 2D 
      or 9x9x9 voxels in 3D.
    
    Example
    -------
    >>> image1 = np.random.rand(256, 256) * 255  # Example 2D image
    >>> image2 = np.random.rand(256, 256) * 255  # Example 2D image
    >>> ssim_value = calculate_ssim_nd(image1, image2)
    >>> print(ssim_value)
    """
    if image1.shape != image2.shape:
        raise ValueError('Input images or volumes must have the same dimensions.')

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    # Convert images to float64
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    
    # Apply Gaussian filter
    mu1 = gaussian_filter(image1, sigma=1.5)
    mu2 = gaussian_filter(image2, sigma=1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = gaussian_filter(image1 ** 2, sigma=1.5) - mu1_sq
    sigma2_sq = gaussian_filter(image2 ** 2, sigma=1.5) - mu2_sq
    sigma12 = gaussian_filter(image1 * image2, sigma=1.5) - mu1_mu2
    
    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()



