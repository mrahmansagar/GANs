# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:15:27 2024

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

Modules to Quantitatively evaluate the GANs  
"""

import numpy as np
import math
from scipy.ndimage import gaussian_filter

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



def ssim(image1, image2):
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





def psnr(img1, img2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two 2D or 3D grayscale images or volumes.

    PSNR is expressed in decibels (dB) and provides a quantitative measurement of the difference
    between the two images. A higher PSNR value indicates closer similarity to the reference image,
    while a lower PSNR value indicates more difference.

    Args:
    - img1: numpy ndarray of shape (H, W) or (H, W, D), representing the first image/volume (ground truth).
    - img2: numpy ndarray of the same shape as img1, representing the second image/volume (predicted).

    Returns:
    - PSNR: Peak Signal-to-Noise Ratio in decibels (dB). If the Mean Squared Error (MSE) is 0, 
      indicating identical images, the function returns infinity.

    Raises:
    - ValueError: If img1 and img2 have different shapes, data types, or if they are not numpy ndarrays.

    Example:
    ```
    psnr_value = calculate_psnr(img1, img2)
    print(f"PSNR: {psnr_value:.2f} dB")
    ```
    """
    
    # Check if inputs are numpy ndarrays
    if not isinstance(img1, np.ndarray) or not isinstance(img2, np.ndarray):
        raise ValueError("Both inputs must be numpy ndarrays.")
    
    # Check if inputs have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    # Check if inputs have the same data type
    if img1.dtype != img2.dtype:
        raise ValueError("Input images must have the same data type.")
    
    # Convert inputs to float64 for PSNR calculation
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)
    
    # If MSE is 0, return infinity (images are identical)
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr


