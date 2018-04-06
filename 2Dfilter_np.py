import numpy as np
from scipy import signal

from rolling_window_np import rolling_window

def gaussian(L, window_size, sigma_s, **kwargs):
    """ Perform gaussian filter 
            Args:
                L (np.ndarray): intensity of an image of shape H x W
                window_size(diameter) (int): sliding window size
                sigma_s (int): gaussian filter parameter
            Returns:
                LB (np.ndarray): filtered image
            Todo:
                - add padding mode argument (constant/symmetric/reflec....)
                - add conv mode argument (same/valid)
    """
    # define a 2D Gaussian kernel array
    midpt = window_size//2
    ax = np.arange(-midpt, midpt+1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma_s**2))
    kernel = kernel / np.sum(kernel)

    # perform 2D convolution
    LB = signal.convolve2d(L, kernel, boundary='symm', mode='same')

    return LB

def bilateral(L, window_size, sigma_s, sigma_r):
    """ Perform bilateral filter 
            Args:
                L (np.ndarray): intensity of an image of shape H x W
                window size(diameter) (int): sliding window size
                sigma_s (int): sigma in spatial term
                sigma_r (float): sigma in range term
            Returns:
                LB (np.ndarray): filtered image
            Notes:
                1. depends on rolling_window_np.py
            Todo:
                - add padding mode argument (constant/symmetric/reflec....)
                - add conv mode argument (same/valid)
    """
    # spatial term can be computed in advance
    midpt = window_size//2
    ax = np.arange(-midpt, midpt+1.)
    xx, yy = np.meshgrid(ax, ax)
    spatial_term = np.exp(-(xx**2 + yy**2) / (2. * sigma_s**2))

    # padding
    Lpad = np.pad(L, (midpt,midpt), 'symmetric')

    # filtering
    LB = np.zeros(L.shape)
    Lpad_patches = rolling_window(Lpad, [window_size, window_size], [1,1])
    pH, pW = Lpad_patches.shape[:2]
    for pi in range(pH):
        for pj in range(pW):
            patch = Lpad_patches[pi, pj]
            patch_midpt = patch[window_size//2, window_size//2]        
            range_term = np.exp(-(patch-patch_midpt)**2 / (2. * sigma_r**2))
            coef = spatial_term * range_term
            LB[pi, pj] = np.sum(patch * coef) / np.sum(coef)

    return LB