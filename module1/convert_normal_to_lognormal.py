import numpy as np
import math 
from scipy.ndimage.interpolation import shift # Used for shifting arrays, specifically for discount factors in cash flow calculations
from scipy.io import loadmat # Used to load fixed parameters from a .mat file (MATLAB format)

#
def convert_normal_to_lognormal(norm_mean, noram_std):
    """
    Converts the mean and standard deviation of a Normal distribution
    to the corresponding parameters (mean and standard deviation) of a Lognormal distribution.
    This is often used when modelling variables that cannot be negative (e.g., prices).

    Args:
        norm_mean (float): The mean of the Normal distribution.
        noram_std (float): The standard deviation of the Normal distribution.

    Returns:
        tuple: A tuple containing (log_mean, log_std), which are the mean and
               standard deviation of the corresponding Lognormal distribution.
    """
    # Calculate the mean of the underlying normal distribution for the lognormal
    log_mean = math.log(norm_mean**2 / math.sqrt(norm_mean**2 + noram_std**2))
    # Calculate the standard deviation of the underlying normal distribution for the lognormal
    # Corrected typo: 'meth.log' changed to 'math.log'
    log_std = math.sqrt(math.log((norm_mean**2 + noram_std**2) / norm_mean**2)) 
    return (log_mean, log_std)

#==========================================================================================================================================
