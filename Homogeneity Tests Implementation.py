# -*- coding: utf-8 -*-
"""
Implementation of homogeneity tests: Pettitt test and SNHT.

@author: Niloufar Beikahmadi
"""

import numpy as np
from scipy.stats import norm

def pettitt_test(x):
    """
    Pettitt's test for a single change point detection.
    
    Args:
        x (np.array): Time series of data.
        
    Returns:
        tuple: (change_point_index, p_value)
    """
    n = len(x)
    k = np.zeros(n)
    for t in range(1, n):
        s_t = np.sign(x[t] - x[:t])
        k[t] = k[t-1] + np.sum(s_t)
        
    k_abs = np.abs(k)
    k_max = np.max(k_abs)
    tau = np.argmax(k_abs)
    
    p_value = 2 * np.exp(-6 * k_max**2 / (n**3 + n**2))
    
    return tau, p_value

def snht_test(x, window_size):
    """
    Standard Normal Homogeneity Test (SNHT) applied over moving windows.
    
    Args:
        x (np.array): Time series of data.
        window_size (int): The size of the moving window.
        
    Returns:
        tuple: (max_test_statistic, change_point_index)
    """
    n = len(x)
    if n < window_size:
        return 0, -1

    max_t = 0
    break_point = -1

    for i in range(n - window_size + 1):
        window = x[i : i + window_size]
        mean_total = np.mean(window)
        
        for k in range(1, window_size - 1):
            z1 = (np.mean(window[:k]) - mean_total) / np.std(window)
            z2 = (np.mean(window[k:]) - mean_total) / np.std(window)
            
            t_k = k * z1**2 + (window_size - k) * z2**2
            
            if t_k > max_t:
                max_t = t_k
                break_point = i + k
                
    # Critical values for SNHT are dependent on sample size.
    # For simplicity here, we return the statistic. The significance can be checked
    # against tabulated values or approximations in the analysis script.
    # A common approximation for the 95% critical value is around 9-11 for large N.
    return max_t, break_point