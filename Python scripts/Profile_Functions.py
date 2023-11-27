import numpy as np
order = 0.5
from scipy.special import iv

def New_maxwell(vel, v0, sigma, k):
    vel = vel/sigma
    Lambda = v0**2/sigma**2
    mod_bessel = iv(order, np.sqrt(Lambda*vel**2))
    non_central_chi_sq =  (0.5 * np.exp( -(vel**2 + Lambda)/2 ) * (vel**2/Lambda)**(3/4 - 1/2) ) * mod_bessel * 2 * vel
    return k * non_central_chi_sq

def Maxwell(v, sigma, k):
    return k * v**2 * np.exp(-v**2/sigma**2) / sigma**(3/2)

def double_maxwell(v, sigma, k, v01, sigma1, k1):
    return Maxwell(v, sigma, k) + New_maxwell(v, v01, sigma1, k1)

def Triple_maxwell(v, sigma, k, v01, sigma1, k1, sigma2, k2):
    return Maxwell(v, sigma, k) + New_maxwell(v, v01, sigma1, k1) + Maxwell(v, sigma2, k2)

def triple_maxwell(v, sigma, k, sigma1, k1, sigma2, k2):
    return Maxwell(v, sigma, k) + Maxwell(v, sigma1, k1) + Maxwell(v, sigma2, k2)