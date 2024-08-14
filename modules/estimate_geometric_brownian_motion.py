# -*- coding: utf-8 -*-
"""
Created on Sat April 23 2024

@author: Felipe Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np # type: ignore

from scipy.stats import norm # type: ignore

# Global options ----
warnings.filterwarnings("ignore")

# Estimation of probability density function for the restricted Brownian Motion (RBM) ----
def estimate_pdf_gbm(x, t, mu, sigma, x_threshold):
    """Estimation of probability density function of restricted geometric
    Brownian motion:

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Arbitrary vector of real values of the same size of t (time)
    t : float or numpy array dtype float
        Arbitrary scalar or vector of real values of the same size of x (space)
    mu : float
        Stochastic drift of geometric Brownian motion
    sigma : float
        Difussion coefficient of geometric Brownian motion
    x_threshold : float
        Threshold value for the support of the probability density function
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        probability density function of restricted geometric Brownian motion
    """
    z_x = (np.log(x) - mu * t) / sigma
    z_v = (np.log(x_threshold) - mu * t) / sigma
    normalization = sigma * norm.sf(x = z_v, loc = 0, scale = np.sqrt(t)) * x

    z = norm.pdf(x = z_x, loc = 0, scale = np.sqrt(t)) / normalization

    return z

# Estimation of stationary probability density function for restricted geometric Brownian Motion (GBM) ----
def estimate_stationary_pdf_gbm(x, x_threshold, lambda_):
    r"""Estimation of stationary probability density function of restricted
    geometric Brownian motion as:
    :math:`\frac{\lambda x_{th}^{\lambda}}{x^{\lambda+1}}`
    where $x\geq x_{th}>0$ and $\lambda=2\mu\sigma^{-2}-2>0$.

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Arbitrary vector of real values
    x_threshold : float
        Threshold value for the support of the probability density function
    lambda_ : float
        Scale parameter defined by the stochastic drift and difussion
        coefficient of geometric Brownian motion
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        stationary probability density function of restrcited geometric
        Brownian motion
    """
    z = lambda_ * np.power(x_threshold, lambda_) / np.power(x, lambda_ + 1)    
    return z

# Estimation of stationary cumulative density function for geometric Brownian Motion (GBM) ----
def estimate_stationary_cdf_gbm(x, x_threshold, lambda_):
    r"""Estimation of stationary cumulative density function of geometric
    Brownian motion as:
    :math:`1-\left(\frac{x_{th}}{x}\right)^{\lambda}`
    where $x\geq x_{th}$ and $\lambda=2\mu\sigma^{-2}-2>0$.

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Arbitrary vector of real values
    x_threshold : float
        Threshold value for the support of the cumulative density function
    lambda_ : float
        Scale parameter defined by the stochastic drift and difussion
        coefficient of geometric Brownian motion
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        stationary cumulative density function of geometric Brownian motion
    """
    z = 1 - np.power(x_threshold / x, lambda_)
    return z

# Estimation of Shannon entropy for the restricted geometric Brownian Motion (GBM) ----
def estimate_shannon_entropy_gbm(t, mu, sigma, amplitude_1, amplitude_2, phi, h):
    r"""Estimation of Shannon entropy of restricted geometric Brownian motion
    as:
    :math:`A\ln{\left(\frac{e\pi\sigma^{2}}{2}(t-\phi)\right)}+H_{0}`
    where $A$ is the amplitude, $H_{0}$ is a gauge condition of the initial
    differential entropy, $\phi$ is the temporal phase of entropy and
    $\sigma>0$. In all cases $mu\to0$.

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of geometric Brownian motion
    sigma : float
        Difussion coefficient of geometric Brownian motion
    amplitude_1 : float
        Amplitude to make the differential entropy an extensive quantity
    amplitude_2 : float
        Amplitude to make the differential entropy an extensive quantity for
        temporal linear term
    phi : float
        Temporal phase of entropy that takes into account temporal translations
    h : float
        Gauge condition of the initial differential entropy
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Shannon differential entropy of restricted geometric Brownian motion
    """
    # Local variables
    t_phase = t - phi
    factor_1 = np.sqrt(2 * np.pi * t_phase * sigma**2)
    factor_2 = (mu - 0.5 * sigma**2) * t_phase

    # Shannon entropy
    z = h + amplitude_1 * (0.5 + np.log(factor_1)) + amplitude_2 * factor_2
    return z

# Estimation of Renyi entropy for the restricted geometric Brownian Motion (GBM) ----
def estimate_renyi_entropy_gbm(t, mu, sigma, amplitude_1, amplitude_2, phi, h, p):
    r"""Estimation of Renyi entropy of restricted geometric Brownian motion as:
    :math:`A\ln{\left(\frac{\pi\sigma^{2}}{2}(t-\phi)\right)}+A\frac{\ln{(p)}}{p-1}+H_{0}`
    where $A$ is the amplitude, $H_{0}$ is a gauge condition of the initial
    differential entropy, $\phi$ is the temporal phase of entropy and
    $\sigma>0$. In all cases $mu\to0$.

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of geometric Brownian motion
    sigma : float
        Difussion coefficient of geometric Brownian motion
    x_threshold : float
        Threshold value for the support of the probability density function
    amplitude_1 : float
        Amplitude to make the differential entropy an extensive quantity
    amplitude_2 : float
        Amplitude to make the differential entropy an extensive quantity for
        temporal linear term
    phi : float
        Temporal phase of entropy that takes into account temporal translations
    h : float
        Gauge condition of the initial differential entropy
    p : float
        Exponent for estimation of Renyi entropy
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Renyi differential entropy of restricted geometric Brownian motion
    """
    # Local variables
    t_phase = t - phi
    factor_1 = np.sqrt(2 * np.pi * t_phase * sigma**2)
    factor_2 = (mu + 0.5 * (1 - 2 * p) * sigma**2 / p) * t_phase
    factor_3 = 0.5 * np.log(p) / (p-1)

    # Shannon entropy
    if p == 1:
        z = estimate_shannon_entropy_gbm(
            t = t,
            mu = mu,
            sigma = sigma,
            amplitude_1 = amplitude_1,
            amplitude_2 = amplitude_2,
            phi = phi,
            h = h
        )
    
    # Min-entropy (Shannon_entropy - A*ln(e))
    elif np.isinf(p) == True:
        z = h + amplitude_1 * factor_1
    
    # Renyi entropy (p > 0)
    else:        
        z = h + amplitude_1 * (factor_3 + np.log(factor_1)) + amplitude_2 * factor_2
    
    # Hartley or max-entropy (p=0) is not defined (ln(p=0)/(p-1) -> -infty)

    return z

# Estimation of Shannon entropy production rate for the restricted geometric Brownian motion (GBM) ----
def estimate_epr_gbm(t, mu, sigma, amplitude_1, amplitude_2, phi, h=0):
    r"""Estimation of Shannon entropy production rate of restricted geometric
    Brownian motion as:
    :math:`\frac{A\sigma}{t-\phi}+h_{0}`
    where $A$ is the amplitude, $h_{0}$ is a gauge condition of the initial 
    production rate of differential entropy, $\phi$ is the temporal phase of
    entropy and $\sigma>0$. In all cases $mu\to0$.

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of geometric Brownian motion
    sigma : float
        Difussion coefficient of geometric Brownian motion
    amplitude_1 : float
        Amplitude to make the differential entropy an extensive quantity
    amplitude_2 : float
        Amplitude to make the differential entropy an extensive quantity for
        temporal linear term
    phi : float
        Temporal phase of entropy that takes into account temporal translations
    h : float
        Gauge condition of the initial production rate of differential entropy
        (default value 0)
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Differential Shannon entropy production rate of restricted geometric
        Brownian motion
    """
    # Local variables
    t_phase = t - phi
    factor_1 = 0.5 / t_phase
    factor_2 = mu - 0.5 * sigma**2

    # Shannon entropy production rate
    z = h + amplitude_1 * factor_1 + amplitude_2 * factor_2
    return z

# Estimation of Renyi entropy production rate for the restricted geometric Brownian motion (GBM) ----
def estimate_renyi_epr_gbm(t, mu, sigma, amplitude_1, amplitude_2, phi, h, p):
    r"""Estimation of Renyi entropy production rate of restricted geometric
    Brownian motion as:
    :math:`\frac{A\sigma}{t-\phi}+h_{0}`
    where $A$ is the amplitude, $h_{0}$ is a gauge condition of the initial 
    production rate of differential entropy, $\phi$ is the temporal phase of
    entropy and $\sigma>0$. This is independent of the value of p in the Renyi
    entropy when $mu\to0$.

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of geometric Brownian motion
    sigma : float
        Difussion coefficient of geometric Brownian motion
    amplitude_1 : float
        Amplitude to make the differential entropy an extensive quantity
    amplitude_2 : float
        Amplitude to make the differential entropy an extensive quantity for
        temporal linear term
    phi : float
        Temporal phase of entropy that takes into account temporal translations
    h : float
        Gauge condition of the initial production rate of differential entropy
        (default value 0)
    p : float
        Exponent for estimation of Renyi entropy
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Differential Renyi entropy production rate of restricted geometric
        Brownian motion
    """
    # Local variables
    t_phase = t - phi
    factor_1 = 0.5 / t_phase
    factor_2 = mu + 0.5 * sigma**2 * (1 - 2 * p) / p

    # Shannon entropy production rate
    if p == 1:
        z = estimate_epr_gbm(
            t = t,
            mu = mu,
            sigma = sigma,
            amplitude_1 = amplitude_1,
            amplitude_2 = amplitude_2,
            phi = phi,
            h = h
        )
    
    # Min-entropy production rate (Shannon_entropy - A*ln(e))
    elif np.isinf(p) == True:
        z = h + amplitude_1 * factor_1
    
    # Renyi entropy production rate (p > 0)
    else:
        z = h + amplitude_1 * factor_1 + amplitude_2 * factor_2
    
    # Hartley or max-entropy (p=0) is not defined (ln(p=0)/(p-1) -> -infty)

    return z
