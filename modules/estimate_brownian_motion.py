# -*- coding: utf-8 -*-
"""
Created on Sat April 20 2024

@author: Felipe Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np # type: ignore

from scipy.stats import norm # type: ignore

# Global options ----
warnings.filterwarnings("ignore")

# Estimation of stationary probability density function for the restricted Brownian Motion (RBM) ----
def estimate_stationary_pdf_rbm(x, x_threshold, lambda_):
    r"""Estimation of stationary probability density function of restricted
    Brownian motion as:
    :math:`\lambda\exp{\left(-\lambda(x-x_{th})\right)}`
    where $x\geq x_{th}$ and $\lambda=2\mu\sigma^{-2}>0$.

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Arbitrary vector of real values
    x_threshold : float
        Threshold value for the support of the probability density function
    lambda_ : float
        Scale parameter defined by the stochastic drift and difussion
        coefficient of Brownian motion
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        stationary probability density function of restricted Brownian motion
    """
    z = lambda_ * np.exp(-lambda_ * (x - x_threshold))
    return z

# Estimation of Shannon entropy for the restricted Brownian Motion (RBM) ----
def estimate_shannon_entropy_rbm(t, mu, sigma, amplitude, phi, h):
    r"""Estimation of Shannon entropy of restricted Brownian motion as:
    :math:`A\ln{\left(\frac{e\pi\sigma^{2}}{2}(t-\phi)\right)}+H_{0}`
    where $A$ is the amplitude, $H_{0}$ is a gauge condition of the initial
    differential entropy, $\phi$ is the temporal phase of entropy and
    $\sigma>0$. In all cases $mu\to0$.

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of Brownian motion
    sigma : float
        Difussion coefficient of Brownian motion
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    phi : float
        Temporal phase of entropy that takes into account temporal translations
    h : float
        Gauge condition of the initial differential entropy
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Shannon differential entropy of restricted Brownian motion
    """
    # Local variables
    t_phase = t - phi
    factor_1 = np.sqrt(2 * np.pi * t_phase * sigma**2)
    factor_2 = mu * np.sqrt(t_phase) / sigma
    factor_3 = 0.5 * mu * t_phase * np.exp(-0.5 * factor_2**2) / norm.cdf(factor_2)

    # Shannon entropy
    z = h + amplitude * (0.5 + np.log(factor_1) + np.log(norm.cdf(factor_2)) - factor_3 / factor_1)
    return z

# Estimation of Renyi entropy for the restricted Brownian Motion (RBM) ----
def estimate_renyi_entropy_rbm(t, mu, sigma, amplitude, phi, h, p):
    r"""Estimation of Renyi entropy of restricted Brownian motion as:
    :math:`A\ln{\left(\frac{\pi\sigma^{2}}{2}(t-\phi)\right)}+A\frac{\ln{(p)}}{p-1}+H_{0}`
    where $A$ is the amplitude, $H_{0}$ is a gauge condition of the initial
    differential entropy, $\phi$ is the temporal phase of entropy and
    $\sigma>0$. In all cases $mu\to0$.

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of Brownian motion
    sigma : float
        Difussion coefficient of Brownian motion
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    phi : float
        Temporal phase of entropy that takes into account temporal translations
    h : float
        Gauge condition of the initial differential entropy
    p : float
        Exponent for estimation of Renyi entropy
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Renyi differential entropy of restricted Brownian motion
    """
    # Local variables
    t_phase = t - phi
    factor_0 = 0.5 * np.log(p) / (p - 1)
    factor_1 = np.sqrt(2 * np.pi * t_phase * sigma**2)
    factor_2 = mu * np.sqrt(t_phase) / sigma
    factor_3 = np.log(norm.cdf(factor_2 * np.sqrt(p))) / (1 - p)
    factor_4 = p * np.log(norm.cdf(factor_2)) / (1 - p)

    # Shannon entropy
    if p == 1:
        z = estimate_shannon_entropy_rbm(
            t = t,
            mu = mu,
            sigma = sigma,
            amplitude = amplitude,
            phi = phi,
            h = h
        )
    
    # Min-entropy (Shannon_entropy - A*ln(e))
    elif np.isinf(p) == True:
        z = h + amplitude * (np.log(factor_1) + np.log(norm.cdf(factor_2)))
    
    # Renyi entropy (p > 0)
    else:
        z = h + amplitude * (factor_0 + np.log(factor_1) + factor_3 - factor_4)
    
    # Hartley or max-entropy (p=0) is not defined (ln(p=0)/(p-1) -> -infty)

    return z

# Estimation of Shannon entropy production rate for the restricted Brownian Motion (RBM) ----
def estimate_epr_rbm(t, mu, sigma, amplitude, phi, h=0):
    r"""Estimation of Shannon entropy production rate of restricted Brownian motion
    as:
    :math:`\frac{A\sigma}{t-\phi}+h_{0}`
    where $A$ is the amplitude, $h_{0}$ is a gauge condition of the initial 
    production rate of differential entropy, $\phi$ is the temporal phase of
    entropy and $\sigma>0$. In all cases $mu\to0$.

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of Brownian motion
    sigma : float
        Difussion coefficient of Brownian motion
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    phi : float
        Temporal phase of entropy that takes into account temporal translations
    h : float
        Gauge condition of the initial production rate of differential entropy
        (default value 0)
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Differential Shannon entropy production rate of restricted Brownian motion
    """
    # Local variables
    t_phase = t - phi
    factor_1 = np.sqrt(2 * np.pi * t_phase * sigma**2)
    factor_2 = mu * np.sqrt(t_phase) / sigma
    factor_3 = 0.25 * mu * np.exp(-0.5 * factor_2**2) / (norm.cdf(factor_2) * factor_1)
    factor_4 = 1 + factor_2**2 + mu * t_phase * np.exp(-0.5 * factor_2**2) / (norm.cdf(factor_2) * factor_1)

    # Shannon entropy production rate
    z = h + amplitude * (0.5 / t_phase + factor_3 * factor_4)
    return z

# Estimation of Renyi entropy production rate for the restricted Brownian Motion (RBM) ----
def estimate_renyi_epr_rbm(t, mu, sigma, amplitude, phi, h, p):
    r"""Estimation of Renyi entropy production rate of restricted Brownian
    motion as:
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
        Stochastic drift of Brownian motion
    sigma : float
        Difussion coefficient of Brownian motion
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
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
        Differential Renyi entropy production rate of restricted Brownian motion
    """
    # Local variables
    t_phase = t - phi
    factor_1 = np.sqrt(2 * np.pi * t_phase * sigma**2)
    factor_2 = mu * np.sqrt(t_phase) / sigma
    factor_3 = 0.5 * mu * np.sqrt(p) / ((1 - p) * factor_1)
    factor_4 = np.exp(-0.5 * p * factor_2**2) / norm.cdf(factor_2 * np.sqrt(p))
    factor_5 = np.sqrt(p) * np.exp(-0.5 * factor_2**2) / norm.cdf(factor_2)

    # Shannon entropy production rate
    if p == 1:
        z = estimate_epr_rbm(
            t = t,
            mu = mu,
            sigma = sigma,
            amplitude = amplitude,
            phi = phi,
            h = h
        )
    
    # Min-entropy production rate (Shannon_entropy production rate - A*ln(e))
    elif np.isinf(p) == True:
        z = h + 0.5 * amplitude * (1 / t_phase + mu * np.exp(-0.5 * factor_2**2) / (norm.cdf(factor_2) * factor_1))
    
    # Renyi entropy production rate (p > 0)
    else:
        z = h + amplitude * (0.5 / t_phase + factor_3 * (factor_4 - factor_5))
    
    # Hartley or max-entropy (p=0) is not defined (ln(p=0)/(p-1) -> -infty)

    return z
