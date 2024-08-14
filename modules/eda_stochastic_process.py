# -*- coding: utf-8 -*-
"""
Created on Sat April 22 2024

@author: Felipe Abril Bermúdez
"""

# Libraries ----
import re
import warnings
import numpy as np # type: ignore
import pandas as pd # type: ignore
import misc_functions as mf
import matplotlib.pyplot as plt # type: ignore
import matplotlib.ticker as mtick # type: ignore
import eda_misc_functions as eda_mf
import estimate_brownian_motion as ebm
import estimate_stochastic_process as esp
import estimate_geometric_brownian_motion as egbm

from scipy.optimize import curve_fit # type: ignore
from matplotlib import rcParams # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Get theoretical histogram data for Brownian motion paths ----
def get_histogram_brownian_motion(
    x,
    t,
    mu,
    sigma,
    x0,
    t0,
    x_threshold,
    hist_rbm,
    geometric_flag=False,
    alpha=0.01,
    beta=-4
):
    """Get theoretical histogram data (probability density function) from
    multiple simulations made for a Brownian motion for a specific time t

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Arbitrary vector of real values of the same size of t (time)
    t : float or numpy array dtype float
        Arbitrary scalar or vector of real values of the same size of x (space)
    mu : float
        Stochastic drift of Brownian motion
    sigma : float
        Difussion coefficient of Brownian motion
    x0 : float
        Initial condition of Brownian motion path
    t0 : float
        Initial time for Brownian motion path
    x_threshold : float
        Threshold value for the support of the probability density function
    hist_rbm : float or numpy array dtype float
        The values of the histogram of simulated data of restricted Brownian
        motion
    geometric_flag : bool
        Flag for the stochastic integration of Brownian motion sample as
        geometric Brownian motion
    alpha : float
        Level of statistical significance at the limits used for curve fitting
        (default value 0.01)
    beta : int
        Exponent for statistical significance at the limits used for curve
        fitting when values are close to zero (default value -4)
    
    Returns:
    ---------------------------------------------------------------------------
    hist_transient : float or numpy array dtype float
        The values of the histogram (probability density function) for
        transient state of restricted Brownian motion
    params_transient : numpy array dtype float
        Optimal values for the parameters t, mu, sigma with bounds
    """

    # Theoretical histogram data (probability density function)
    hist_transient = esp.estimate_pdf_rbm(
        x = x,
        t = t,
        mu = mu,
        sigma = sigma,
        x0 = x0,
        t0 = t0,
        x_threshold = x_threshold,
        geometric_flag = geometric_flag
    )
    
    # Curve fitting - Bounds
    t_lower, t_upper = eda_mf.get_bounds(z = t, alpha = alpha, beta = beta)
    mu_lower, mu_upper = eda_mf.get_bounds(z = mu, alpha = alpha, beta = beta)
    sigma_lower, sigma_upper = eda_mf.get_bounds(z = sigma, alpha = alpha, beta = beta)
    x0_lower, x0_upper = eda_mf.get_bounds(z = x0, alpha = alpha, beta = beta)
    t0_lower, t0_upper = eda_mf.get_bounds(z = t0, alpha = alpha, beta = beta)
    x_threshold_lower, x_threshold_upper = eda_mf.get_bounds(z = x_threshold, alpha = alpha, beta = beta)

    bounds_transient = (
        [t_lower, mu_lower, sigma_lower, x0_lower, t0_lower, x_threshold_lower],
        [t_upper, mu_upper, sigma_upper, x0_upper, t0_upper, x_threshold_upper]
    )
    
    # Curve fitting - Adjustment parameters
    def estimate_pdf_temp(x, t, mu, sigma, x0, t0, x_threshold):
        return esp.estimate_pdf_rbm(x, t, mu, sigma, x0, t0, x_threshold, geometric_flag)
    
    popt_transient, pcov_transient = curve_fit(
        estimate_pdf_temp,
        x,
        hist_rbm,
        p0 = [t, mu, sigma, x0, t0, x_threshold],
        bounds = bounds_transient
    )

    # Curve fitting - Adjustment parameters with bounds
    popt_transient, lower_transient, upper_transient = eda_mf.get_params_error(
        popt = popt_transient,
        pcov = pcov_transient
    )
    params_transient = [popt_transient, lower_transient, upper_transient]

    return hist_transient, params_transient

# Get theoretical entropy and entropy production rate data for the restricted Brownian Motion (RBM) ----
def get_entropy_rbm(
    t_rbm,
    entropy_rbm,
    t_midpoint_rbm,
    epr_rbm,
    mu_guess,
    sigma_guess,
    amplitude_1_guess,
    amplitude_2_guess,
    phi_guess,
    entropy_gauge_guess,
    epr_gauge_guess,
    geometric_flag=False
):
    """Get theoretical entropy and entropy production rate from multiple
    simulations made for a restricted Brownian motion

    Args:
    ---------------------------------------------------------------------------
    t_rbm : float or numpy array dtype float
        The values of times used to estimated the entropy of simulated data of
        restricted Brownian motion
    entropy_rbm : float or numpy array dtype float
        The values of the entropy of simulated data of restricted Brownian
        motion
    t_midpoint_rbm : float or numpy array dtype float
        The values of times used to estimated the entropy production rate of
        simulated data of restricted Brownian motion
    epr_rbm : float or numpy array dtype float
        The values of the entropy production rate of simulated data of
        restricted Brownian motion
    mu_guess : float
        Initial guess for the drift of Brownian motion
    sigma_guess : float
        Initial guess for the diffusion coefficient of Brownian motion
    amplitude_1_guess : float
        Initial guess for the amplitude of entropy of Brownian motion
    amplitude_2_guess : float
        Initial guess for the amplitude of temporal linear term entropy of
        geometric Brownian motion
    phi_guess : float
        Initial guess for the temporal phase of entropy of Brownian motion
    entropy_gauge_guess : float
        Initial guess for the initial entropy of the restricted Brownian motion
        (gauge value of entropy)
    epr_gauge_guess : float
        Initial guess for the initial entropy production rate of the Brownian
        motion (gauge value of entropy production rate)
    geometric_flag : bool
        Flag for the stochastic integration of Brownian motion sample as
        geometric Brownian motion
    
    Returns:
    ---------------------------------------------------------------------------
    params_entropy : numpy array dtype float
        Optimal values for the entropy parameters sigma, amplitude, phi, h
        (gauge value of entropy) with bounds
    params_epr : numpy array dtype float
        Optimal values for the entropy production rate (epr) parameters sigma,
        amplitude, phi, h (gauge value of epr) with bounds
    """

    # Curve fitting - Adjustment parameters
    if geometric_flag == True:
        # Curve fitting - Adjustment parameters
        popt_entropy, pcov_entropy = curve_fit(
            egbm.estimate_shannon_entropy_gbm,
            t_rbm,
            entropy_rbm,
            p0 = [mu_guess, sigma_guess, amplitude_1_guess, amplitude_2_guess, phi_guess, entropy_gauge_guess]
        )
        popt_epr, pcov_epr = curve_fit(
            egbm.estimate_epr_gbm,
            t_midpoint_rbm,
            epr_rbm,
            p0 = [mu_guess, sigma_guess, 0, amplitude_2_guess, phi_guess, epr_gauge_guess]
        )
    else:
        popt_entropy, pcov_entropy = curve_fit(
            ebm.estimate_shannon_entropy_rbm,
            t_rbm,
            entropy_rbm,
            p0 = [mu_guess, sigma_guess, amplitude_1_guess, phi_guess, entropy_gauge_guess]
        )
        popt_epr, pcov_epr = curve_fit(
            ebm.estimate_epr_rbm,
            t_midpoint_rbm,
            epr_rbm,
            p0 = [mu_guess, sigma_guess, amplitude_1_guess, phi_guess, epr_gauge_guess]
        )

    # Curve fitting - Adjustment parameters with bounds
    popt_entropy, lower_entropy, upper_entropy = eda_mf.get_params_error(
        popt = popt_entropy,
        pcov = pcov_entropy
    )
    popt_epr, lower_epr, upper_epr = eda_mf.get_params_error(
        popt = popt_epr,
        pcov = pcov_epr
    )

    params_entropy = [popt_entropy, lower_entropy, upper_entropy]
    params_epr = [popt_epr, lower_epr, upper_epr]

    return params_entropy, params_epr

# Plot final Exploratory Data Analysis for restricted Brownian motion (RBM) paths ----
def plot_brownian_motion(
    df_bm,
    mu,
    sigma,
    x0,
    t0,
    threshold,
    n_steps,
    geometric_flag=False,
    bins=10,
    density=True,
    alpha=0.01,
    beta=-4,
    p=1,
    ma_window=10,
    p_norm=1,
    significant_figures=2,
    width=12,
    height=28,
    fontsize_labels=13.5,
    fontsize_legend=11.5,
    n_cols=4,
    n_x_breaks=10,
    n_y_breaks=10,
    fancy_legend=False,
    usetex=False,
    dpi=200,
    save_figures=True,
    output_path="../output_files",
    information_name="",
    input_generation_date="2024-04-22"
):
    """Plot exploratory data analysis from multiple simulations made for a
    restricted Brownian motion

    Args:
    ---------------------------------------------------------------------------
    df_bm : pandas DataFrame
        DataFrame with the data of the multiple simulations made for a Brownian
        motion with three columns namely:
            simulation: the number assigned to the simulation 
            time: computational time of the simulation
            value: value of the Brownian motion
    mu : float
        Stochastic drift of Brownian motion
    sigma : float
        Difussion coefficient of Brownian motion
    x0 : float
        Initial condition of Brownian motion path
    t0 : float
        Initial time for Brownian motion path
    x_threshold : float
        Threshold value for the support of the probability density function
    n_steps : int
        Number of temporal steps used in the simulation of Brownian motion path
    geometric_flag : bool
        Flag for the stochastic integration of Brownian motion sample as
        geometric Brownian motion
    bins : int
        Number of numerical ranges in which the histogram data is aggregated
        (default value 10)
    density : bool
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the probability density function
        at the bin, normalized such that the integral over the range is 1
        (default value True)
    alpha : float
        Level of statistical significance (default value 0.01)
    beta : int
        Exponent for statistical significance when values are close to zero
        (default value -4)
    p : float
        Exponent for estimation of Renyi entropy (default value 1 for Shannon
        entropy)
    ma_window : int
        Moving average window for smoothing the entropy production rate
        (default value 10)
    p_norm : float
        p norm used in the estimation on mean absolute error MAE_p (default
        value 1)
    significant_figures : int
        Number of significant figures in labels (default value 2)
    width : int
        Width of final plot (default value 12)
    height : int
        Height of final plot (default value 28)
    fontsize_labels : float
        Font size in axis labels (default value 13.5)
    fontsize_legend : float
        Font size in legend (default value 11.5)
    n_cols : int
        Number of columns in legend (default value 4)
    n_x_breaks : int
        Number of divisions in x-axis (default value 10)
    n_y_breaks : int
        Number of divisions in y-axis (default value 10)
    fancy_legend : bool
        Fancy legend output (default value False)
    usetex : bool
        Use LaTeX for renderized plots (default value False)
    dpi : int
        Dot per inch for output plot (default value 200)
    save_figures : bool
        Save figures flag (default value True)
    output_path : string
        Local path for outputs (default value is "../output_files")
    information_name : string
        Name of the output plot (default value "")
    input_generation_date : string
        Date of generation (control version) (default value "2024-04-22")
        
    Returns:
    ---------------------------------------------------------------------------
    df_final : pandas DataFrame
        DataFrame with the resume of parameters after multiple curve fitting
        with eight columns namely:
            fitting: Name of the information fitted
            params_name: Names of parameters
            params_value: Optimal values for the curve fitting
            params_lower: Lower bound for optimal values for the curve fitting
            params_upper: Upper bound for optimal values for the curve fitting
            r_squared: Coefficient of determination for the curve fitting
            p_norm: p norm used in the estimation on mean absolute error MAE_p
            mae_p: Mean absolute error MAE_p for the curve fitting
    """

    # Local parameters
    t_min = df_bm["time"].min()
    t_max = df_bm["time"].max()
    dt = (t_max - t_min) / (n_steps - 1)

    # Histogram data of simulated data
    bins_rbm, bins_midpoint_rbm, hist_rbm = eda_mf.get_histogram_stochastic_process(
        df_sp = df_bm,
        t = t_max,
        bins = bins,
        density = density
    )
    del(bins_rbm)

    # Histogram theoretical data
    hist_transient, params_transient = get_histogram_brownian_motion(
        x = bins_midpoint_rbm,
        t = t_max,
        mu = mu,
        sigma = sigma,
        x0 = x0,
        t0 = t0,
        x_threshold = threshold,
        hist_rbm = hist_rbm,
        geometric_flag = geometric_flag,
        alpha = alpha,
        beta = beta
    )

    # Diffusion coefficient data
    mean_rbm, variance_rbm, params_diffusion = eda_mf.get_diffusion_law_stochastic_process(df_sp = df_bm)

    # Entropy and entropy production rate data
    t_rbm, entropy_rbm, t_midpoint_rbm, epr_rbm, epr_smooth_rbm = eda_mf.get_entropy_stochastic_process(
        df_sp = df_bm,
        dt = dt,
        p = p,
        ma_window = ma_window
    )

    epr_smooth_rbm_time = epr_smooth_rbm[0]
    epr_smooth_rbm_mean = epr_smooth_rbm[1]
    epr_smooth_rbm_lower = epr_smooth_rbm[2]
    epr_smooth_rbm_upper = epr_smooth_rbm[3]

    # Entropy and entropy production rate theoretical data
    amplitude_1_guess = 1
    amplitude_2_guess = 1
    entropy_gauge_guess = np.min(entropy_rbm)
    epr_gauge_guess = np.min(epr_rbm)
    
    if geometric_flag == True:
        amplitude_1_guess = -1 * 10**-4
        amplitude_2_guess = -1 * 10**-1
        entropy_gauge_guess = np.max(entropy_rbm)
        epr_gauge_guess = np.mean(epr_rbm)
    
    params_entropy, params_epr = get_entropy_rbm(
        t_rbm = t_rbm,
        entropy_rbm = entropy_rbm,
        t_midpoint_rbm = t_midpoint_rbm,
        epr_rbm = epr_rbm,
        mu_guess = mu,
        sigma_guess = sigma,
        amplitude_1_guess = amplitude_1_guess,
        amplitude_2_guess = amplitude_2_guess,
        phi_guess = 0,
        entropy_gauge_guess = entropy_gauge_guess,
        epr_gauge_guess = epr_gauge_guess,
        geometric_flag = geometric_flag
    )

    # Theoretical data fitting from parameters and respective errors
    def estimate_pdf_temp(x, t, mu, sigma, x0, t0, x_threshold):
        return esp.estimate_pdf_rbm(x, t, mu, sigma, x0, t0, x_threshold, geometric_flag)
        
    transient_prome = estimate_pdf_temp(bins_midpoint_rbm, *params_transient[0])    
    transient_lower = estimate_pdf_temp(bins_midpoint_rbm, *params_transient[1])
    transient_upper = estimate_pdf_temp(bins_midpoint_rbm, *params_transient[2])

    diffusion_prome = mf.temporal_fluctuation_scaling(mean_rbm, *params_diffusion[0])
    diffusion_lower = mf.temporal_fluctuation_scaling(mean_rbm, *params_diffusion[1])
    diffusion_upper = mf.temporal_fluctuation_scaling(mean_rbm, *params_diffusion[2])

    if geometric_flag == True:
        entropy_prome = egbm.estimate_shannon_entropy_gbm(t_rbm, *params_entropy[0])
        entropy_lower = egbm.estimate_shannon_entropy_gbm(t_rbm, *params_entropy[1])
        entropy_upper = egbm.estimate_shannon_entropy_gbm(t_rbm, *params_entropy[2])

        epr_prome = egbm.estimate_epr_gbm(t_midpoint_rbm, *params_epr[0])
        epr_lower = egbm.estimate_epr_gbm(t_midpoint_rbm, *params_epr[1])
        epr_upper = egbm.estimate_epr_gbm(t_midpoint_rbm, *params_epr[2])
    else:
        entropy_prome = ebm.estimate_shannon_entropy_rbm(t_rbm, *params_entropy[0])
        entropy_lower = ebm.estimate_shannon_entropy_rbm(t_rbm, *params_entropy[1])
        entropy_upper = ebm.estimate_shannon_entropy_rbm(t_rbm, *params_entropy[2])

        epr_prome = ebm.estimate_epr_rbm(t_midpoint_rbm, *params_epr[0])
        epr_lower = ebm.estimate_epr_rbm(t_midpoint_rbm, *params_epr[1])
        epr_upper = ebm.estimate_epr_rbm(t_midpoint_rbm, *params_epr[2])

    # Resume parameters of regressions (params, R squared, Mean Absolute Error (MAE_p))
    list_rbm = ["mu", "sigma", "amplitude", "phi", "entropy gauge"]
    list_epr_rbm = ["mu", "sigma", "amplitude", "phi", "epr gauge"]
    if geometric_flag == True:
        list_rbm = ["mu", "sigma", "amplitude_1", "amplitude_2", "phi", "entropy gauge"]
        list_epr_rbm = ["mu", "sigma", "amplitude_1", "amplitude_2", "phi", "epr gauge"]

    df_final = pd.concat(
        [
            eda_mf.resume_params(
                params = params_transient,
                params_type = "Transient state",
                params_names = ["time", "mu", "sigma", "x0", "t0", "threshold"],
                y = hist_rbm,
                y_fitted = transient_prome,
                p_norm = p_norm,
                significant_figures = significant_figures
            ),
            eda_mf.resume_params(
                params = params_diffusion,
                params_type = "Diffusion law",
                params_names = ["coefficient", "exponent"],
                y = variance_rbm,
                y_fitted = diffusion_prome,
                p_norm = p_norm,
                significant_figures = significant_figures
            ),
            eda_mf.resume_params(
                params = params_entropy,
                params_type = "Entropy",
                params_names = list_rbm,
                y = entropy_rbm,
                y_fitted = entropy_prome,
                p_norm = p_norm,
                significant_figures = significant_figures
            ),
            eda_mf.resume_params(
                params = params_epr,
                params_type = "Entropy production rate",
                params_names = list_epr_rbm,
                y = epr_rbm,
                y_fitted = epr_prome,
                p_norm = p_norm,
                significant_figures = significant_figures
            )
        ]
    )

    # Plot data
    rcParams.update({"font.family": "serif", "text.usetex": usetex, "pgf.rcfonts": False})
    fig, ax = plt.subplots(4, 1)
    fig.set_size_inches(w = width, h = height)

    # Histogram of simulated data plot
    ax[0].hist(
        df_bm[df_bm["time"] == t_max]["value"],
        bins = bins,
        alpha = 0.19,
        facecolor = "blue",
        edgecolor = "darkblue",
        density = density,
        histtype = "stepfilled",
        cumulative = False,
        label = "Simulated data"
    )
    ax[0].scatter(bins_midpoint_rbm, hist_rbm, color = "darkblue", label = "Simulated data")
    ax[0].plot(bins_midpoint_rbm, hist_transient, color = "red", label = "Transient")
    ax[0].plot(bins_midpoint_rbm, transient_prome, color = "orange", label = "Transient fitting")
    ax[0].fill_between(
        bins_midpoint_rbm,
        transient_lower,
        transient_upper,
        where = (
            (transient_upper >= transient_lower) &
            (transient_upper >= transient_prome) &
            (transient_prome >= transient_lower)
        ),
        alpha = 0.19,
        facecolor = "orange",
        interpolate = True,
        label = "Theory fitting"
    )
    
    # Diffusion coefficient plot
    ax[1].plot(mean_rbm, variance_rbm, label = "Diffusion")
    ax[1].plot(mean_rbm, diffusion_prome, color = "darkgreen", label = "Diffusion fitting")
    ax[1].fill_between(
        mean_rbm,
        diffusion_lower,
        diffusion_upper,
        where = (
            (diffusion_upper >= diffusion_lower) &
            (diffusion_upper >= diffusion_prome) &
            (diffusion_prome >= diffusion_lower)
        ),
        alpha = 0.19,
        facecolor = "darkgreen",
        interpolate = True,
        label = "Diffusion fitting"
    )

    # Entropy plot
    ax[2].plot(t_rbm, entropy_rbm, label = "Entropy data")
    ax[2].plot(t_rbm, entropy_prome, color = "darkgreen", label = "Entropy fitting")
    ax[2].fill_between(
        t_rbm,
        entropy_lower * 0 + entropy_prome,
        entropy_upper * 0 + entropy_prome,
        where = (
            (entropy_upper >= entropy_lower) &
            (entropy_upper >= entropy_prome) &
            (entropy_prome >= entropy_lower)
        ),
        alpha = 0.19,
        facecolor = "darkgreen",
        interpolate = True,
        label = "Entropy fitting"
    )
    
    # Entropy production rate plot
    ax[3].hlines(y = 0.0, xmin = 0.0, xmax = t_max, color = "r", linewidth = 2, zorder = 2)
    ax[3].plot(t_midpoint_rbm, epr_rbm, alpha = 0.3, label = "Entropy production rate")
    ax[3].plot(epr_smooth_rbm_time, epr_smooth_rbm_mean, label = "Entropy production rate smoothing")
    ax[3].fill_between(
        epr_smooth_rbm_time,
        epr_smooth_rbm_lower,
        epr_smooth_rbm_upper,
        where = (
            (epr_smooth_rbm_upper >= epr_smooth_rbm_lower) &
            (epr_smooth_rbm_upper >= epr_smooth_rbm_mean) &
            (epr_smooth_rbm_mean >= epr_smooth_rbm_lower)
        ),
        alpha = 0.19,
        facecolor = "orange",
        interpolate = True,
        label = "Entropy production rate smoothing"
    )
    ax[3].plot(t_midpoint_rbm, epr_prome, color = "darkgreen", label = "Entropy production rate fitting")
    ax[3].fill_between(
        t_midpoint_rbm,
        epr_lower,
        epr_upper,
        where = (
            (epr_upper >= epr_lower) &
            (epr_upper >= epr_prome) &
            (epr_prome >= epr_lower)
        ),
        alpha = 0.19,
        facecolor = "darkgreen",
        interpolate = True,
        label = "Entropy production rate fitting"
    )
    
    # Histogram and Entropy plot - Other features ----
    process = "Brownian Motion"
    if geometric_flag == True:
        process = "Geometric Brownian Motion"

    titles_ = [
        "Restricted {} - PDF".format(process),
        "Restricted {} - Diffusion".format(process),
        "Restricted {} - Entropy".format(process),
        "Restricted {} - Entropy production rate".format(process)
    ]
    titles_x = ["$x$", "Mean", "Time", "Time"]
    titles_y = ["$P(x)$", "Variance", "Shannon entropy", "Entropy production rate"]

    for j in [0, 1, 2, 3]:
        ax[j].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
        ax[j].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
        ax[j].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
        ax[j].xaxis.set_minor_locator(mtick.MaxNLocator(4 * n_x_breaks))
        ax[j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        ax[j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))

        #ax[j].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: mf.define_sci_notation_latex(x, significant_figures)))
        #ax[j].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: mf.define_sci_notation_latex(x, significant_figures)))
        #y_tick_labels = mf.define_sci_notation_latex_vectorize(ax[j].yaxis.get_majorticklocs(), significant_figures)
        #ax[j].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: y_tick_labels[pos]))
        
        ax[j].tick_params(axis = "x", labelrotation = 90)
        ax[j].set_xlabel(titles_x[j], fontsize = fontsize_labels)        
        ax[j].set_ylabel(titles_y[j], fontsize = fontsize_labels)
        
        if j == 3:
        #    ax[j].set_xscale(value = "log")        
            ax[j].set_yscale(value = "log")        
        
        ax[j].set_title(
            r"({}) {}".format(chr(j + 65), titles_[j]),
            loc = "left",
            y = 1.005,
            fontsize = fontsize_labels
        )
        ax[j].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)

    plt.show()
    fig.tight_layout()
    if save_figures:
        fig.savefig(
            "{}/eda_{}_{}.png".format(
                output_path,
                information_name,
                re.sub("-", "", input_generation_date)
            ),
            bbox_inches = "tight",
            facecolor = fig.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
        plt.close()
    
    return df_final
