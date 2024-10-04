***Preprint:*** [arXiv:2410.01387 [math-ph]](https://arxiv.org/abs/2410.01387v1)

# Path Integral for Multiplicative Noise: Generalized Fokker-Planck Equation and Entropy Production Rate in Stochastic Processes With Threshold

This paper introduces a comprehensive extension of the path integral formalism to model stochastic processes with arbitrary multiplicative noise. To do so, Itô diffusive process is generalized by incorporating a multiplicative noise term $(\eta(t))$ that affects the diffusive coefficient in the stochastic differential equation. Then, using the Parisi-Sourlas method, we estimate the transition probability between states of a stochastic variable $(X(t))$ based on the cumulant generating function $(\mathcal{K}_{\eta})$ of the noise. A parameter $\gamma\in[0,1]$ is introduced to account for the type of stochastic calculation used and its effect on the Jacobian of the path integral formalism. Next, the Feynman-Kac functional is then employed to derive the Fokker-Planck equation for generalized Itô diffusive processes, addressing issues with higher-order derivatives and ensuring compatibility with known functionals such as Onsager-Machlup and Martin-Siggia-Rose-Janssen-De Dominicis in the white noise case. The general solution for the Fokker-Planck equation is provided when the stochastic drift is proportional to the diffusive coefficient and $\mathcal{K}_{\eta}$ is scale-invariant. Finally, the Brownian motion ($BM$), the geometric Brownian motion ($GBM$), the Levy $\alpha$-stable flight ($LF(\alpha)$), and the geometric Levy $\alpha$-stable flight ($GLF(\alpha)$) are simulated with thresholds, providing analytical comparisons for the probability density, Shannon entropy, and entropy production rate. It is found that restricted $BM$ and restricted $GBM$ exhibit quasi-steady states since the rate of entropy production never vanishes. It is also worth mentioning that in this work the $GLF(\alpha)$ is defined for the first time in the literature and it is shown that its solution is found without the need for Itô's lemma.

## File structure

The structure of the data repository consists of:

*   ***Modules:*** It corresponds to the different modules developed in Python for the development of the study, namely:
    *   **estimate_brownian_motion.py:** Module designed for the simulation of the restricted Brownian motion ($RBM$) and for the implementation of theoretical expresions for the temporal Renyi entropy.
    *   **estimate_geometric_brownian_motion.py:** Module designed for the simulation of the restricted geometric Brownian motion ($RGBM$) and for the implementation of theoretical expresions for the temporal Renyi entropy.
    *   **estimate_stochastic_processes.py:** Module designed for the simulation of $RBM$, $RGBM$, restricted Levy $\alpha$-stable flight ($RLF(\alpha)$), and the restricted geometric Levy $\alpha$-stable flight ($GLF(\alpha)$) and for the implementation of theoretical expresions for the probability density function according to the path integral formalism results.
    *   **misc_functions.py:** Module designed for the auxiliary functions used in all modules. For instance, the parallelization of simulations.
    *   **eda_brownian_motion.py:** Module designed for the exploratory data analysis (plots) of the $RBM$.
    *   **eda_geometric_brownian_motion.py:** Module designed for the exploratory data analysis (plots) of the $RGBM$.
    *   **eda_levy_flight.py:** Module designed for the exploratory data analysis (plots) of the $RLF(\alpha)$ and $RGLF(\alpha)$.
    *   **eda_misc_functions.py:** Module designed for the auxiliary functions used in all EDA modules. For instance, the estimation of the coefficient of determination ($R^{2}$) or the mean absolute error (MAE).


*   ***Logs (optional):*** It corresponds to an optional folder in which different log files are generated to know what is failing in any of the parallelized functions in the different modules of the data repository if any of these files suddenly stops working.

*   ***Output files:*** It corresponds to the folder with the output files after processing different data sets. For example, in this folder, the figures and tables for analysis will be by default.

*   ***Scripts:*** It corresponds to different Jupyter notebooks where the study analyses were carried out and to emphasize some additional aspects, a section is dedicated to them later.

## Metadata of the data sets

The metadata of the different dataframes obtained from the modules has similar structure where the ```time``` column refers to the computational time to integrate the stochastic processes and ```value``` column is the value of the stochastic processes for each time. Finally, other variables as ```simulation``` or ```restricted``` are in the dataframes and represents the simulation number (simulation ID) and the impose of a threshold in the stochastic process, repectively.

## Scripts order

The set of codes developed for this data repository is divided into two parts specified below.

### Probability density function (PDF)

To estimate a PDF evolution: ```estimate_distribution_sp.ipynb```

### Entropy production rate (fts)

To estimate a entropy production rate: ```estimate_distribution_bm.ipynb``` and ```estimate_distribution_gbm.ipynb```

## Code/Software

All the information shown in this data repository is organized in the different folders mentioned and with all the files shown in the following public Github repository [[1]](#references).

To run the different notebooks in the ```scripts``` folder, it is recommended to use version 2.1.4 of ```pandas``` and version 1.24.4 of ```numpy```. Also, it is recommended to install other Python libraries such as ```tqdm```.

## References

\[1] F. Abril. *Fokker Planck Distribution*. Github repository. Available on: [https://github.com/fsabrilb/Fokker_Planck_Distribution](https://github.com/fsabrilb/Fokker_Planck_Distribution)
