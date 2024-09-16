# Generalized Shannon Index

## File structure

The structure of the data repository consists of:

*   ***Input files:*** It corresponds to the input data sets for the study developed with the generalized Ito diffusive processes. There are two types of input files:
    *   **raw_data:** The raw data can correspond to .
    *   **processed_data:** The processed data corresponds to the processing of different time series data where.

*   ***Modules:*** It corresponds to the different modules developed in Python for the development of the study, namely:
    *   **estimate_.py:** Module designed for the calculation of the g

*   ***Logs (optional):*** It corresponds to an optional folder in which different log files are generated to know what is failing in any of the parallelized functions in the different modules of the data repository if any of these files suddenly stops working.

*   ***Output files:*** It corresponds to the folder with the output files after processing different data sets. For example, in this folder, the figures and tables for analysis will be by default. Some of these analyses are to show the estimation of the.

*   ***Scripts:*** It corresponds to different Jupyter notebooks where the study analyses were carried out and to emphasize some additional aspects, a section is dedicated to them later.

## Metadata of the data sets

The metadata of the different data sets that appear in this repository are organized by the ```.csv``` or ```.xlsx``` files placed in the input_files and output_files folders, namely:

## Scripts order

The set of codes developed for this data repository is divided into two parts specified below.

### Probability density function (PDE)

To estimate a 

### Entropy production rate (fts)

To show the 

## Code/Software

All the information shown in this data repository is organized in the different folders mentioned and with all the files shown in the following public Github repository [[1]](#references).

To run the different notebooks in the ```scripts``` folder, it is recommended to use version 2.1.4 of ```pandas``` and version 1.24.4 of ```numpy```. Also, it is recommended to install other Python libraries such as ```yfinance```, ```MFDFA``` and ```tqdm```.

## References

\[1] F. Abril. *Fokker Planck Distribution*. Github repository. Available on: [https://github.com/fsabrilb/Fokker_Planck_Distribution](https://github.com/fsabrilb/Fokker_Planck_Distribution)
