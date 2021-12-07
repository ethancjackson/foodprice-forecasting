# foodprice-forecasting

University of Guelph and Vector Institute contribution to the 2022 edition of Canada's Food Price Report. 

This repository can be used to replicate the experiments described on pages 16 and 17 of the report, under the header *Mixed Ensembles of Machine Learning Models*. Instructions for replicating the forecasts and analysis is below. 

## Dependencies

To replicate the experiments and forecasts, you will need to use these packages to access data and fit models. 


## Setup

To use this repository, please clone or fork it, then install dependencies using a Python environment. We used `conda` to create our environment and `pip` to install all dependencies.

### GluonTS & MXNet

We used the N-BEATS implementation provided by GluonTS, which depends on MXNet. If you are planning to use N-BEATS, we recommend that you configure your environment to use a GPU-enabled version of MXNet. Because GPU support for MXNet depends on your system configuration (e.g. GPU type and CUDA version), we cannot provide general instructions here. 

As an example only, we used the following steps to install MXNet and GluonTS:

`pip install --upgrade mxnet-cu110~=1.7`

`pip install gluonts`

For more information or support with installing MXNet and GluonTS, please visit the official [documentation site](https://ts.gluon.ai/install.html) for GluonTS. 

### Installing Other Dependencies

Other dependencies can be installed using the following command:

`pip install -r requirements.txt`

## Overview

To replicate the experiments we used to produce contributed forecasts for the 2022 edition of Canada's Food Price Report, please use the following steps.

### Load and Preprocess Data File

We include a notebook (`load_data.ipynb`) for loading and preprocessing data from Statistics Canada and the Federal Reserve Economic Database (FRED). When running this notebook, you will be prompted to enter a FRED API key, which you can request directly at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html). 

After `load_data.ipynb` runs successfully, a file named `all_data.csv` will be produced. Please check that this file contains rows with datestamps up to the most recently elapsed month.

### Running Experiments

To produce this year's forecast, we used ensembling experiments to search for aggregate (i.e. mean) forecasts that would have been optimally accurate over the last 6 report years. The notebooks in this repository can be used to replicate our experiment.

For example, running the notebook `forecasting-prophet-experiment.ipynb` will produce a number of output files that correspond retrospective forecasting experiments. 
