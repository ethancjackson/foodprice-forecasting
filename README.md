# foodprice-forecasting

University of Guelph and Vector Institute contribution to the 2022 edition of Canada's Food Price Report. 

This repository can be used to replicate the experiments described on pages 16 and 17 of the report, under the header *Mixed Ensembles of Machine Learning Models*. Instructions for replicating the forecasts and analysis is below. 

## Dependencies

To replicate the experiments and forecasts, you will need to use these packages to access data and fit models. 


## Setup

To use this repository, please clone or fork it, then install dependencies using a clean Python environment. We used `conda` to create our environment and `pip` to install all dependencies.

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

### Experiments Summary

To produce this year's forecast, we used ensembling experiments to search for aggregate (i.e. mean) forecasts that would have been optimally accurate over the last 6 report years. The notebooks in this repository can be used to replicate our experiment, which is summarized as follows.

For each model configuration, fit a model using data up to a cutoff point, e.g. July 1, 2015. Then use the model to produce a forecast for the following 18 months. Store the forecast using a consistent output and format so that many models and forecasts can be compared as members of ensembles. Repeat this procedure for other cutoff points under consideration. Once many models have been fitted and used to produce forecasts, ensembles may be considered to find optimal combinations of model configurations in retrospective analysis. 

An example result of an ensembling experiment could be as follows:

|     Category    |                                            Best   Ensemble                                          |     Average   MAPE (± Std. Dev.)    |
|:---------------:|:---------------------------------------------------------------------------------------------------:|:-----------------------------------:|
|     **Meat**        |     ['nbeatsfredvars_202110',   'neuralprophet_202110_CUSR0000SAF112_nlags_24', 'nbeats_202110']    |           2.62%   (±1.10%)          |

This indicates that the best ensemble of models and forecasts for **Meat** we found in retrospective analysis over the last 6 report years used two N-BEATS models (with and without FRED variables) and a NeuralProphet model that uses 24 monthly historical observations of prior Meat CPI and an additional lagged regressor (CUSR0000SAF112), which is the U.S. CPI for Meats, Poultry, Fish, and Eggs in urban areas.

### Running Experiments

Experiments can be replicated by following these steps:

- Run the notebook `load_data.ipynb` to produce the data file `all_data.csv`. You will be prompted for a FRED API key.
- Run the notebook `forecasting-prophet-experiment.ipynb` to produce retrospective forecasts using Prophet.
- Run the notebook `forecasting-nbeats-experiment.ipynb` to produce retrospective forecasts using N-BEATS (with historical food CPI only).
- Run the notebook `forecasting-nbeatsfredvars-experiment.ipynb` to produce retrospective forecasts using N-BEATS (with historical food CPI and other FRED variables)
- Run the notebook `forecasting-neuralprophet-experiment.ipynb` to produce many retrospective forecasts using NeuralProphet (many combinations of lag sizes and additional regressors are considered). 

### Producing Forecasts

## Our Results


