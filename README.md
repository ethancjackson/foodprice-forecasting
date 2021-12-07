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
|     Meat        |     ['nbeatsfredvars_202110',   'neuralprophet_202110_CUSR0000SAF112_nlags_24', 'nbeats_202110']    |           2.62%   (±1.10%)          |

This indicates that the best ensemble of models and forecasts for **Meat** we found in retrospective analysis over the last 6 report years used two N-BEATS models (with and without FRED variables) and a NeuralProphet model that uses 24 monthly historical observations of prior Meat CPI and an additional lagged regressor (CUSR0000SAF112), which is the U.S. CPI for Meats, Poultry, Fish, and Eggs in urban areas.

### Running Experiments

Experiments can be replicated by following these steps:

1. Run the notebook `load_data.ipynb` to produce the data file `all_data.csv`. You will be prompted for a FRED API key.
2. Run the notebook `forecasting-prophet-experiment.ipynb` to produce retrospective forecasts using Prophet.
3. Run the notebook `forecasting-nbeats-experiment.ipynb` to produce retrospective forecasts using N-BEATS (with historical food CPI only).
4. Run the notebook `forecasting-nbeatsfredvars-experiment.ipynb` to produce retrospective forecasts using N-BEATS (with historical food CPI and other FRED variables)
5. Run the notebook `forecasting-neuralprophet-experiment.ipynb` to produce many retrospective forecasts using NeuralProphet (many combinations of lag sizes and additional regressors are considered). 

After these each of these notebooks have finished running, the resulting retrospective forecasts will be stored under the `./output` directory. For example, after running the notebook `forecasting-prophet-experiment.ipynb` using the default `output_path = 'prophet_TEST'` in Cell 2, you should find the following files in `./output/prophet_TEST`:

- forecasts_2015-07-01.csv
- forecasts_2016-07-01.csv
- forecasts_2017-07-01.csv
- forecasts_2018-07-01.csv
- forecasts_2019-07-01.csv
- forecasts_2010-07-01.csv
- mean_fc_valid_metrics.csv

You may also find other output files (i.e. plots and figures) if options to save figures are left enabled. 

### Producing Forecasts

After successfully loading data and running the `experiment` notebooks to produce retrospective forecasts, we can move on to searching for ensembles of retrospective forecasts that have the lowest average error (MAPE).

6. Run the notebook `forecasting-ensemble-experiment.ipynb` to produce the ensemble experiment output file `ensemble_results.pkl`.

The file `ensemble_results.pkl` is a serialized Pandas DataFrame containing two types of important information. For each food category, it specifies the composition of the best ensemble in retrospective analysis (e.g. `prophet_TEST`, `nbeatsfredvars_TEST`, and `neuralprophet_202110_CUSR0000SAF112_nlags_24_TEST`) and the corresponding scores (mean and standard deviation MAPE). These scores can be used to charactarize uncertainty around future forecasts. 

With these results in hand, we can now produce this year's forecasts by refitting models (selected according to retrospective performance) that consume all available data.

7. Run the notebook `forecasting-prophet-final.ipynb` to produce final forecasts using Prophet.
8. Run the notebook `forecasting-nbeats-final.ipynb` to produce final forecasts using NBEATS (with historical food CPI only).
9. Run the notebook `forecasting-nbeatsfredvars-final.ipynb` to produce final forecasts using N-BEATS (with historical food CPI and other FRED variables)
10. Run the notebook `forecasting-neuralprophet-final.ipynb` to produce final forecasts using NeuralProphet, which depends on the presence of `ensemble_results.pkl` to indicate which features and number of lags to use.

Finally, aggregate and analyze ensembles of final forecasts. All figures and results (i.e. predicted change in CPI) are produced within the notebook.

11. Run the notebook `forecasting-ensemble-final.ipynb` to produce this year's forecasts and analysis.

## Our Results

The results of our retrospective (model selection) experiment are summarized below. The ensemble configurations in the middle column were then used to fit the final forecasting models used to produce our contribution to the 2022 Canada's Food Price Report. 

|                          Category                         |                                                                        Best   Ensemble                                                                       |     Average   MAPE (± Std. Dev.)    |
|:---------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------:|
|     Bakery   and cereal products (excluding baby food)    |     ['neuralprophet_202110_IRLTLT01CAM156N_nlags_36',   'nbeatsfredvars_202110', 'neuralprophet_202110_CUSR0000SAF112_nlags_24']                             |     1.40%   (±0.88%)                |
|     Dairy   products and eggs                             |     ['neuralprophet_202110_CUSR0000SAF112_nlags_48',   'nbeats_202110']                                                                                      |     1.55%   (±0.53%)                |
|     Fish,   seafood and other marine products             |     ['neuralprophet_202110_CUSR0000SAF113_nlags_48',   'nbeats_202110', 'neuralprophet_202110_QCAR368BIS_nlags_60']                                          |     1.09%   (±0.20%)                |
|     Food   purchased from restaurants                     |     ['nbeatsfredvars_202110',   'nbeats_202110', 'neuralprophet_202110_WILL5000IND_nlags_36']                                                                |     0.61%   (±0.37%)                |
|     Food                                                  |     ['nbeatsfredvars_202110',   'neuralprophet_202110_CUSR0000SAF112_nlags_24',   'neuralprophet_202110_IRLTLT01CAM156N_nlags_60']                           |     0.76%   (±0.26%)                |
|     Fruit,   fruit preparations and nuts                  |     ['neuralprophet_202110_CUSR0000SAF112_nlags_24',   'neuralprophet_202110_CUSR0000SAF112_nlags_36',   'neuralprophet_202110_CPALCY01CAM661N_nlags_24']    |     2.04%   (±0.71%)                |
|     Meat                                                  |     ['nbeatsfredvars_202110',   'neuralprophet_202110_CUSR0000SAF112_nlags_24', 'nbeats_202110']                                                             |     2.62%   (±1.10%)                |
|     Other   food products and non-alcoholic beverages     |     ['neuralprophet_202110_VXOCLS_nlags_36',   'neuralprophet_202110_CUSR0000SAF112_nlags_24', 'prophet_202110']                                             |     0.86%   (±0.39%)                |
|     Vegetables   and vegetable preparations               |     ['neuralprophet_202110_DEXCAUS_nlags_24',   'neuralprophet_202110_WILL5000IND_nlags_60',   'neuralprophet_202110_CPGRLE01CAM657N_nlags_60']              |     2.94%   (±1.16%)                |
