# foodprice-forecasting

Notebooks and tools for forecasting consumer price index (CPI) using scikit-learn models and Optuna.

## Dependencies

- [Optuna](https://optuna.org/)
- [FRED API](https://pypi.org/project/fredapi/)
- [stats_can API](https://github.com/ianepreston/stats_can)

## Overview

The notebook [multisklearn-foodprice_forecasting_notebook.ipynb](multisklearn-foodprice_forecasting_notebook.ipynb) demonstrates data loading, preprocessing, visualization, model fitting, and forecasting using the included tools. Functions for loading data from Statistics Canada and the Federal Reserve Bank of St. Louis are provided in [data.py](data.py). The notebook uses ForecastPipelines as defined in [models.py](models.py) as wrappers for multiple, independently trained models that generate future point predictions. This is an example of _direct forecasting_ methods, as opposed to recursive methods or sequence prediction methods. 

The included Optuna study can be modified to work with any scikit-learn compatible regression model. 

