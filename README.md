# foodprice-forecasting

Notebooks and tools for forecasting consumer price index (CPI) using scikit-learn models and Optuna.

## Dependencies

- [Optuna](https://optuna.org/)
- [FRED API](https://pypi.org/project/fredapi/)
- [stats_can API](https://github.com/ianepreston/stats_can)

## Usage

The notebook (multisklearn-foodprice_forecasting_notebook.ipynb)[multisklearn-foodprice_forecasting_notebook.ipynb] demonstrates data loading, preprocessing, visualization, model fitting, and forecasting using the included tools. The notebook uses ForecastPipelines as defined in (models.py)[models.py] as wrappers for multiple, independently trained models that generate future point predictions. This is an example of _direct forecasting_ methods, as opposed to recursive methods or sequence prediction methods. 

