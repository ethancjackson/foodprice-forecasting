from data import *
import pandas as pd
from pandas.tseries.offsets import Day, Week, MonthEnd, YearEnd, Hour, MonthBegin
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
import pickle


def data_generator(X, y, data_indices, lag_size, lead_size, batch_size=-1, randomize=False):
    x_batch, y_batch = [], []
    x_dates, y_dates = [], []
    batch_indices = data_indices.copy()
    if randomize:
        np.random.shuffle(batch_indices)
    for i, batch_index in enumerate(batch_indices):
        x_batch.append(X[batch_index: batch_index + lag_size])
        x_dates.append(X.index[batch_index: batch_index + lag_size])
        y_batch.append(y[batch_index + lag_size: batch_index + lag_size + lead_size])
        y_dates.append(y.index[batch_index + lag_size: batch_index + lag_size + lead_size])
        if len(x_batch) == batch_size:
            x_ = np.stack(x_batch)
            y_ = np.stack(y_batch)
            x_d = np.stack(x_dates)
            y_d = np.stack(y_dates)
            yield x_, y_, x_d, y_d
            x_batch, y_batch = [], []
    if len(x_batch) > 0:
        x_ = np.stack(x_batch)
        y_ = np.stack(y_batch)
        x_d = np.stack(x_dates)
        y_d = np.stack(y_dates)
        yield x_, y_, x_d, y_d


def get_indices(df, lag_size, lead_size, train_size, valid_size):
    all_indices = np.arange(0, len(df) - lag_size - lead_size)
    train_indices = all_indices[:int(train_size * len(all_indices))]
    valid_indices = all_indices[int(train_size * len(all_indices)):int((train_size + valid_size) * len(all_indices))]
    test_indices = all_indices[int((train_size + valid_size) * len(all_indices)):]
    return train_indices, valid_indices, test_indices


def train_valid_test_split(x_df, y_df, lag_size, lead_size, train_size, valid_size, test_only=False):
    # Split data into train / test sets
    train_index, valid_index, test_index = get_indices(x_df, lag_size, lead_size,
                                                       train_size=train_size, valid_size=valid_size)
    if not test_only:

        x, y, x_dates, y_dates = list(data_generator(x_df, y_df, train_index, lag_size, lead_size))[0]

        if valid_size > 0:
            x_v, y_v, x_v_dates, y_v_dates = list(data_generator(x_df, y_df, valid_index, lag_size, lead_size))[0]
        else:
            x_v, y_v, x_v_dates, y_v_dates = None, None, None, None
    else:
        x, y, x_dates, y_dates = None, None, None, None
        x_v, y_v, x_v_dates, y_v_dates = None, None, None, None

    if 1 - train_size - valid_size > 0:
        x_t, y_t, x_t_dates, y_t_dates = list(data_generator(x_df, y_df, test_index, lag_size, lead_size))[0]
    else:
        x_t, y_t, x_t_dates, y_t_dates = None, None, None, None

    return {'x_train': x, 'x_train_dates': x_dates, 'y_train': y, 'y_train_dates': y_dates,
            'x_valid': x_v, 'x_valid_dates': x_v_dates, 'y_valid': y_v, 'y_valid_dates': y_v_dates,
            'x_test': x_t, 'x_test_dates': x_t_dates, 'y_test': y_t, 'y_test_dates': y_t_dates}


def get_date_offset(sample_freq, steps):
    offset_map = {'D': Day, 'W': Week, 'M': MonthBegin, 'Y': YearEnd, 'H': Hour}
    return offset_map[sample_freq](steps)


class SKDirect:
    """
    A wrapper class for any scikit-learn regression model.
    """

    def __init__(self, model_class, model_name, lag_size, lead_size, sample_freq, sk_params=None):
        self.model_class = model_class
        self.model_name = model_name
        self.lag_size = lag_size
        self.lead_size = lead_size
        self.sample_freq = sample_freq
        self.sk_params = sk_params
        self.lead_models = None

    def format_features(self, x):
        # Flatten features
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        return x

    def fit(self, x, y):

        x = self.format_features(x)

        # Separate lead targets into dict
        lead_targets = {lead: y[:, lead] for lead in range(y.shape[1])}
        lead_models = {}
        lead_training_errors = {}

        for lead, y_lead in lead_targets.items():
            # Fit sklearn model
            model = self.model_class(**self.sk_params)
            model.fit(x, y_lead)
            lead_models[lead] = model
            train_score = mean_absolute_error(y_true=y_lead, y_pred=model.predict(x))
            # print(f"Lead {lead}, {self.model_name}\tError: {train_score:.3f}", end='\r')
            lead_training_errors[lead] = train_score

        self.lead_models = lead_models

    def predict(self, x, x_dates, target_names):

        forecasts = []
        for example_index, example in enumerate(x):
            predicted_dates = []
            predicted_values = []
            init_date = pd.to_datetime(x_dates[example_index][-1]) + get_date_offset(self.sample_freq, 1)
            for lead, model in sorted(self.lead_models.items(), key=lambda x: x[0]):
                predicted_dates.append(init_date + get_date_offset(self.sample_freq, lead))
                predicted_values.append(model.predict([example]))
            predicted_values = np.array(predicted_values).reshape(len(predicted_values), len(target_names)).T
            forecast = pd.DataFrame({target_names[i]: predicted_values[i] for i in range(len(target_names))})
            forecast.index = pd.to_datetime(predicted_dates)
            forecasts.append(forecast)
        return forecasts

    def predict_df(self, x_df, expl_names, target_names):
        x_df = x_df[expl_names]
        x = x_df.iloc[-self.lag_size:].values
        x = self.format_features(np.array([x]))
        x_dates = x_df.index[-self.lag_size:].values
        return self.predict(x, [x_dates], target_names)[0]

    def score(self, x, x_dates, y, y_dates, target_names):

        x = self.format_features(x)

        forecast_dfs = self.predict(x, x_dates, target_names)
        actual_dfs = []
        score_dfs = []
        total_error = 0
        for example_index, example in enumerate(y):
            actual_values = np.array(example).reshape(len(example), len(target_names)).T
            actual_df = pd.DataFrame({target_names[i]: actual_values[i] for i in range(len(target_names))})
            actual_df.index = pd.to_datetime(y_dates[example_index])
            actual_dfs.append(actual_df)
            score_df = np.abs(actual_dfs[example_index] - forecast_dfs[example_index])
            score_dfs.append(score_df)
            total_error += np.mean(score_df)
        return total_error / len(y), score_dfs, actual_dfs, forecast_dfs


class VectorAutoRegressive:
    """
    Wrapper class for statsmodels VAR model.
    """

    def __init__(self, all_targets, forecast_targets, lag_size, lead_size, sample_freq, statsmodels_params=None):
        self.all_targets = all_targets
        self.forecast_targets = forecast_targets
        self.lag_size = lag_size
        self.lead_size = lead_size
        self.sample_freq = sample_freq
        self.statsmodels_params = statsmodels_params
        self.model = None
        self.model_results = None

    def fit(self, x, y=None):
        self.model = VAR(x)
        self.model_results = self.model.fit(self.lag_size)

    def predict(self, x, x_dates, target_names):
        forecasts = []
        for example_index, example in enumerate(x):
            init_date = pd.to_datetime(x_dates[example_index][-1]) + get_date_offset(self.sample_freq, 1)
            forecast_df = pd.DataFrame(self.model_results.forecast(example, self.lead_size), columns=self.all_targets)
            dates = [pd.to_datetime(init_date) + get_date_offset('M', offset + 1) for offset in forecast_df.index]
            forecast_df = forecast_df.set_index(pd.DatetimeIndex(dates))
            forecasts.append(forecast_df[self.forecast_targets])
        return forecasts

    def predict_df(self, x_df, target_names):
        x = x_df.iloc[-self.lag_size:].values
        x_dates = x_df.index[-self.lag_size:].values
        return self.predict(x, [x_dates], target_names)[0]

    def score(self, x, x_dates, y, y_dates, target_names):
        forecast_dfs = self.predict(x, x_dates, target_names)
        actual_dfs = []
        score_dfs = []
        total_error = 0

        for example_index, example in enumerate(y):
            actual_values = np.array(example).reshape(len(example), len(target_names)).T
            actual_df = pd.DataFrame({target_names[i]: actual_values[i] for i in range(len(target_names))})
            actual_df.index = pd.to_datetime(y_dates[example_index])
            actual_dfs.append(actual_df)
            score_df = np.abs(actual_dfs[example_index] - forecast_dfs[example_index])
            score_dfs.append(score_df)
            total_error += np.mean(score_df)
        return total_error / len(y), score_dfs, actual_dfs, forecast_dfs


class SARIMAX:
    """
    Wrapper class for SARIMAX-based models.
    Seasonal Auto Regression of Integrated Moving Averages with eXogenous features.
    Here, though, we probably won't use exogenous features.
    """
    def __init__(self, all_targets, forecast_targets, lag_size, lead_size, sample_freq, statsmodels_params=None):
        self.all_targets = all_targets
        self.forecast_targets = forecast_targets
        self.lag_size = lag_size
        self.lead_size = lead_size
        self.sample_freq = sample_freq
        self.statsmodels_params = statsmodels_params
        self.model = None
        self.model_results = None
        self.init_forecast_date = None

    def fit(self, x, y=None):
        self.model = ARIMA(x, **self.statsmodels_params)
        self.model_results = self.model.fit()
        self.init_forecast_date = pd.to_datetime(x.iloc[-1][-1]) + get_date_offset(self.sample_freq, 1)

    def predict(self, x, x_dates, target_names, y_dates=None):
        n_points = len(pd.date_range(start=self.init_forecast_date, end=y_dates[-1][-1], freq=self.sample_freq))
        arima_forecast = self.model_results.forecast(n_points)
        arima_forecast.index = pd.DatetimeIndex(pd.to_datetime(arima_forecast.index))
        forecasts = []
        for dates_index, d in enumerate(y_dates):
            dates = pd.DatetimeIndex(d)
            forecasts.append(pd.DataFrame(arima_forecast[dates], columns=target_names))
        return forecasts

    def predict_df(self, x_df, target_names):
        x = x_df.iloc[-self.lag_size:].values
        x_dates = x_df.index[-self.lag_size:].values
        return self.predict(x, [x_dates], target_names)[0]

    def score(self, x, x_dates, y, y_dates, target_names):
        forecast_dfs = self.predict(x, x_dates, target_names, y_dates)
        actual_dfs = []
        score_dfs = []
        total_error = 0

        for example_index, example in enumerate(y):
            actual_values = np.array(example).reshape(len(example), len(target_names)).T
            actual_df = pd.DataFrame({target_names[i]: actual_values[i] for i in range(len(target_names))})
            actual_df.index = pd.to_datetime(y_dates[example_index])
            actual_dfs.append(actual_df)
            score_df = np.abs(actual_dfs[example_index] - forecast_dfs[example_index])
            score_dfs.append(score_df)
            total_error += np.mean(score_df)

        return total_error / len(y), score_dfs, actual_dfs, forecast_dfs
    

class NbeatsUnivariate:
    """
    Wrapper class for univariate N-BEATS model
    """

    def __init__(self):
        pass


class ForecastPipeline:

    def __init__(self,
                 expl_names,
                 target_names,
                 include_targets,
                 sample_freq,
                 model_type,
                 train_size,
                 valid_size,
                 model_dict):
        self.expl_names = expl_names
        self.target_names = target_names
        self.expl_df = None
        self.targets_df = None
        self.feature_names = self.expl_names.copy()
        if include_targets:
            self.feature_names += self.target_names
        self.include_targets = include_targets
        self.sample_freq = sample_freq
        self.model_type = model_type
        self.train_size = train_size
        self.valid_size = valid_size
        self.model_dict = model_dict
        self.model = None
        self.lag_size = self.model_dict['lag_size']
        self.lead_size = self.model_dict['lead_size']
        self.data_dict = None

    def set_data(self, expl_df, targets_df):
        self.expl_df = expl_df
        self.targets_df = targets_df
        
    def set_data_dict(self, data_dict):
        self.data_dict = data_dict

    def load_data(self):
        return self.expl_df, self.targets_df

    def load_data_dict_(self):
        # Load data
        expl_df, targets_df = self.load_data()

        # Split data
        self.data_dict = train_valid_test_split(expl_df, targets_df,
                                                self.lag_size, self.lead_size,
                                                self.train_size, self.valid_size)

    def fit(self, load_data_dict=True):

        # Train model
        self.model = self.model_type(**self.model_dict)
        if self.model_type is SKDirect:
            # Load data and split into data_dict for training and prediction
            if load_data_dict:
                self.load_data_dict_()
            self.model.fit(self.data_dict['x_train'], self.data_dict['y_train'])
        elif self.model_type in (VectorAutoRegressive, SARIMAX):
            expl_df, targets_df = self.load_data()
            expl_df = expl_df.iloc[:int(self.train_size * len(expl_df))]
            # expl_df = expl_df.iloc[:-self.lag_size]
            self.model.fit(expl_df)

    def forecast(self, df):
        return self.model.predict_df(df, self.feature_names, self.target_names)

    def forecast_by_date(self, df, max_date):
        df = df.loc[df.index <= max_date]
        df = df.iloc[-self.lag_size:]
        return self.forecast(df)

    def actual_by_date(self, df, min_date):
        df = df.loc[df.index >= min_date]
        df = df.iloc[:self.lead_size]
        return df

    def train_score(self):
        if self.data_dict is None:
            self.load_data_dict_()

        if self.data_dict['x_train'] is not None:
            return self.model.score(self.data_dict['x_train'], self.data_dict['x_train_dates'],
                                    self.data_dict['y_train'], self.data_dict['y_train_dates'],
                                    self.target_names)[0]
        
    def train_score_full(self):
        if self.data_dict is None:
            self.load_data_dict_()

        if self.data_dict['x_train'] is not None:
            return self.model.score(self.data_dict['x_train'], self.data_dict['x_train_dates'],
                                    self.data_dict['y_train'], self.data_dict['y_train_dates'],
                                    self.target_names)

    def valid_score(self):
        if self.data_dict is None:
            self.load_data_dict_()

        if self.data_dict['x_valid'] is not None:
            return self.model.score(self.data_dict['x_valid'], self.data_dict['x_valid_dates'],
                                    self.data_dict['y_valid'], self.data_dict['y_valid_dates'],
                                    self.target_names)[0]
        
    def valid_score_full(self):
        if self.data_dict is None:
            self.load_data_dict_()

        if self.data_dict['x_valid'] is not None:
            return self.model.score(self.data_dict['x_valid'], self.data_dict['x_valid_dates'],
                                    self.data_dict['y_valid'], self.data_dict['y_valid_dates'],
                                    self.target_names)

    def test_score(self):
        if self.data_dict is None:
            self.load_data_dict_()

        if self.data_dict['x_test'] is not None:
            return self.model.score(self.data_dict['x_test'],
                                    self.data_dict['x_test_dates'],
                                    self.data_dict['y_test'],
                                    self.data_dict['y_test_dates'],
                                    self.target_names)[0]
        
    def test_score_full(self):
        if self.data_dict is None:
            self.load_data_dict_()

        if self.data_dict['x_test'] is not None:
            return self.model.score(self.data_dict['x_test'],
                                    self.data_dict['x_test_dates'],
                                    self.data_dict['y_test'],
                                    self.data_dict['y_test_dates'],
                                    self.target_names)

    def save(self, filename):
        data_dict = self.data_dict
        self.data_dict = None
        pickle.dump(self, open(f"{filename}", 'wb'))
        self.data_dict = data_dict
