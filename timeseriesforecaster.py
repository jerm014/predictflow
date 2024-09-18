#!/usr/bin/env python3
"""Module for time series forecasting using Prophet."""

import os
import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


class TimeSeriesForecaster:
    """TimeSeriesForecaster class"""
    def __init__(self):
        """Initialize the TimeSeriesForecaster class"""
        self.name = "TimeSeriesForecaster"
        self.model = Prophet()

    def preprocess_data(self, data):
        """Preprocess the data for Prophet"""
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        # Ensure data is in the correct format for Prophet
        data = data.rename(columns={'timestamp': 'ds', 'value': 'y'})
        return data

    def train(self, data):
        """Train the Prophet model"""
        preprocessed_data = self.preprocess_data(data)
        self.model.fit(preprocessed_data)

    def predict(self, periods):
        """Predict using time periods"""
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast

    def evaluate(self, actual, predicted):
        """Evaluate MAE and RMSE"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        return {'MAE': mae, 'RMSE': rmse}

"""
# Usage
forecaster = TimeSeriesForecaster()
data = pd.read_csv('your_time_series_data.csv')
forecaster.train(data)
forecast = forecaster.predict(periods=30)
# Print the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
"""
