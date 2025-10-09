"""
Calibration takes in raw data and develops the parameters that will be used to
seed the inference codes.

The goal with the below functions is to fit some parameters. The parameters that need to be fit are:

    1. slope
    2. sensitivity_parameter
    3.

"""
import pandas as pd


def calibration(eia_data: pd.DataFrame,
                weather_data: pd.DataFrame) -> dict:
    """
    Calibration of a variety of parameters using input data.

    """

    slope_parameter = fit_slope(eia_data)
    sensitivity_parameter = fit_sensitivity_parameter()

    return {"slope": slope_parameter,
            "sensitivity_parameter": sensitivity_parameter}

def fit_slope(monthly_time_series):
    """
    A monthly time series is provided, and this aims to calculate the slope.


    """
    pass


def fit_sensitivity_parameter(consumption_ts, weather_ts):
    """
    Fits the (1) consumption_ts and (2) weather_ts.

    """
    pass






