"""
Calibration takes in raw data and develops the parameters that will be used to
seed the inference codes.

The goal with the below functions is to fit some parameters. The parameters that need to be fit are:

    1. slope
    2. sensitivity_parameter
    3.

"""
import pandas as pd
from data.eia_consumption import eia_consumption
import logging
from scipy import stats



#TODO: Need to review the picks made here.

def fit_daily_consumption_error(eia_data: pd.DataFrame, state: str):
    """
    Fits the daily consumption error.

    :return:
    """

    return eia_data[state].astype(float).mean() / 30

def fit_minimum_consumption(eia_data: pd.DataFrame, state: str):
    """
    The major goal is to calculate the minimum consumption factor.

    :return:
    """
    return (eia_data[state].astype(float).abs().mean()) / 30

def fit_minimum_consumption_sig(eia_data, state):
    return 0.2 * fit_minimum_consumption(eia_data, state)

def fit_theta_1_mu_parameter(consumption_factor, eia_data: pd.DataFrame, state: str):
    return 0.1 * eia_data[state].astype(float).abs().mean() / 30

def fit_theta_1_sig_parameter(consumption_factor, eia_data: pd.DataFrame, state: str):
    return 0.1 * 0.1 * eia_data[state].astype(float).abs().mean() / 30

def fit_theta_2_mu_parameter(consumption_factor, eia_data: pd.DataFrame, state: str):
    return 0.1 * eia_data[state].astype(float).abs().mean() / 30

def fit_theta_2_sig_parameter(consumption_factor, eia_data: pd.DataFrame, state: str):
    return 0.1 * 0.1 * eia_data[state].astype(float).abs().mean() / 30


def fit_monthly_consumption_error(eia_data: pd.DataFrame, state: str):
    return 0.1

def calibration(consumption_factor,
                eia_data: pd.DataFrame,
                state: str
                ) -> dict:
    """
    Calibration of a variety of parameters using input data.

    """

    slope_parameter = fit_slope(eia_data, state)
    sensitivity_parameter = fit_sensitivity_parameter(consumption_factor, eia_data, state)
    minimum_consumption = fit_minimum_consumption(eia_data, state)
    daily_consumption_error = fit_daily_consumption_error(eia_data, state)
    monthly_consumption_error = fit_monthly_consumption_error(eia_data, state)


    return {"slope": slope_parameter,
            "alpha_mu": 0.7 * sensitivity_parameter,
            "alpha_2_mu": 0.3 * sensitivity_parameter,
            "alpha_sigma": 0.3 * sensitivity_parameter,
            "alpha_2_sigma": 0.3 * sensitivity_parameter,
            "daily_consumption_error": daily_consumption_error,
            "monthly_consumption_error": monthly_consumption_error
            }

def fit_slope(eia_monthly_time_series, state: str):
    """
    A monthly time series is provided, and this aims to calculate the slope.

    Calculate the day over day adjustment.

    As an example of the information that was found in eia_monthly_time_series.
    The relevant calculations can be made from below.

    '01': [19365, 13158], = -6000 = -6000 / 13000 = 30 percent
    '02': [13591, 10478], = -3000 = 30%
    '03': [8991, 9964],   = 1000 = 10%
    '04': [4455, 4160],   = -300 = 8%
    '05': [2909, 2774],   = -200 = 10%
    '06': [1815, 2103],   = +200 = 10%
    '07': [1487, 1579],   = +100 = 6%
    '08': [1408, 1510],   = +100 = 6%
    '09': [1760, 1757],   = +100 = 5%
    '10': [4359, 3275],   = -1000 = 33%
    '11': [8400, 9316],   = +1000 = 10%
    '12': [15716, 12729]} = -3000 = 25%

    Mean error is 13.75%.

    """

    monthly_values = dict()
    for period, row in eia_monthly_time_series.iterrows():
        month = period.split("-")[1]
        l = monthly_values.get(month, [])
        l.append(int(row[state]))
        monthly_values[month] = l

    slope_values = []
    for key in monthly_values:
        slope_vals_for_month = []
        if len(monthly_values[key]) > 1:
            for i in range(1, len(monthly_values[key])):
                slope = ((monthly_values[key][i] / 30) - (monthly_values[key][i - 1] / 30)) / 365.0
                slope_vals_for_month.append(slope)

            avg = sum(slope_vals_for_month) / len(slope_vals_for_month)
            slope_values.append(avg)


    if len(slope_values) == 0:
        return float("nan")
    else:
        slope_average = sum(slope_values) / len(slope_values)
    return slope_average

def fit_sensitivity_parameter(consumption_ts, eia_data, state: str):
    """
    Fits the (1) consumption_ts and (2) weather_ts.

    There is a certain amount of consumption factor.

    The consumption factor will be correlated with the EIA data.

    We need to develop an estimate for this.

    We can look at the (a) total variation in consumption and (b) total variation in
    consumption factor.

    The range of (a) gets mapped into (b).

    """

    consumption_factor_min = consumption_ts["Consumption_Factor_Normalizied"].min()
    consumption_factor_max = consumption_ts["Consumption_Factor_Normalizied"].max()

    consumption_ts["Year"] = consumption_ts["Date"].dt.year
    consumption_ts["Month"] = consumption_ts["Date"].dt.month
    consumption_ts["Day"] = consumption_ts["Date"].dt.day

    eia_data_min = int(eia_data[state].astype(float).min()) / 30
    eia_data_max = int(eia_data[state].astype(float).max()) / 30

    logging.info(f"Calibration Datasets are consumption factor:"
                 f" {consumption_ts} and "
                 f"eia data: {eia_data} "
                 f"for the state {state}")

    eia_data_by_month = eia_data.groupby(["Year", "Month"])["month_diff"].mean().reset_index()

    consumption_ts_by_month = consumption_ts.groupby(["Year", "Month"])["Consumption_Factor_Normalizied"].sum().reset_index()

    merged_data = eia_data_by_month.merge(consumption_ts_by_month,
                                          on=["Year", "Month"],
                                          how="outer",
                                          validate='one_to_one')

    correlation = merged_data["month_diff"].corr(merged_data["Consumption_Factor_Normalizied"], method="pearson")

    res = stats.linregress(merged_data["Consumption_Factor_Normalizied"], merged_data["month_diff"])


    logging.info(f"Correlation between consumption factor and EIA data is {correlation}. "
                 f"A correlation near 1 is good and the slope"
                 f"that was calculated is {res.slope}.")

    return res.slope









