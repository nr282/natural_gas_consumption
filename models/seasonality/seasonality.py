"""
The major goal of the module is to address seasonality.

Seasonality will be addressed via adding of a seasonal time series.

"""

import numpy as np
import pandas as pd
import datetime

def get_day_of_year(date_str: str):
    if isinstance(date_str, str):
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    elif isinstance(date_str, datetime.datetime):
        date_obj = date_str
    else:
        date_obj = date_str
    day_of_year = date_obj.timetuple().tm_yday
    return day_of_year

def get_cosine_1_year_period_value(day_of_year):

    return np.cos(2 * np.pi * day_of_year / 365)

def get_cosine_6_month_period_value(day_of_year):

    return np.cos(2 * np.pi * day_of_year / 182)

def get_time_series_1(dates):

    time_vals = list(map(lambda x: get_cosine_1_year_period_value(get_day_of_year(x)), dates))
    time_val = np.array(time_vals)
    return time_vals


def get_time_series_2(dates):

    time_vals = list(map(lambda x: get_cosine_6_month_period_value(get_day_of_year(x)), dates))
    time_val = np.array(time_vals)
    return time_vals

def example():
    from datetime import datetime

    # Example date: February 7, 2023
    year, month, day = 2023, 2, 7
    date_obj = datetime(year, month, day)
    day_of_year = date_obj.timetuple().tm_yday

    # Get the day of the year
    print(f"The day of the year for {date_obj.strftime('%Y-%m-%d')} is: {day_of_year}")


if __name__ == "__main__":
    example()