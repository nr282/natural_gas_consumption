"""
Runs tests to ensure stability of the codebase.

The stability of the codebase is paramount to the success of the projects.

The core elements are:
    1. Unit Tests
    2. Integration Tests
    3. Component Tests

"""


import unittest

import date_utils.date_utils
from data.weather import test_get_weather
from inference.inference import spectral_inference_engine
import datetime
from data.weather import PrescientWeather
from baseline.baseline import calculate_eia_daily_values, ComponentType
from data.eia_consumption.eia_consumption import read_eia_consumption_data
import json
import pandas as pd
import numpy as np
from data.eia_consumption.eia_consumption import get_eia_consumption_data_bulk_df
from testing_utility import check_near_consumption_value


from models.seasonality.seasonality import (
    get_day_of_year, 
    get_cosine_1_year_period_value, 
    get_cosine_6_month_period_value,
    get_time_series_1,
    get_time_series_2,
    calculate_climatology
)

class TestResidentialConsumption(unittest.TestCase):


    def test_run_inference(self):

        start_time = "2023-01-01"
        end_time = "2024-12-31"
        state = "Virginia"
        component_type = "residential"
        spectral_inference_engine(state,
                                  start_time,
                                  end_time,
                                  component_type,
                                  args={})


    def test_get_weather(self):

        state = "Virginia"
        prescient_weather_data = PrescientWeather([state])
        data = prescient_weather_data.get_standardizied_data()

    def test_get_baseline_prediction_system(self):

        daily_values = calculate_eia_daily_values("2023-01-01",
                                                  "2024-12-31",
                                                  "2024-01-01",
                                                  "2024-12-01",
                                                  "2010-01-01",
                                                  "2022-12-01",
                                                  "2025-12-26",
                                                  "2020-01-01",
                                                  ComponentType.ELECTRIC,
                                                  "South Carolina")

    def test_get_baseline_prediction_system_for_current_date(self):

        current_date = datetime.datetime.now()
        current_date_str = current_date.strftime("%Y-%m-%d")
        future_date_str = (current_date + datetime.timedelta(days=14)).strftime("%Y-%m-%d")
        daily_values = calculate_eia_daily_values("2023-01-01",
                                                  future_date_str,
                                                  "2024-01-01",
                                                  "2024-12-01",
                                                  "2018-01-01",
                                                  "2022-12-01",
                                                  current_date_str,
                                                  "2020-01-01",
                                                  ComponentType.ELECTRIC,
                                                  "Virginia")

        self.assertTrue(len(daily_values) > 0)



    def test_prescient_weather_cdd(self):

        state = "Virginia"
        start_date = "2023-01-01"
        end_date = "2027-12-31"
        prescient_weather = PrescientWeather([state])
        weather = prescient_weather.get_cdd([state], start_date, end_date)
        self.assertFalse(weather.isnull().any().any())


    def test_prescient_weather_hdd(self):

        state = "Virginia"
        start_date = "2023-01-01"
        end_date = "2027-12-31"
        prescient_weather = PrescientWeather([state])
        weather = prescient_weather.get_hdd([state], start_date, end_date)
        self.assertFalse(weather["HDD"].isnull().any())


    def test_get_baseline_prediction_system_for_current_date_plus_three_weeks_Virginia(self):
        state = 'Virginia'
        current_date = datetime.datetime.now()
        current_date_str = current_date.strftime("%Y-%m-%d")
        future_date_str = (current_date + datetime.timedelta(days=21)).strftime("%Y-%m-%d")
        future_date_end_month_str = date_utils.date_utils.get_last_date_of_month(future_date_str)
        daily_values, diff_pct_error = calculate_eia_daily_values("2023-01-01",
                                                            future_date_end_month_str,
                                                  "2024-01-01",
                                                  "2024-12-31",
                                                  "2018-01-01",
                                                  "2022-12-31",
                                                            current_date_str,
                                                  "2020-01-01",
                                                            ComponentType.ELECTRIC,
                                                                  state)

        self.assertLess(diff_pct_error, 100.0)
        self.assertTrue("Date" in daily_values.columns)
        self.assertFalse(daily_values["Value"].isna().any())

        pct_close = check_near_consumption_value(state,
                                                daily_values,
                                                ComponentType.ELECTRIC)

        self.assertGreater(pct_close, 80.0)


    def test_get_baseline_prediction_system_for_current_date_plus_three_weeks_New_York(self):
        state = 'New York'
        current_date = datetime.datetime.now()
        current_date_str = current_date.strftime("%Y-%m-%d")
        future_date_str = (current_date + datetime.timedelta(days=21)).strftime("%Y-%m-%d")
        future_date_end_month_str = date_utils.date_utils.get_last_date_of_month(future_date_str)
        daily_values, diff_pct_error = calculate_eia_daily_values("2023-01-01",
                                                            future_date_end_month_str,
                                                  "2024-01-01",
                                                  "2024-12-31",
                                                  "2018-01-01",
                                                  "2022-12-31",
                                                            current_date_str,
                                                  "2020-01-01",
                                                            ComponentType.ELECTRIC,
                                                                  state)

        self.assertLess(diff_pct_error, 100.0)
        self.assertTrue("Date" in daily_values.columns)
        self.assertFalse(daily_values["Value"].isna().any())

        pct_close = check_near_consumption_value(state,
                                                daily_values,
                                                ComponentType.ELECTRIC)

        self.assertGreater(pct_close, 80.0)




    def test_eia_consumption_data_in_pivot_format(self):
        from data.eia_consumption.eia_consumption import get_eia_consumption_data_in_pivot_format
        df = get_eia_consumption_data_in_pivot_format(start_date="2023-01-01",
                                                    end_date="2024-12-31",
                                                    canonical_component_name="Residential")


    def test_eia_consumption_data_in_bulk_format_short_period(self):
        """
        For this call, I get more than 95 percent of the data points back
        in their proper format.
        """


        state = "New York"
        start_date_str = "2023-01-01"
        end_date_str = "2025-01-01"
        df = get_eia_consumption_data_bulk_df(start_date_str,
                                                end_date_str,
                                                create_new_data = True)

        dates = pd.date_range(start_date_str, end_date_str, freq="MS")
        virginia_df = df[df["standard_state_name"] == state]
        virginia_periods = set(virginia_df["period"].unique())
        candidate_periods = set([f"{date.year}-{str(date.month).zfill(2)}" for date in dates])
        pct = len(virginia_periods) / len(candidate_periods) * 100.0
        missing_periods = candidate_periods - virginia_periods
        self.assertGreater(pct, 95.0, f"Missing periods: {missing_periods}")

    def test_eia_consumption_data_in_bulk_format_long_period(self):
        """
        In the extended form, only around 30 percent of the data points are provided back.

        :return:
        """

        state = "New York"
        start_date_str = "2010-01-01"
        end_date_str = "2025-01-01"

        df = get_eia_consumption_data_bulk_df(start_date_str,
                                              end_date_str,
                                              create_new_data=True)

        dates = pd.date_range(start_date_str, end_date_str, freq="MS")
        virginia_df = df[df["standard_state_name"] == state]
        virginia_periods = set(virginia_df["period"].unique())
        candidate_periods = set([f"{date.year}-{str(date.month).zfill(2)}" for date in dates])
        pct = len(virginia_periods) / len(candidate_periods) * 100.0
        missing_periods = candidate_periods - virginia_periods
        self.assertGreater(pct, 30, f"Missing periods: {missing_periods}")


class TestSeasonality(unittest.TestCase):
    """
    Tests for the seasonality module functions that handle seasonal time series calculations.
    """
    
    def test_get_day_of_year(self):
        """Test the get_day_of_year function with different input formats."""
        # Test with string input
        self.assertEqual(get_day_of_year("2023-01-01"), 1)
        self.assertEqual(get_day_of_year("2023-12-31"), 365)
        
        # Test with datetime object input
        date_obj = datetime.datetime(2023, 2, 15)
        self.assertEqual(get_day_of_year(date_obj), 46)
        
        # Test with leap year
        self.assertEqual(get_day_of_year("2024-12-31"), 366)
        self.assertEqual(get_day_of_year("2024-02-29"), 60)

if __name__ == '__main__':
    unittest.main()



