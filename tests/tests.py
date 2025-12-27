"""
Runs tests to ensure stability of the codebase.

The stability of the codebase is paramount to the success of the projects.

The core elements are:
    1. Unit Tests
    2. Integration Tests
    3. Component Tests

"""


import unittest
from data.weather import test_get_weather
from inference.inference import spectral_inference_engine
import datetime
from data.weather import PrescientWeather
from baseline.baseline import calculate_eia_daily_values, ComponentType
from data.eia_consumption.eia_consumption import read_eia_consumption_data
import json
import pandas as pd
from data.eia_consumption.eia_consumption import get_eia_consumption_data_bulk_df

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


    def test_get_baseline_prediction_system_for_current_date_plus_three_weeks(self):

        current_date = datetime.datetime.now()
        current_date_str = current_date.strftime("%Y-%m-%d")
        future_date_str = (current_date + datetime.timedelta(days=28)).strftime("%Y-%m-%d")
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


if __name__ == '__main__':
    unittest.main()



