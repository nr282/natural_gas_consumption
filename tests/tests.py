"""
Runs tests to ensure stability of the codebase.

The stability of the codebase is paramount to the success of the projects.

The core elements are:
    1. Unit Tests
    2. Integration Tests
    3. Component Tests






"""
import logging
import time
import unittest

import date_utils.date_utils
from data.weather import (test_get_weather,
                          upload_weather_df_to_s3_bucket,
                          gather_weather_data,
                          download_dataframe_from_s3_bucket)

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
from deployment.spectral_api import parse_lambda_event
import requests
from deployment.external.api_call import call_predict_gas_api, test_api_gateway, get_request_with_authentication
from data.eia_consumption.eia_consumption import get_eia_consumption_data_in_pivot_format
from dateutil.relativedelta import relativedelta


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

    def test_prescient_weather_cdd(self):

        state = "Virginia"
        start_date = "2023-01-01"
        end_date = "2027-12-31"
        current_date = "2025-12-31"
        prescient_weather = PrescientWeather([state])
        weather = prescient_weather.get_cdd([state], start_date, end_date, current_date)

        self.assertFalse(weather.isnull().any().any())


    def test_prescient_weather_hdd(self):

        state = "Virginia"
        start_date = "2023-01-01"
        end_date = "2027-12-31"
        current_date = "2025-12-31"
        prescient_weather = PrescientWeather([state])
        weather_previous_day = prescient_weather.get_hdd([state], start_date, end_date, current_date)
        self.assertFalse(weather_previous_day["HDD"].isnull().any())

        state = "Virginia"
        start_date = "2023-01-01"
        end_date = "2027-12-31"
        current_date = "2026-01-01"
        prescient_weather = PrescientWeather([state])
        weather_current_day = prescient_weather.get_hdd([state], start_date, end_date, current_date)
        self.assertFalse(weather_current_day["HDD"].isnull().any())

        hdd_previous_day = weather_previous_day[weather_previous_day["Date"] == current_date]["HDD"].iloc[0]
        hdd_current_day = weather_current_day[weather_current_day["Date"] == current_date]["HDD"].iloc[0]

        self.assertNotEqual(hdd_current_day, hdd_previous_day)



    def test_get_baseline_prediction_system_for_current_date_plus_three_weeks_for_states(self):
        """
        Function aims to check the calculate eia daily values for a large set of states.

        :return:
        """

        states = ['Virginia']
        for state in states:
            component_type = ComponentType.RESIDENTIAL
            current_date = datetime.datetime.now()
            current_date_str = current_date.strftime("%Y-%m-%d")
            future_date_str = (current_date + datetime.timedelta(days=21)).strftime("%Y-%m-%d")
            future_date_end_month_str = date_utils.date_utils.get_last_date_of_month(future_date_str)
            daily_values, diff_pct_error = calculate_eia_daily_values("2023-01-01",
                                                                future_date_end_month_str,
                                                                "2023-01-01",
                                                                "2025-10-31",
                                                                "2015-01-01",
                                                                "2022-12-31",
                                                                current_date_str,
                                                                component_type,
                                                                      state)

            logging.info(f"diff_pct_error is provided by: {diff_pct_error}")

            pct_close = check_near_consumption_value(state,
                                                    daily_values,
                                                    component_type,
                                                    number_to_check=0)

            self.assertGreater(pct_close, 80.0)
            self.assertTrue(len(daily_values[daily_values["Value"].isna()]) == 0)
            self.assertLess(diff_pct_error, 150) #This is not good, might as well guess the average.
            self.assertTrue("Date" in daily_values.columns)
            self.assertFalse(daily_values["Value"].isna().any())

    def test_spectral_api(self):
        event = {"headers": {"start_date": "2021-01-01",
                              "end_date": "2027-12-31",
                              "state": "Virginia",
                              "component_type": "residential"}}
        context = {}
        lambda_result = parse_lambda_event(event, context)
        self.assertTrue(lambda_result["statusCode"] == 200)

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


    def test_unauthorizied_predict_gas_api(self):

        request_string, headers = get_request_with_authentication()
        headers["x-api-key"] = ""
        result, status_code = test_api_gateway(request_string, headers)
        self.assertEqual(status_code, 401)

    def test_authorizied_predict_gas_api(self):
        start = time.time()
        request_string, headers = get_request_with_authentication()
        result, status_code = test_api_gateway(request_string, headers)
        end = time.time()
        print(f"Time taken to call API Gateway: {end - start}")
        self.assertEqual(status_code, "Success")
        self.assertTrue(len(result) > 0)

    def test_authorizied_predict_gas_api_with_live_session(self):
        """
        Goal with this test is to assess latency from different regions.

        A link to this is provided here:
            1. https://www.cloudping.co/

        :return:
        """


        import logging
        logging.basicConfig(level=logging.DEBUG)
        s = requests.Session()
        request_string, headers = get_request_with_authentication()
        result, status_code = test_api_gateway(request_string, headers)
        for i in range(20):
            start = time.time()
            s.get(request_string, headers=headers)
            end = time.time()
            print(f"Time taken to call API Gateway: {end - start}")
        self.assertEqual(status_code, 200)

    def test_concurrent_request_execution(self):
        from requests_futures.sessions import FuturesSession
        import logging
        logging.basicConfig(level=logging.DEBUG)
        session = FuturesSession()
        request_string, headers = get_request_with_authentication()
        results = []
        for i in range(20):
            start = time.time()
            future_result = session.get(request_string, headers=headers)
            results.append(future_result)
            end = time.time()
            print(f"Time taken to call API Gateway: {end - start}")

        for future_result in results:
            response = future_result.result()
            result, status_code = test_api_gateway(response.text, headers, result=response)
            self.assertEqual(status_code, "Success")

    def test_s3_bucket_upload(self):
        """
        Test aims to see if the s3 bucket upload is proper.

        Goal is to keep the latency under 10 seconds.

        :return:
        """


        start_date = "1980-01-01"
        current_date = datetime.datetime.now() - relativedelta(months=3)
        end_date = current_date.strftime("%Y-%m-%d")

        create_new_data = True
        df = get_eia_consumption_data_in_pivot_format(start_date=start_date,
                                                 end_date=end_date,
                                                 create_new_data = create_new_data)

        start_time = time.time()
        create_new_data = False
        df = get_eia_consumption_data_in_pivot_format(start_date=start_date,
                                                      end_date=end_date,
                                                      create_new_data=create_new_data)
        end_time = time.time()
        time_elapsed = end_time - start_time
        self.assertLess(time_elapsed, 2)


    def test_s3_bucket_download(self):
        import logging
        logging.basicConfig(level=logging.INFO)

        start_date = "1980-01-01"
        current_date = datetime.datetime.now() - relativedelta(months=3)
        end_date = current_date.strftime("%Y-%m-%d")


        start_time = time.time()
        create_new_data = False
        df = get_eia_consumption_data_in_pivot_format(start_date=start_date,
                                                      end_date=end_date,
                                                      create_new_data=create_new_data)
        end_time = time.time()
        time_elapsed = end_time - start_time
        self.assertLess(time_elapsed, 1.2)


    def test_s3_bucket_upload_for_weather(self):

        df = gather_weather_data()
        upload_weather_df_to_s3_bucket(df)

    def test_s3_bucket_download_for_weather(self):

        df = download_dataframe_from_s3_bucket()



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



