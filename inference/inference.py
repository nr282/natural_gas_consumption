"""
Inference module takes already trained parameters and weather data
and makes in-sample and out-of-sample predictions.

We need to check against Virginia's true values for 2024.

TODO: The goal is to figure out if there are repeated misses for monthly values.
TODO: We can check this by running inference for a longer period of time than
TODO: we previously ran it.

"""
import os

from models.residential import ResidentialModel, load_residential_data
import pandas as pd
from datetime import datetime
import logging
import json
from utils import get_base_path
from multiprocessing_methods.multiprocessing_framework import init_logs


logging.basicConfig(level=logging.INFO)

def load_parameters(state):
    """
    Load parameters from the state's parameters json file
    """

    base_path = get_base_path()
    path = os.path.join(base_path, "params", state.lower(), "params.json")
    if not os.path.exists(path):
        raise RuntimeError(f"File has not been found at the path provided by: {path}.")

    with open(path, 'r') as file:
        params = json.load(file)
        params = params["parameters"]
        final_params = {}
        for key, value in params.items():
            final_params[key] = float(value)
    return final_params

def inference_engine():
    """
    Calculates eia_estimated_daily_observations and
    the estimated monthly data. These can then be used in
    comparison to out-of-sample 2024 data.


    :return:
    """

    residential_model = ResidentialModel(calibrated_parameters=None,
                                         parameter_list=None)

    state = "Virginia"
    params = load_parameters(state)

    start_datetime = "2022-01-01"
    end_datetime = "2025-08-31"
    eia_start_time = "2022-01-01"
    eia_end_time = "2023-12-31"


    start_training_time = start_datetime
    end_training_time = end_datetime


    file_handler, log_handler = init_logs(state, "residential")
    app_params = dict()
    app_params["file_handler"] = file_handler
    app_params["log_handler"] = log_handler

    data, _, _ = load_residential_data(state,
                                       start_training_time,
                                       end_training_time,
                                       app_params=app_params)



    eia_estimated_daily_observations, estimated_monthly_data, params = residential_model.inference(start_datetime,
                                                                                                   end_datetime,
                                                                                                   eia_start_time,
                                                                                                   eia_end_time,
                                                                                                   params,
                                                                                                   data,
                                                                                                   app_params=app_params)

    logging.info(f"EIA estimated daily observations: {estimated_monthly_data}")
    logging.info(f"Parameters are provided by: {params}")

    return eia_estimated_daily_observations, estimated_monthly_data, params


def calculate_adjustments(eia_estimated_daily_observations, estimated_monthly_data):
    """
    After observing the fact that inclusion of the month is a good
    predictor for EIA data, I have decided to add this to our inferences.

    Namely, that for Virginia the years 2022 and 2023 monthly data accurately
    predicts 2024 monthly data. It motivates the inclusion of monthly variables
    into the calculation.

    In this function, we aim to calculate a historical monthly adjustment.
    """

    pass


def calculate_adjusted_daily_values(eia_estimated_daily_observations,
                                    adjustments):
    """
    Calculates adjusted daily values with the daily values developed from inference
    engine and monthly adjustments from historical


    """

    pass


def inference_with_monthly_adjustments():

    eia_estimated_daily_observations, estimated_monthly_data = inference_engine()
    adjustments = calculate_adjustments(eia_estimated_daily_observations, estimated_monthly_data)
    eia_estimated_daily_observations_adjusted = calculate_adjusted_daily_values(eia_estimated_daily_observations,
                                                                                adjustments)


    return eia_estimated_daily_observations, estimated_monthly_data, adjustments




if __name__ == "__main__":
    inference_with_monthly_adjustments()