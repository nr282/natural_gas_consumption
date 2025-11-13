"""
Inference module takes already trained parameters and weather data
and makes in-sample and out-of-sample predictions.

We need to check against Virginia's true values for 2024.


"""
import os

from models.residential import ResidentialModel, load_residential_data
import pandas as pd
from datetime import datetime
import logging
import json
from utils import get_base_path


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

    TODO: We are almost done with this calculation.

    :return:
    """

    residential_model = ResidentialModel(calibrated_parameters=None,
                                         parameter_list=None)

    state = "Virginia"
    params = load_parameters(state)

    start_datetime = "2022-01-01"
    end_datetime = "2024-12-31"
    eia_start_time = "2022-01-01"
    eia_end_time = "2023-12-31"


    start_training_time = start_datetime
    end_training_time = end_datetime

    data, _, _ = load_residential_data(state,
                                       start_training_time,
                                       end_training_time)


    eia_estimated_daily_observations, estimated_monthly_data, params = residential_model.inference(start_datetime,
                                                                                                   end_datetime,
                                                                                                   eia_start_time,
                                                                                                   eia_end_time,
                                                                                                   params,
                                                                                                   data)

    logging.info(f"EIA estimated daily observations: {estimated_monthly_data}")
    logging.info(f"Parameters are provided by: {params}")


if __name__ == "__main__":
    inference_engine()