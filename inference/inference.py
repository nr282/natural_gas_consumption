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

def load_parameters(state, consumption_type):
    """
    Load parameters from the state's parameters json file
    """

    base_path = get_base_path()
    path = os.path.join(base_path, "params", state.lower(), consumption_type + "_params.json")
    if not os.path.exists(path):
        raise RuntimeError(f"File has not been found at the path provided by: {path}.")

    with open(path, 'r') as file:
        params = json.load(file)
        params = params["parameters"]
        final_params = {}
        for key, value in params.items():
            final_params[key] = float(value)
    return final_params


def example_of_spectral_inference_engine():


    start_time = "2024-01-01"
    end_time = "2024-12-31"
    state = "Virginia"
    component_type = "residential"
    spectral_inference_engine(state,
                              start_time,
                              end_time,
                              component_type,
                              args={})

def spectral_inference_engine(state,
                              start_datetime,
                              end_datetime,
                              component_type,
                              args=None,
                              ):
    """
    Calculates the daily EIA values using the statistical techniques for a
    set of parameters. This will run Spectral's methodology.
    This is the inference stage, so the results provided back should
    not take a long time to run:

        1. start_date: datetime.datetime
        2. end_date: datetime.datetime
        3. state: str
        4. methodology: str

    The inference engine should use the information that has been carefully calculated on
    AWS in the more compute intensive stage.

    It should also be able to take in modified parameters to allow for a play-ground type behavior.

    :return:
    """

    consumption_type = "residential"
    residential_model = ResidentialModel(calibrated_parameters=None,
                                         parameter_list=None)

    params = load_parameters(state, consumption_type)
    file_handler, log_handler = init_logs(state, consumption_type)
    app_params = dict()
    app_params["file_handler"] = file_handler
    app_params["log_handler"] = log_handler

    data, _, _ = load_residential_data(state,
                                       start_datetime,
                                       end_datetime,
                                       app_params=app_params,
                                       differencing=True)


    #eia_estimated_daily_observations represents the differences from the average value.
    #eia_estimated_daily_observations is based on (a) training parameters and (b) consumption factor values
    #that have been estimated.
    eia_estimated_daily_observations, estimated_monthly_data, params = residential_model.inference(start_datetime,
                                                                                                   end_datetime,
                                                                                                   start_datetime,
                                                                                                   end_datetime,
                                                                                                   params,
                                                                                                   data,
                                                                                                   app_params=app_params)

    data, _, _ = load_residential_data(state,
                                       start_datetime,
                                       end_datetime,
                                       app_params=app_params,
                                       differencing=False)


    #This adds together (a) eia_estimated_daily_observations to the
    #eia daily values discovered by looking at averages.
    daily_values = residential_model.inference_for_daily_values(start_datetime,
                                                                end_datetime,
                                                                data,
                                                                eia_estimated_daily_observations,
                                                                app_params=app_params)



    logging.info(f"EIA estimated daily observations: {estimated_monthly_data}")
    logging.info(f"Parameters are provided by: {params}")

    return daily_values




if __name__ == "__main__":
    example_of_spectral_inference_engine()