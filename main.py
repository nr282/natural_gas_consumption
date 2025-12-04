"""
The main module executes the program.

The main module will primarily be run out of main function.
    - In the function, it will first instantiate the dependencies.
    - Then it will place the dependencies in the major loop of the application.
    - It is likely the case that we will need to also institute the config file also.


How are we going to handle configuration.
    - Should we institute the configruation on a per-state basis
    -

TODO: We should be able to run this in the could in short order.
TODO: We should be able to run this in the cloud as early as 12/2/2025.
    TODO: -Need to update the logging so that the output has a unique log indexed by state and by model type.

"""

import pandas as pd
from config import config
from data import data_preparation
from models.residential import fit_residential_model
from models.commercial import fit_commercial_model
from models.electric_power_consumption import fit_electric_power_model
import argparse
from inference.inference import inference_engine
from multiprocessing_methods import multiprocessing_framework
import logging
from typing import List
from data.eia_consumption.eia_geography_mappings import us_state_to_abbrev

def read_configuration():
    """
    Reads the configuration that will be used later.

    """


    configuration = config.get_configuration()
    return configuration


def residential_training_func(state):

    start_training_time = "2023-01-01"
    end_training_time = "2023-12-31"
    start_test_time = "2024-01-01"
    end_test_time = "2024-12-31"
    method = "GLOBAL"
    consumption_factor_method = "POPULATION_WEIGHTED_HDD"
    fit_residential_model(start_training_time,
                          end_training_time,
                          start_test_time,
                          end_test_time,
                          state,
                          method=method,
                          consumption_factor_method=consumption_factor_method)

def commercial_training_func(state):

    start_training_time = "2023-01-01"
    end_training_time = "2023-12-31"
    start_test_time = "2024-01-01"
    end_test_time = "2024-12-31"
    method = "GLOBAL"
    consumption_factor_method = "POPULATION_WEIGHTED_HDD"
    fit_commercial_model(start_training_time,
                          end_training_time,
                          start_test_time,
                          end_test_time,
                          state,
                          method=method,
                          consumption_factor_method=consumption_factor_method)


def electric_power_training_func(state):

    start_training_time = "2023-01-01"
    end_training_time = "2023-12-31"
    start_test_time = "2024-01-01"
    end_test_time = "2024-12-31"
    method = "GLOBAL"
    consumption_factor_method = "POPULATION_WEIGHTED_CDD"
    fit_electric_power_model(start_training_time,
                          end_training_time,
                          start_test_time,
                          end_test_time,
                          state,
                          method=method,
                          consumption_factor_method=consumption_factor_method)

def run_multiprocessing_over_states_for_residential(states: List[str]):
    multiprocessing_framework.run_states_in_parallel(states, residential_training_func)

def run_multiprocessing_over_states_for_commercial(states: List[str]):
    multiprocessing_framework.run_states_in_parallel(states, commercial_training_func)

def run_multiprocessing_over_states_for_electric_power(states: List[str]):
    multiprocessing_framework.run_states_in_parallel(states, electric_power_training_func)


def main():
    electric_power_training_func("New York")


def run_inference():
    """
    Run Inference.

    :return:
    """

    inference_engine()


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Parses arguments")
    parser.add_argument("--training", action='store_true', help="States if we want to run training")
    parser.add_argument("--inference", action='store_true', help="States if we want to run inference")
    parser.add_argument("--model_type", help="Either Residential, Commercial or Electric Power", default="Residential")

    args = parser.parse_args()

    if args.training and args.inference:
        raise ValueError("Cannot both run training and inference at the same time")
    
    states = list(us_state_to_abbrev.keys())

    if args.training:
        if args.model_type == "Residential":
            run_multiprocessing_over_states_for_residential(states)
        elif args.model_type == "Commercial":
            run_multiprocessing_over_states_for_commercial(states)
        elif args.model_type == "Electric Power":
            run_multiprocessing_over_states_for_electric_power(states)
        else:
            raise ValueError(f"Model type provided by: {args.model_type} is not supported.")
    elif args.inference:
        run_inference()


