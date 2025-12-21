"""
The main module executes the program.

The main module will primarily be run out of main function.
    - In the function, it will first instantiate the dependencies.
    - Then it will place the dependencies in the major loop of the application.
    - It is likely the case that we will need to also institute the config file also.


Other results and research:
1. The goal of the program is to run many states all at once. There are 50 states and there are
three components: (1) Residential, (2) Commercial and (3) Electric Power. There are 150 trainings
that need to occur in the multiprocessing framework. As such, with one training being associated to
one process, we will need to have 150 processes. If an EC2 instance only has one vCPU, then we will have
lots of context switching. As such, the selected EC2 instance must have a high number of vCPUs ideally
which are compute optimized. A good EC2 instance is: (1) c8g.24xlarge, (2) c8g.48xlarge.

These will have about one Virtual Processor for each process, and this will provide us with successful training.

TODO: there are concerns with respect to missing data.

Timeline:
    1. Get Residential working for Virginia. TODO: IN PROGRESS
    2. Train with Virginia Residential data, look at uplift.
    3. Train with more data over a larger period of time.
    4. After training for a week, look at the accuracy.
    5. Train with New York Commercial
    6. Train with California Electric Power Consumption
    7. Document Results
    8. Scaling to all states might be costly. Training might cost 3k.
    9. After building confidence that this material is good, look to scale to all states.
    9. There is confidence that our machine learning technique is better than other weather-eia correlation
    techniques, because of previous training examples.
    10. The statistical model is over parameterizied. A more parameterizied model includes more basic models.
        - To expand this suppose that there are a set of models X, expressed in the code, and there is a set of
        models Y expressed in the code. The set of models Y is a subset of set of models X. More training will be
        required to find a good model X, but it can find the model Y if model Y is good.
    11.

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
from multiprocessing_methods.multiprocessing_framework import init_logs


pd.set_option('display.max_columns', None)


def read_configuration():
    """
    Reads the configuration that will be used later.
    """

    configuration = config.get_configuration()
    return configuration


def residential_training_func(state):
    file_handler, log_handler = init_logs(state, "residential")
    start_training_time = "2020-01-01"
    end_training_time = "2023-12-31"
    start_test_time = "2024-01-01"
    end_test_time = "2025-08-31"
    method = "GLOBAL"
    consumption_factor_method = "POPULATION_WEIGHTED_HDD"
    d = dict()
    d["file_handler"] = file_handler
    d["log_handler"] = log_handler
    fit_residential_model(start_training_time,
                          end_training_time,
                          start_test_time,
                          end_test_time,
                          state,
                          method=method,
                          consumption_factor_method=consumption_factor_method,
                          differencing=True,
                          app_params=d)

def commercial_training_func(state):

    file_handler, log_handler = init_logs(state, "commercial")
    d = dict()
    d["file_handler"] = file_handler
    d["log_handler"] = log_handler
    start_training_time = "2021-01-01"
    end_training_time = "2023-12-31"
    start_test_time = "2024-12-01"
    end_test_time = "2024-12-31"
    method = "GLOBAL"
    consumption_factor_method = "POPULATION_WEIGHTED_HDD"
    fit_commercial_model(start_training_time,
                          end_training_time,
                          start_test_time,
                          end_test_time,
                          state,
                          method=method,
                          consumption_factor_method=consumption_factor_method,
                          app_params=d)


def electric_power_training_func(state):
    file_handler, log_handler = init_logs(state, "electric")
    d=dict()
    d["file_handler"] = file_handler
    d["log_handler"] = log_handler
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
                          consumption_factor_method=consumption_factor_method,
                          app_params=d)

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

    residential_training_func("Virginia")

    """
    parser = argparse.ArgumentParser(description="Parses arguments")
    parser.add_argument("--training", action='store_true', help="States if we want to run training")
    parser.add_argument("--inference", action='store_true', help="States if we want to run inference")
    parser.add_argument("--model_type", help="Either Residential, Commercial or Electric Power", default="Residential")

    args = parser.parse_args()

    if args.training and args.inference:
        raise ValueError("Cannot both run training and inference at the same time")
    
    states = list(us_state_to_abbrev.keys())
    states = ["Alabama", "New York"]
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
    
    
    """


