"""
The main module executes the program.

The main module will primarily be run out of main function.
    - In the function, it will first instantiate the dependencies.
    - Then it will place the dependencies in the major loop of the application.
    - It is likely the case that we will need to also institute the config file also.


How are we going to handle configuration.
    - Should we institute the configruation on a per-state basis
    -

"""

import pandas as pd
from config import config
from data import data_preparation
from models.residential import fit_residential_model
import argparse
from inference.inference import inference_engine

def read_configuration():
    """
    Reads the configuration that will be used later.

    """


    configuration = config.get_configuration()
    return configuration


def main():


    # Run the fitting of the residential model for Virginia.
    # TODO: Currently running with the global optimization.

    start_training_time = "2022-01-01"
    end_training_time = "2023-12-31"
    start_test_time = "2022-01-01"
    end_test_time = "2023-01-01"
    state = "Virginia"
    method = "LINEAR"

    fit_residential_model(start_training_time,
                          end_training_time,
                          start_test_time,
                          end_test_time,
                          state,
                          method=method)

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

    args = parser.parse_args()

    if args.training and args.inference:
        raise ValueError("Cannot both run training and inference at the same time")


    if args.training:
        main()
    elif args.inference:
        run_inference()



