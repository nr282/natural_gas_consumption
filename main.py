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

def read_configuration():
    """
    Reads the configuration that will be used later.

    """


    configuration = config.get_configuration()
    return configuration


def main():


    config = read_configuration()
    datasets = data_preparation.gather_relevant_data(config)





if __name__ == "__main__":
    main()