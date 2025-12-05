"""
Multiprocessing helps parallelize training various states at the same time.

This parallelization is critical so that we can keep all of our code on just one EC2 instance
and properly use all of the processors on that EC2 instance. Each processor has multiple cores.
We will want to use all of these cores on each processor.

The plan here is:
    1. Run a simple version of the program.
        - This will prove out and provide a framework for the rest of the program.
    2. Likewise, we will develop a function which will multiprocess a set of states.
    3. Likewise, from there, we will develop a function that runs all of the states.


"""

import multiprocessing
from multiprocessing import Pool
import logging
import time


def worker_function(name):
    print(f"Worker process {name} is running.")


def get_path_to_logs():
    pass

def init_logs(state: str, model_type: str = "residential"):


    file_name = f'output_{state}_{model_type}.log'

    # Create a logger
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)

    # Create a FileHandler
    file_handler = logging.FileHandler(file_name)
    logger.addHandler(file_handler)

    # Log some messages
    logger.info("Initialization of the logs...")

    # Flush the file handler
    file_handler.flush()
    time.sleep(0.5)

    return file_handler, logger


def square(log_id: int):

    init_logs(log_id)
    s = 0
    for i in range(100000):
        s += i
    return s

def multiprocessing_example():
    """
    Multiprocessing example.

    """

    with Pool(processes=4) as pool:  # Create a pool with 4 worker processes
        results = pool.map(square, [1, 2, 3, 4, 5])
        print(results)


def test_1():
    multiprocessing_example()

def run_states_in_parallel(states, training_func):

    s_n = len(states)
    with Pool(processes=s_n) as pool:  # Create a pool with 50 worker processes
        results = pool.map(training_func, states)

    if all(results):
        logging.info("All states completed successfully.")
    else:
        logging.info("One or more states failed.")


if __name__ == "__main__":


    multiprocessing_example()
