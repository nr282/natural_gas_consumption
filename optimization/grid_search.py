"""
Generates the grid that will be used for the search.

A key goal will be to generate the grid that will be used in optimization.


"""

from typing import List
import numpy as np
from itertools import product

def generate_grid(params_to_val: dict) -> List[dict]:
    """
    Generates a list of parameters.

    """

    n = len(params_to_val)
    grid_coordinates = []
    k = 10
    param_names = []
    for i, param in enumerate(params_to_val):
        val = params_to_val[param]
        lower_val = 0.1 * val
        upper_val = 10 * val
        x = np.linspace(lower_val, upper_val, k)
        grid_coordinates.append(x)
        param_names.append(param)

    cartesian_product_iterator = product(*grid_coordinates)

    def param_generator():

        while True:
            try:
                next(cartesian_product_iterator)
            except StopIteration:
                return
            param_values = next(cartesian_product_iterator)
            res = dict()
            for index, val in enumerate(param_values):
                param = param_names[index]
                res[param] = val
            yield res

    return param_generator, cartesian_product_iterator


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    params_to_val = {"alpha": 0.1, "beta": 0.1}
    grid, cartesian_product = generate_grid(params_to_val)
    for elem in grid():
        logging.info(elem)
