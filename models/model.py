"""
A major goal of the model module is to develop a class that represents a module.

From initial investigation, the parameters and the inference code should live in the same
location.
"""


from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple
from optimization import grid_search
import numpy as np

class Model(ABC):
    """
    States the parameters and the methods that will be housed in the model.

    A key element of this is that:
        1. parameters in the model. The seven parameters that define the model.
        2. functions to do the inference, such as infer_residential.

    """

    def __init__(self, calibrated_parameters, parameter_list):
        self.calibrated_parameters = calibrated_parameters
        self.parameter_list = parameter_list


    @abstractmethod
    def inference(self,
                  start_datetime: datetime,
                  end_datetime: datetime,
                  params: dict,
                  data: dict) -> Tuple[dict, float]:
        pass

    @abstractmethod
    def get_params_for_model(self) -> dict:
        pass

    def calculate_accuracy(self,
                           estimated_monthly_data,
                           data,
                           state):

        eia_data = data["full_eia_data"]
        merged_df = eia_data.merge(estimated_monthly_data, on="Date")

        merged_df["error"] = (merged_df[state].astype(np.float64) - merged_df["eia_observations"]).abs()
        merged_df["relative_error_non_percent"] = merged_df["error"] / merged_df[state].astype(np.float64)
        return float(merged_df["relative_error_non_percent"].mean())


    def run_inference_engine(self,
                             start_datetime: str,
                             end_datetime: str,
                             eia_start_time: str,
                             eia_end_time: str,
                             params: dict,
                             data: dict) -> Tuple[dict, float]:

        #Gather Base Parameters
        base_parameters = self.get_params_for_model()

        #Creates grid from the base parameters
        #Use calibrated parameters where they exist.
        param_to_value = dict()
        for param in base_parameters:
            if param in self.calibrated_parameters:
                param_to_value[param] = self.calibrated_parameters[param]
            else:
                param_to_value[param] = base_parameters.get(param)

        parameter_grid, _ = grid_search.generate_grid(param_to_value)
        optimal_param = None
        optimal_val = None
        relative_error = None
        for param_to_value in parameter_grid():
            estimated_daily_data, estimated_monthly_data, params = self.inference(start_datetime,
                                                                                 end_datetime,
                                                                                 eia_start_time,
                                                                                 eia_end_time,
                                                                                 param_to_value,
                                                                                 data)

            relative_error = self.calculate_accuracy(estimated_monthly_data, data, data["state"])

            import logging
            logging.info(f"The relative error is {relative_error} for params {params}")
            logging.info(f"The estimated monthly data is: {estimated_monthly_data}")
            logging.info(f"The actual data is: {data['full_eia_data']}")

            if optimal_param is None:
                optimal_param = param_to_value
                optimal_val = relative_error
            else:
                if relative_error < optimal_val:
                    optimal_param = param_to_value
                    optimal_val = relative_error

        return optimal_param, relative_error

