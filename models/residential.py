"""
Provides residential model that can be fitted and inferred.

TODO: It would be nice to grab the data only once.
TODO: If we would like to grab the data only once, where should we look to do that.
TODO:

"""

import datetime
from abc import ABC

import pandas as pd
import numpy as np
from calibration.calibration import calibration
from data.consumption_factor.consumption_factor_calculation import calculate_consumption_factor
from optimization import grid_search
import pickle
from data.consumption_factor import consumption_factor_calculation
from data.state_config.virginia.virginia_consumption_factor import VirginiaPopulationData
from data.weather import PyWeatherData
from data.eia_consumption.eia_consumption import get_eia_consumption_data_in_pivot_format
import logging
import pymc as pm
from models.model import Model
from typing import Tuple
import calendar

logging.basicConfig(
    level=logging.DEBUG
)

def map_date_to_index(consumption_factor: pd.DataFrame):
    """
    Map the date to an index and map the index to a date.

    :return:
    """

    consumption_factor_date_to_index = dict()
    index_to_consumption_factor_date = dict()
    for index, row in consumption_factor.iterrows():
        dt = row["Date"]
        consumption_factor_date_to_index[dt] = index
        index_to_consumption_factor_date[index] = dt
    return consumption_factor_date_to_index, index_to_consumption_factor_date, consumption_factor_date_to_index.keys()


class ResidentialModel(Model):
    """
    States the Residential Model that will be used to calculate natural gas
    consumption.

    """

    def __init__(self,
                 calibrated_parameters=None,
                 parameter_list=None,
                 ):

        super().__init__(calibrated_parameters, parameter_list)


    def _calculate_estimated_eia_monthly_data(self, idata):
        """
        Calculates the estimated eia monthly data via the relevant model
        that was sampled.

        :param idata:
        :return:
        """

        eia_observations = idata.posterior.eia_observations
        eia_observations_df = eia_observations.to_dataframe()
        eia_observations_df = eia_observations_df.reset_index()
        eia_observations_by_date = eia_observations_df.groupby(["dates"])["eia_observations"].mean().to_frame()
        eia_observations_by_date["Day"] = eia_observations_by_date.index.to_series().apply(lambda x: x.day)
        eia_observations_by_date["Month"] = eia_observations_by_date.index.to_series().apply(lambda x: x.month)
        eia_observations_by_date["Year"] = eia_observations_by_date.index.to_series().apply(lambda x: x.year)
        eia_observations_by_month = eia_observations_by_date.groupby(["Year", "Month"]).sum().reset_index()
        eia_observations_by_month["Date"] = pd.to_datetime(eia_observations_by_month.apply(lambda row: datetime.datetime(int(row["Year"]), int(row["Month"]), 1), axis=1))
        return eia_observations_by_date, eia_observations_by_month

    def inference(self,
                start_datetime: str,
                end_datetime: str,
                eia_start_datetime: str,
                eia_end_datetime: str,
                params: dict,
                data: dict):
        """
        Inference in the residential model.

        """

        dates = pd.date_range(start_datetime, end_datetime)
        consumption_factor_values = data["consumption_factor_values"]["Consumption_Factor_Normalizied"].values
        consumption_factor_lagged_values = calculate_consumption_lagged(consumption_factor_values)
        mean_consumption_factor = np.mean(consumption_factor_values)
        variance_consumption_factor = np.var(consumption_factor_values)
        eia_monthly_values = data["eia_monthly_values"]
        full_eia_data = data["full_eia_data"]

        coords = {
            "dates": list(dates),
        }
        
        with pm.Model(coords=coords) as model:
            consumption_factor = pm.Normal("consumption_factor",
                                           mu=mean_consumption_factor,
                                           sigma=variance_consumption_factor,
                                           observed=consumption_factor_values.astype(np.float32),
                                           dims="dates")

            consumption_factor_lagged = pm.Normal("consumption_factor_lagged",
                                                  mu=mean_consumption_factor,
                                                  sigma=variance_consumption_factor,
                                                  observed=consumption_factor_lagged_values.astype(np.float32),
                                                  dims="dates")

            logging.debug("Parameters are provided by: {params}".format(params=params))

            alpha = pm.Normal("alpha_1",
                              mu=float(params.get("alpha_mu")),
                              sigma=float(params.get("alpha_sigma")))

            alpha_2 = pm.Normal("alpha_2", mu=float(params.get("alpha_2_mu")), sigma=float(params.get("alpha_2_sigma")))

            minimum_consumption = pm.Normal("minimum_consumption",
                                            mu=params.get("minimum_consumption_mu"),
                                            sigma=params.get("minimum_consumption_sig"))


            eia_daily_observations = pm.Normal("eia_observations",
                                               mu=(alpha + alpha_2) * consumption_factor + alpha_2 * consumption_factor_lagged + minimum_consumption,
                                               sigma=params.get("daily_consumption_error"),
                                               dims="dates")

            consumption_factor_to_index, index_to_consumption_factor_date, dates = map_date_to_index(data["consumption_factor_values"])

            calculate_eia_monthly_consumption_constraints(model,
                                                          eia_daily_observations,
                                                          full_eia_data,
                                                          consumption_factor_to_index,
                                                          state,
                                                          eia_monthly_start_date=eia_start_datetime,
                                                          eia_monthly_end_date=eia_end_datetime,
                                                          sigma=params.get("monthly_consumption_error"))

            idata = pm.sample(draws=20, tune=20)
            eia_estimated_daily_observations, estimated_estimated_monthly_data = self._calculate_estimated_eia_monthly_data(idata)
            return eia_estimated_daily_observations, estimated_estimated_monthly_data, params

    def get_params_for_model(self) -> dict:
        """
        Calculates parameters for the model.

        The parameters that will be used in the model are:
            1. alpha_mu
            2. alpha_sigma
            3. alpha_2_mu
            4. alpha_2_sigma
            5. minimum_consumption_mu
            6. minimum_consumption_sig
            7. daily_consumption_error

        """

        return {"alpha_mu": 0,
                "alpha_sigma": 10,
                "alpha_2_mu": 0,
                "alpha_2_sigma": 1,
                "minimum_consumption_mu": 0,
                "minimum_consumption_sig": 0.1,
                "daily_consumption_error": 0,
                "monthly_consumption_error": 0.0}



def calculate_consumption_lagged(consumption_factor_values):
    """
    Calculate consumption factor lagged values.
    """

    consumption_factor_values_lagged = np.roll(consumption_factor_values, shift=1)
    val = consumption_factor_values_lagged[1]
    consumption_factor_values_lagged[0] = val
    return consumption_factor_values_lagged

def is_data_between_dates(eia_monthly_start_date, eia_monthly_end_date, current_month):

    return (eia_monthly_start_date <= current_month) and (current_month <= eia_monthly_end_date)

def calculate_eia_monthly_consumption_constraints(model,
                                                  eia_daily_observations,
                                                  eia_monthly_data,
                                                  consumption_factor_to_index,
                                                  state: str,
                                                  eia_monthly_start_date="2022-01-01",
                                                  eia_monthly_end_date="2024-01-01",
                                                  sigma=10):
    """
    Calculate eia monthly consumption constraints.
    """


    print("Calculating EIA Monthly Constraints...")

    eia_monthly_start_date = datetime.datetime.strptime(eia_monthly_start_date, "%Y-%m-%d")
    eia_monthly_end_date = datetime.datetime.strptime(eia_monthly_end_date, "%Y-%m-%d")
    eia_monthly_data["Date"] = pd.to_datetime(
        eia_monthly_data.index.to_series().apply(lambda x: datetime.datetime.strptime(x + "-01","%Y-%m-%d")))

    constraint_random_variables = dict()
    for index, row in eia_monthly_data.iterrows():
        start_month_dt = row["Date"]
        start_date_str = start_month_dt.strftime("%Y-%m-%d")
        year = start_month_dt.year
        month = start_month_dt.month
        day = start_month_dt.day

        if is_data_between_dates(eia_monthly_start_date, eia_monthly_end_date, start_month_dt):

            print(f"Applying Constraint for {start_date_str} with index {index}")
            monthly_value = float(row[state])
            day_of_week, end_of_month_day_number = calendar.monthrange(year, month)
            end_of_month_datetime = datetime.datetime(year, month, end_of_month_day_number)
            indicies = []
            for date in consumption_factor_to_index:
                if (start_month_dt <= date) and (date <= end_of_month_datetime):
                    indicies.append(consumption_factor_to_index[date])

            constraint_random_variable = pm.Normal(f"month_{start_date_str}",
                                                   mu=sum([eia_daily_observations[index] for index in indicies]),
                                                   sigma=sigma,
                                                   observed=monthly_value)

            constraint_random_variables[start_date_str] = constraint_random_variable
    return constraint_random_variables

def save_parameters(accuracy_result):
    pickle.dump(accuracy_result, open("accuracy_result.pkl", "wb"))


def get_population(state: str):

    if state == "Virginia":
        return VirginiaPopulationData()
    else:
        raise NotImplementedError("State {state} not implemented.".format(state=state))

def get_eia_residential_data(start_date: datetime.date, end_date: datetime.date):
    """
    Gets the EIA Residential data between the start_date and end_date.

    :return:
    """

    residential_df = get_eia_consumption_data_in_pivot_format(start_date=start_date,
                                                              end_date=end_date,
                                                              canonical_component_name="Residential")

    return residential_df


def fit_residential_model(start_training_time: str,
                          end_training_time: str,
                          eia_start_time: str,
                          eia_end_time: str,
                          state: str):
    """
    Fits the residential model.

    https://en.wikipedia.org/wiki/List_of_cities_and_counties_in_Virginia

    :return:
    """

    #How can we look at correctly formatting the code.

    logging.info("Acquiring EIA Residential Data")
    eia_data = get_eia_residential_data(start_training_time, end_training_time)
    eia_data = eia_data[[state]]
    logging.info(f"Finished EIA Residential Data. Some EIA Data is provided as: {eia_data.head()}")

    population = get_population(state)
    weather_service = PyWeatherData(population)
    consumption_factor = calculate_consumption_factor(population,
                                                     weather_service,
                                                     start_training_time,
                                                     end_training_time)

    calibrated_parameters = calibration(consumption_factor,
                                        eia_data,
                                        state)

    #TODO: This is a check.
    calibrated_parameters["daily_consumption_error"] = 0.03


    data = dict()
    data["eia_monthly_values"] = eia_data
    data["consumption_factor_values"] = consumption_factor
    data["full_eia_data"] = eia_data
    data["state"] = state

    params = dict()
    params["alpha_mu"] = 0.0
    params["alpha_2_mu"] = 0.0
    params["alpha_sigma"] = 0.0
    params["minimum_consumption_mu"] = 0.0
    params["minimum_consumption_sig"] = 0.0
    params["daily_consumption_error"] = 0.0

    best_parameters, optimal_rel_error = ResidentialModel(calibrated_parameters, params).run_inference_engine(start_training_time,
                                                                                                            end_training_time,
                                                                                                            eia_start_time,
                                                                                                            eia_end_time,
                                                                                                            params,
                                                                                                            data)


    logging.info("Parameters are provided by {params} ".format(params=best_parameters))
    logging.info(f"Relative Error {optimal_rel_error}".format(val=optimal_rel_error))




if __name__ == '__main__':

    #Run the fitting of the residential model for Virginia.

    start_training_time = "2022-01-01"
    end_training_time = "2024-01-01"
    start_test_time = "2022-01-01"
    end_test_time = "2023-01-01"
    state = "Virginia"

    fit_residential_model(start_training_time,
                          end_training_time,
                          start_test_time,
                          end_test_time,
                          state)



