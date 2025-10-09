"""
Provides residential model that can be fitted and inferred.

TODO: It would be nice to grab the data only once.
TODO: If we would like to grab the data only once, where should we look to do that.
TODO:

"""

import pymc3 as pm
import pandas as pd
import numpy as np
import calibration
from optimization import grid_search
import pickle

def calculate_consumption_lagged(consumption_factor_values):
    """
    Calculate consumption factor lagged values.
    """

    pass

def calculate_eia_monthly_consumption_constraints(model,
                                                  eia_daily_observations,
                                                  eia_monthly_values,
                                                  sigma=1000):
    """
    Calculate eia monthly consumption constraints.

    """
    pass


def save_parameters(accuracy_result):
    pickle.dump(accuracy_result, open("accuracy_result.pkl", "wb"))


def fit_residential_model(consumption_factor: pd.DataFrame,
                          weather_data: pd.DataFrame,
                          daily_start_date,
                          daily_end_date,
                          eia_monthly_values,
                          consumption_factor_values):
    """
    Fits the residential model.

    https://en.wikipedia.org/wiki/List_of_cities_and_counties_in_Virginia

    :return:
    """

    calibrated_parameters = calibration.calibration(consumption_factor, weather_data)
    thin_grid = grid_search.get_thin_grid(calibrated_parameters)
    broad_grid = grid_search.get_thick_grid(calibrated_parameters)
    result  = []
    for params in thin_grid:
        accuracy = infer_residential_model(daily_start_date,
                                daily_end_date,
                                eia_monthly_values,
                                consumption_factor_values,
                                params)
        result.append((params, accuracy))

    for params in broad_grid:
        accuracy = infer_residential_model(daily_start_date,
                                daily_end_date,
                                eia_monthly_values,
                                consumption_factor_values,
                                params)
        result.append((params, accuracy))


    save_parameters(result)
    return result


def infer_residential_model(daily_start_date,
                            daily_end_date,
                            eia_monthly_values,
                            consumption_factor_values,
                            params):
    """
    The goal is to infer a statistical model for a set of dates starting
    at (1) daily_start_date and ending at (2) daily_end_date.

    There will also be monthly eia constraints that are passed in via dictionary.
    called eia_monthly_values: {"2025-01-01": 3400, "2025-02-01": 3500, ...}

    The hyperparameters that set core elements of the statistical distribution are passed in via params.

    ---------------------------------------------------------------------------------------------------

    The parameter weather_wind_population_weighted is a measure of the consumption that will be required,
    it should be highly correlated with the Natural Gas Consumption, and correctly capture the many effects
    that could impact consumption.

    For instance, weather_wind_population_weighted should capture the effects of:
        1. population: if population goes up, consumption goes up.
        2. wind: if wind goes up, consumption goes up.
        3. weather: if hdd goes up, then consumption goes up.
        4. economic factors, if economic factors go up, consumption goes up.

    """

    dates = pd.date_range(daily_start_date, daily_end_date)
    consumption_factor_lagged_values = calculate_consumption_lagged(consumption_factor_values)
    mean_consumption_factor = np.mean(consumption_factor_values)
    variance_consumption_factor = np.var(consumption_factor_values)

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


        alpha = pm.Normal("alpha_1", mu=params.get("alpha_mu"), simga=params.get("alpha_sigma"))
        alpha_2 = pm.Normal("alpha_2", mu=params.get("alpha_2_mu"), simga=params.get("alpha_2_sigma"))

        minimum_consumption = pm.Normal("minimum_consumption",
                                        mu=params.get("minimum_consumption_mu"),
                                        sigma=params.get("minimum_consumption_sig"))

        eia_daily_observations = pm.Normal("eia_observations",
                                           mu=(alpha + alpha_2) * consumption_factor + alpha_2 * consumption_factor_lagged + minimum_consumption,
                                           sigma=0.2,
                                           dims="dates")


        calculate_eia_monthly_consumption_constraints(model,
                                                      eia_daily_observations,
                                                      eia_monthly_values,
                                                      params.get("daily_consumption_error"))

        idata = pm.sample(draws=1000, tune=1000)
        return idata




