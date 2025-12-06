"""



"""

from data.eia_consumption.eia_consumption import get_eia_consumption_data_in_pivot_format
from models.residential import load_residential_data
from multiprocessing_methods.multiprocessing_framework import init_logs
import datetime


def calculate_mean_values_per_month(data, state, years):


    full_eia_data = data["full_eia_data"]
    full_eia_data = full_eia_data.reset_index()
    full_eia_data["Date"] = full_eia_data["period"].apply(lambda x: datetime.datetime.strptime(x + "-01", "%Y-%m-%d"))
    full_eia_data["Month"] = full_eia_data["Date"].dt.month
    full_eia_data["Year"] = full_eia_data["Date"].dt.year
    full_eia_data["Day"] = full_eia_data["Date"].dt.day
    full_eia_data[state] = full_eia_data[state].astype(float)
    full_eia_data = full_eia_data[full_eia_data["Year"].isin(years)]
    average_monthly_values = full_eia_data.groupby("Month")[state].mean().reset_index()

    return average_monthly_values

def run_historical_analysis(data, state, mean_years, validate_years):
    """
    Calculate the error for a variety of years.


    :param data:
    :param state:
    :return:
    """


    average_monthly_values = calculate_mean_values_per_month(data, state, mean_years)
    full_eia_data = data["full_eia_data"]
    full_eia_data = full_eia_data.reset_index()
    full_eia_data["Date"] = full_eia_data["period"].apply(lambda x: datetime.datetime.strptime(x + "-01", "%Y-%m-%d"))
    full_eia_data["Month"] = full_eia_data["Date"].dt.month
    full_eia_data["Year"] = full_eia_data["Date"].dt.year
    full_eia_data["Day"] = full_eia_data["Date"].dt.day
    full_eia_data[state] = full_eia_data[state].astype(float)
    full_eia_data = full_eia_data[full_eia_data["Year"].isin(validate_years)]

    pct_errors = []
    for index, row in full_eia_data.iterrows():
        row_date = row["Date"]
        month = row["Month"]
        month_value = float(row[state])
        average_monthly_value = float(average_monthly_values.query(f"Month == {month}")[state].iloc[0])
        pct_error = (month_value - average_monthly_value) / average_monthly_value
        pct_errors.append(pct_error)

    average_error = sum(pct_errors) / len(pct_errors)

    print(f"Average error for {state} is: {average_error}")


def historical_benchmark(state, consumption_factor_method):

    file_handler, log_handler = init_logs(state, "residential")
    app_params = dict()
    app_params["file_handler"] = file_handler
    app_params["log_handler"] = log_handler

    start_training_time = "2020-01-01"
    end_training_time = "2025-01-01"
    consumption_factor_method = "POPULATION_WEIGHTED_HDD"

    data, consumption_factor, eia_data = load_residential_data(state,
                                                               start_training_time,
                                                               end_training_time,
                                                               consumption_factor_method=consumption_factor_method,
                                                               app_params=app_params)

    run_historical_analysis(data, state, [2020, 2021,2022, 2023], [2024])



if __name__ == "__main__":
    historical_benchmark("New York", "POPULATION_WEIGHTED_HDD")