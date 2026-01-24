"""
In the following module, I will aim to use the daily apis to develop
trading daily trading signals.


    daily_values = calculate_eia_daily_values("2016-01-01",
                               "2025-09-30",
                               "2016-01-01",
                               "2025-09-30",
                               "2010-01-01",
                               "2015-12-31",
                               "2025-09-30",
                               ComponentType.RESIDENTIAL,
                               "Virginia")


"""

import datetime
import pandas as pd
from baseline.baseline_stepwise_regression import ComponentType, calculate_eia_daily_values
from data.eia_consumption.eia_geography_mappings import us_state_to_abbrev, us_state_to_abbrev_supported_for_trading, \
    us_state_to_abbrev_supported_by_prescient
import logging




def calculate_total_gas_consumed(current_date) -> float:

    start_date = "2016-01-01"
    end_date = "2027-01-31"
    total_gas_consumed = 0
    for state in us_state_to_abbrev_supported_for_trading:
        daily_values = calculate_eia_daily_values(start_date,
                                                    end_date,
                                                    "2016-01-01",
                                                    "2025-10-31",
                                                    "2010-01-01",
                                                    "2015-12-31",
                                                    current_date,
                                                    ComponentType.RESIDENTIAL,
                                                    state)

        state_sum = daily_values.sum()["Value"]
        total_gas_consumed += state_sum
    return total_gas_consumed


def calculate_total_day_over_day_gas_difference(current_date = datetime.datetime.now()):
    previous_date = current_date - datetime.timedelta(days=1)
    current_date_total_gas = calculate_total_gas_consumed(current_date)
    previous_date_total_gas = calculate_total_gas_consumed(previous_date)
    return (current_date_total_gas - previous_date_total_gas) / previous_date_total_gas


def calculate_day_over_day_difference(start_date_calc: str,
                                      end_date_calc: str):

    dates = pd.date_range(start=start_date_calc, end=end_date_calc)
    date_to_difference = dict()
    for date in dates:
        date_to_difference[date] = calculate_total_day_over_day_gas_difference(date)
    return date_to_difference

if __name__ == "__main__":

    day_over_day_difference = calculate_day_over_day_difference("2026-01-13",
                                                                "2026-01-14")


    logging.info(day_over_day_difference)








