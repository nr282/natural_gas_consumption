"""
The file aims to provide utility functions.
    1. Checking that the values are close to being near the average.
    2. These are aimed to be checks against various values that are calculated.
"""

#TODO: How can we implement the relevant checks.
#TODO: One can implement alternative ways of calculating the values.
from data.eia_consumption.eia_consumption import get_eia_consumption_data_bulk_df
from datetime import datetime
import pandas as pd
from baseline.baseline import ComponentType
import numpy as np

component_type_to_component_name = {ComponentType.RESIDENTIAL: "Residential",
                                    ComponentType.ELECTRIC: "Electric",
                                    ComponentType.COMMERCIAL: "Commercial"}


def check_near_consumption_value(state: str,
                                 daily_values,
                                 component_type: ComponentType):


    sampled = daily_values.sample(n=10)
    total = 0
    num_accurate = 0
    for index, row in sampled.iterrows():
        date = row["Date"]
        candidate_daily_value = float(row["Value"])
        year = date.year
        month = date.month
        first_date_of_year_str = f"{year}-01-01"
        last_date_of_year_str = f"{year}-12-31"
        period = f"{year}-{str(month).zfill(2)}"
        df = None
        for i in range(10):
            try:
                df = get_eia_consumption_data_bulk_df(first_date_of_year_str,
                                                    last_date_of_year_str)
            except Exception as e:
                pass


        if df is None:
            continue

        df = df[df["period"] == period]
        df = df[df["standard_state_name"] == state]
        if len(df) == 0:
            pass
        else:
            df = df[df["series-description"].apply(lambda description: component_type_to_component_name[component_type] in description)]
            monthly_value = float(df["value"].iloc[0])
            if monthly_value == float('inf') or monthly_value == float('nan'):
                pass
            else:
                daily_value = monthly_value / 30
                if np.isclose(daily_value, candidate_daily_value, rtol=0.3):
                    num_accurate += 1
                total += 1

    return num_accurate / total * 100

