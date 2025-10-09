"""
Tests various aspects of the child files.
"""

import pandas as pd
import os
from data.state_config.virginia.virginia_consumption_factor import VirginiaPopulationData
from data.consumption_factor.consumption_factor_calculation import calculate_consumption_factor
from data.weather import Weather, PyWeatherData
import datetime
from location import location

if __name__ == "__main__":

    virginia_population = VirginiaPopulationData()
    locations = [(37.7, -75.80)]
    weather_service = PyWeatherData(locations)
    start_datetime = datetime.datetime(2022, 1, 1)
    end_datetime = datetime.datetime(2024, 1, 1)
    calculate_consumption_factor(virginia_population,
                                 weather_service,
                                 start_datetime,
                                 end_datetime)




    import pdb
    pdb.set_trace()

