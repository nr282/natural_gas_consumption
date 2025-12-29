"""
The weather_mod module will aim to be the major supplier of weather_mod data to the application.

A key component will be providing an abstract class (ie interface) to which weather_mod datasets will be
mapped.

This will allow the user of a weather_mod data interopability between a variety of different weather_mod datasets.

To express this, I present the diagram below:

Weather Dataset 1 ------------->
Weather Dataset 2 ------------->   Weather Interface -------> Client Code uses Weather Interface.
Weather Dataset 3 ------------->


TODO: Need to get New York setup.
TODO: After getting New York setup, we can move on to other states.

"""

import pandas as pd
import python_weather
from meteostat import Point, Daily
from datetime import datetime, timedelta
import os
from sklearn import linear_model
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Optional
from collections import namedtuple

from data.eia_consumption.eia_geography_mappings import abbrev_to_us_state, us_state_to_abbrev
from location import raw_name_to_standard_name, get_list_of_standardizied_name
import logging
from .mathematical_models_natural_gas import calculate_hdd, calculate_cdd, TemperatureType
from utils import *
from data.weather_mod.forecast.acquire_prescient import get_weather_data_for_all_states, get_weather_data_for_state
import requests
from io import StringIO



location = namedtuple('Location', ['Latitude', 'Longitude'])

def get_longitude_and_latitude_of_locations():
    d = {
        ("Washington", "DC"): (38.9072, -77.0369),
        ("Richmond", "Virginia"): (37.5, -77.43),
        ("Norfolk", "Virginia"): (36.85, -76.28),
        ("Blacksburg", "Virginia"): (37.2301, -80.41),
        ("Jersey City", "New Jersey"): (40.7195, -74.04),
        ("New York", "New York"): (40.71, -74.006),
        ("San Francisco", "California"): (37.7749, -122.4194),
    }

    return d

def get_weather_data(start: datetime,
                     end: datetime,
                     locations=None) -> pd.DataFrame:
    dataframes = []

    if locations is None:
        longitude_and_latitude_of_locations = get_longitude_and_latitude_of_locations()
    else:
        longitude_and_latitude_of_locations = locations

    location_num = 0
    for location in longitude_and_latitude_of_locations:
        city, state = location
        long, lat = longitude_and_latitude_of_locations.get(location)
        loc_point = Point(long, lat)
        data = Daily(loc_point, start, end)
        data = data.fetch()
        if not data.empty:
            data["City"] = city
            data["State"] = state
            dataframes.append(data)

            if data.index.name != "time":
                raise ValueError("Dataframe index name is not time.")

            location_num += 1
        else:
            logging.warning(f"No weather_mod data for {city}, {state}")

    complete_data = pd.concat(dataframes)

    pivot_complete_data = pd.pivot_table(complete_data, values=["tavg"], columns=["City", "State"], index="time")
    pivot_complete_data = pivot_complete_data.reset_index()
    pivot_complete_data["Datetime"] = pivot_complete_data["time"]
    pivot_complete_data["Year"] = pivot_complete_data["Datetime"].dt.year
    pivot_complete_data["Month"] = pivot_complete_data["Datetime"].dt.month
    pivot_complete_data["Day"] = pivot_complete_data["Datetime"].dt.day

    pivot_complete_data.columns = pivot_complete_data.columns.map(''.join)
    return pivot_complete_data


def format_df(forecast_df: pd.DataFrame) -> pd.DataFrame:


    forecast_df["Region Type"] = "STATE"
    forecast_df["Date"] = forecast_df["fcstdate"]
    forecast_df = forecast_df.rename(columns={"popcdd": "Population CDD",
                                              "region": "Region",
                                              "pophdd": "Population HDD"})
    forecast_df["Forecast Type"] = "Forecast"

    return forecast_df

def get_prescient_weather_data_via_api(state: str,
                                       current_date=datetime.now()) -> pd.DataFrame:
    """
    Accumulate both forecast and historical data for the given state.

    """


    if state in abbrev_to_us_state:
        pass
    elif state not in abbrev_to_us_state:
        if state in us_state_to_abbrev:
            state = us_state_to_abbrev[state]
        if state not in abbrev_to_us_state:
            raise ValueError(f"State {state} not found in abbrev_to_us_state.")

    data_historical = requests.get("https://s2s.worldclimateservice.com/wcs/regional_degree_day_history_daily_v2025.csv")
    if data_historical.status_code == 200:
        data_string = data_historical.content.decode('utf-8')
        historical_df = pd.read_csv(StringIO(data_string))
        historical_df["Forecast Type"] = "Historical"

    forecast_cdd_df = get_weather_data_for_state(current_date, state, "popcdd")
    forecast_hdd_df = get_weather_data_for_state(current_date, state, "pophdd")
    forecast_hdd_df = forecast_hdd_df.drop(columns=["fcstdate", "initdate", "region"])
    forecast_df = pd.concat([forecast_hdd_df, forecast_cdd_df], axis=1)
    forecast_df = format_df(forecast_df)
    df = pd.concat([historical_df, forecast_df])
    df = df[df["Region Type"] == "STATE"]
    df = df[df["Region"] == state]

    assert(df.Date.value_counts().max() == 1)
    assert("Historical" in df["Forecast Type"].unique())
    assert("Forecast" in df["Forecast Type"].unique())

    return df


def prescient_weather_data_via_csv_handler(state):
    """
    A major goal of this function is to find a correct csv path to get data.



    """

    p_0 = os.path.join(get_base_path(), "data", "weather_mod", state, f"{state.lower()}_hdd_cdd_obs.csv")
    p_1 = os.path.join(get_base_path(), "data", "weather_mod", "State", f"regional_degree_day_history_daily_v2025.csv")

    paths_for_prescient_weather_data = [p_1, p_0]
    for path in paths_for_prescient_weather_data:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df
        else:
            logging.warning(f"Could not find path {path}")
    raise Exception("Could not find path for prescient weather_mod data.")


def standardize_df(df):

    if "Region Type" in df and "Gas HDD" in df and "Region" in df:
        df = df[df["Region Type"] == 'STATE']
        df["Region"] = df["Region"].apply(lambda abbrv: abbrev_to_us_state.get(abbrv))
        df = df.rename(columns={"Gas HDD": "HDD"})
    else:
        pass

    return df


def get_prescient_weather_data_via_csv(state):

    df = prescient_weather_data_via_csv_handler(state)
    standardizied_df = standardize_df(df)
    standardizied_df = standardizied_df[standardizied_df["Region"] == state]

    return standardizied_df

def get_prescient_weather_data(state):
    """
    Gets the prescient weather_mod data. Prescient Weather Data is a particular weather_mod
    vendor who has provided us with both csv and api access.

    :return:
    """

    try:
        df = get_prescient_weather_data_via_api(state)
    except NotImplementedError:
        df = get_prescient_weather_data_via_csv(state)
    except Exception as e:
        logging.error(f"Could not get prescient weather_mod data via api or csv. Error: {e}")
        raise e

    return df



class Weather(ABC):
    """
    Weather object that standardizes the weather_mod data.

    This will decouple (1) the weather_mod data from a particular source from (2) the clients
    that use the weather_mod data.

    Weather data can include temperature, wind speed, humidity etc, but the primary location
    is temperature.

    A diagram can summarize this information:

    Weather Input Source 1 -------->
    Weather Input Source 2 --------> -------------> Weather Interface --------> Client Code uses Weather Interface.
    Weather Input Source 3 -------->

    """

    def __init__(self, locations):
        self.native_name = None
        self.locations = locations
        self.raw_df = self.acquire_native_data()
        self.refactor_date()
        self.refactor_locations()
        self.calculate_hdd_and_cdd()


    @abstractmethod
    def get_locations(self) -> List[location]:
        """
        Provides list of tuples where the (1) first element is longitude and (2) second element is latitude.

        """
        pass

    @abstractmethod
    def get_temperature(self, locations: List[location], start: datetime, end: datetime) -> pd.DataFrame:
        """
        Get temperature for a list of locations from (1) start datetime to (2) end datetime.
        """
        pass


    @abstractmethod
    def get_complete_time_span(self) -> (datetime, datetime):
        """
        Weather data is found between (1) start time and (2) end time for all locations for all dates.

        """
        pass

    @abstractmethod
    def get_min_and_max_time_span(self) -> (datetime, datetime):
        """
        Weather data is found between (1) start time and (2) end time for at least one location for all dates.
        """
        pass

    @abstractmethod
    def get_hdd(self, locations: List[location], start: datetime, end: datetime) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_cdd(self, locations: List[location], start: datetime, end: datetime) -> pd.DataFrame:
        pass

    @abstractmethod
    def set_native_date_name(self, native_name: str):
        pass

    @abstractmethod
    def get_native_date_name(self) -> str:
        pass

    @abstractmethod
    def acquire_native_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_standardizied_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_type_of_temperature(self):
        pass

    def calculate_hdd_and_cdd(self) -> pd.DataFrame:

        temperatureType = self.get_type_of_temperature()
        valid_locations = get_list_of_standardizied_name()
        for column in self.df.columns:
            if column in valid_locations:
                self.df[column + " HDD"] = self.df[column].apply(lambda x: calculate_hdd(x, temperatureType))
                self.df[column + " CDD"] = self.df[column].apply(lambda x: calculate_cdd(x, temperatureType))

    def get_standardizied_name(self):
        return "Date"

    def _convert_to_datetime(self, date_ser: pd.Series) -> pd.Series:
        return pd.to_datetime(date_ser, format="%Y-%m-%d")

    def refactor_date(self):
        if self.get_native_date_name() in self.raw_df and not self.get_standardizied_name() in self.raw_df:
            self.raw_df[self.get_standardizied_name()] = self.raw_df[self.get_native_date_name()]
        elif not (self.get_standardizied_name() in self.raw_df):
            raise ValueError(f"Native Name {self.get_native_date_name()} not Found in Dataframe "
                             f"and Standardized Name {self.get_standardizied_name()} not found."
                             f" Dataframe cannot be refactored.")
        date_ser = self.raw_df[self.get_standardizied_name()]
        self.raw_df[self.get_standardizied_name()] = self._convert_to_datetime(date_ser)
        assert(self.raw_df[self.get_standardizied_name()].dtype == "datetime64[ns]")
        self.df = self.raw_df.copy()

    def refactor_locations(self):
        """
        Refactor location columns.

        """

        logging.debug("Refactor locations in dataframe")
        columns = self.df.columns.tolist()
        for column in columns:
            standard_name = raw_name_to_standard_name(column)
            if standard_name != False:
                self.df.rename(columns={column: standard_name}, inplace=True)

def calculate_average_for_missing(res: pd.DataFrame,
                                  start_dt: datetime,
                                  end_dt: datetime,
                                  min_time: datetime,
                                  max_time: datetime,
                                  degree_day_type: str) -> pd.DataFrame:

    if "Date" in res.columns:
        res["Year"] = res["Date"].dt.year
        res["Month"] = res["Date"].dt.month
        res["Day"] = res["Date"].dt.day

    res = res.groupby(["Month", "Day"])[degree_day_type].mean().reset_index()
    date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
    averages = pd.DataFrame(date_range, columns=["Date"])
    averages["Month"] = averages["Date"].dt.month
    averages["Day"] = averages["Date"].dt.day
    averages["Year"] = averages["Date"].dt.year
    averages = averages.merge(res, on=["Month", "Day"], how="left")

    return averages


class PyWeatherData(Weather):
    """
    PyWeather implements Weather using the python-weather_mod library.

    It aims to map the incoming data via the python weather_mod library to
    a universal class that will be the interface for all weather_mod data.

    """

    def __init__(self, locations):
        super().__init__(locations)

    def get_standardizied_data(self):
        return self.df

    def get_temperature(self, locations: List[location], start: datetime, end: datetime) -> pd.DataFrame:
        pass

    def get_locations(self) -> List[location]:
        return self.locations

    def get_complete_time_span(self) -> (datetime, datetime):
        pass

    def get_min_and_max_time_span(self) -> (datetime, datetime):
        pass

    def get_cdd(self, locations: List[location], start: datetime, end: datetime) -> pd.DataFrame:
        pass

    def get_hdd(self, locations: List[location], start: datetime, end: datetime) -> dict:

        locations_dict = dict()
        for index, location in enumerate(locations):
            locations_dict[(str(location[0]), str(location[1]))] = location
        df = self.acquire_native_data(locations_dict)
        self.df = df

        d = dict()
        for column in df.columns:
            if column.startswith("tavg"):
                val = column[4:]
                long, lat = float(val.split("-")[0]), -1 * float(val.split("-")[1])
                df["HDD"] = df[column].apply(lambda x: calculate_hdd(x, self.get_type_of_temperature()))
                df.rename(columns={"time": "Date"}, inplace=True)
                d[(long, lat)] = df[["Date", "HDD"]]
        return d

    def get_type_of_temperature(self) -> TemperatureType:
        return TemperatureType.CELCIUS

    def set_native_date_name(self, native_name: str):
        self.native_name = native_name

    def acquire_native_data(self, locations=None) -> pd.DataFrame:
        df = get_weather_data(datetime(2020, 1, 1),
                                        datetime(2025, 1, 31),
                                        locations=locations)
        return df

    def get_native_date_name(self) -> str:
        return "Datetime"


class PrescientWeather(Weather):
    """
    Implements Prescient Weather data service either from the API or the data
    that was sent over for Virginia.

    The data comes in HDD format for the entire state and is already properly population
    weighted.
    """

    def __init__(self, locations):
        super().__init__(locations)

    def get_standardizied_data(self):
        raise NotImplemented()

    def get_temperature(self, locations: List[location], start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplemented()

    def get_locations(self) -> List[location]:
        raise NotImplemented()

    def get_complete_time_span(self) -> (datetime, datetime):
        raise NotImplemented()

    def get_min_and_max_time_span(self) -> (datetime, datetime):
        """
        Gets the minimum and maximum time span for the weather_mod data.
        """

        if "Date" in self.raw_df.columns:
            min_time = self.raw_df["Date"].min()
            max_time = self.raw_df["Date"].max()
            return min_time, max_time
        else:
            return None, None

    def get_cdd(self, locations: List[location], start: datetime, end: datetime) -> pd.DataFrame:

        assert(type(start) == str and type(end) == str)
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        if end_dt < start_dt:
            raise ValueError("Start date must be before end date.")

        date_range = pd.date_range(start=start, end=end, freq="D")
        df = pd.DataFrame(date_range, columns=["Date"])

        res = df.merge(self.raw_df, on="Date", how="left", validate="one_to_one")
        res = res[["Date", "Population CDD"]]
        res = res.rename(columns={"Population CDD": "CDD"})
        assert(type(start) == str and type(end) == str)
        min_time, max_time = self.get_min_and_max_time_span()
        if min_time >= start_dt or max_time <= end_dt:
            averages = calculate_average_for_missing(res,
                                                     start_dt,
                                                     end_dt,
                                                     min_time,
                                                     max_time,
                                                     "CDD")

            averages = averages.merge(res, on="Date", how="left", validate="one_to_one")
            averages["CDD"] = averages["CDD_y"].combine_first(averages["CDD_x"])


            res = averages
        else:
            pass

        assert("Date" in res.columns)
        assert("CDD" in res.columns)

        return res


    def get_hdd(self, locations: List[location], start, end) -> dict:

        assert (type(start) == str and type(end) == str)
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        if end_dt < start_dt:
            raise ValueError("Start date must be before end date.")

        date_range = pd.date_range(start=start, end=end, freq="D")
        df = pd.DataFrame(date_range, columns=["Date"])
        res = df.merge(self.raw_df, on="Date", how="left", validate="one_to_one")
        res = res[["Date", "Population HDD"]]
        res = res.rename(columns={"Population HDD": "HDD"})
        assert (type(start) == str and type(end) == str)
        min_time, max_time = self.get_min_and_max_time_span()
        if min_time >= start_dt or max_time <= end_dt:
            averages = calculate_average_for_missing(res,
                                                     start_dt,
                                                     end_dt,
                                                     min_time,
                                                     max_time,
                                                     "HDD")

            averages = averages.merge(res, on="Date", how="left", validate="one_to_one")
            averages["HDD"] = averages["HDD_y"].combine_first(averages["HDD_x"])

            res = averages
        else:
            pass

        assert ("Date" in res.columns)
        assert ("HDD" in res.columns)

        return res


    def get_type_of_temperature(self) -> TemperatureType:
        return TemperatureType.CELCIUS

    def set_native_date_name(self, native_name: str):
        raise NotImplemented()

    def acquire_native_data(self, locations=None) -> pd.DataFrame:

        if locations is None:
            df = get_prescient_weather_data(self.locations[0])
        else:
            raise NotImplemented(f"Locations passed to the function acquire native {locations}")
        return df

    def get_native_date_name(self) -> str:

        return "date"
        #self.raw_df.date


def test_get_weather():
    locations = dict()
    locations[("Washington", "DC")] = (38.9072, -77.0369)
    pyweather_data = PyWeatherData(locations)
    data = pyweather_data.get_standardizied_data()
    return data


if __name__ == "__main__":
    pass