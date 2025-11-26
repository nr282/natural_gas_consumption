"""
The weather module will aim to be the major supplier of weather data to the application.

A key component will be providing an abstract class (ie interface) to which weather datasets will be
mapped.

This will allow the user of a weather data interopability between a variety of different weather datasets.

To express this, I present the diagram below:

Weather Dataset 1 ------------->
Weather Dataset 2 ------------->   Weather Interface -------> Client Code uses Weather Interface.
Weather Dataset 3 ------------->

"""

from datetime import datetime
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
from location import raw_name_to_standard_name, get_list_of_standardizied_name
import logging
from .mathematical_models_natural_gas import calculate_hdd, calculate_cdd, TemperatureType
from utils import *


logging.basicConfig(level=logging.INFO)
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
            logging.warning(f"No weather data for {city}, {state}")

    complete_data = pd.concat(dataframes)

    pivot_complete_data = pd.pivot_table(complete_data, values=["tavg"], columns=["City", "State"], index="time")
    pivot_complete_data = pivot_complete_data.reset_index()
    pivot_complete_data["Datetime"] = pivot_complete_data["time"]
    pivot_complete_data["Year"] = pivot_complete_data["Datetime"].dt.year
    pivot_complete_data["Month"] = pivot_complete_data["Datetime"].dt.month
    pivot_complete_data["Day"] = pivot_complete_data["Datetime"].dt.day

    pivot_complete_data.columns = pivot_complete_data.columns.map(''.join)
    return pivot_complete_data


def get_prescient_weather_data_via_api():
    raise NotImplementedError("Currently the Prescient API is not implemented.")

def get_prescient_weather_data_via_csv(state):

    path = os.path.join(get_base_path(), "data", "weather", state, f"{state.lower()}_hdd_cdd_obs.csv")
    df = pd.read_csv(path)
    return df

def get_prescient_weather_data(state):
    """
    Gets the prescient weather data. Prescient Weather Data is a particular weather
    vendor who has provided us with both csv and api access.

    :return:
    """

    try:
        df = get_prescient_weather_data_via_api()
    except NotImplementedError:
        df = get_prescient_weather_data_via_csv(state)
    except Exception as e:
        logging.error(f"Could not get prescient weather data via api or csv. Error: {e}")
        raise e

    return df



class Weather(ABC):
    """
    Weather object that standardizes the weather data.

    This will decouple (1) the weather data from a particular source from (2) the clients
    that use the weather data.

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
        self.raw_df[self.get_standardizied_name()] = self.raw_df[self.get_native_date_name()]
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

class PyWeatherData(Weather):
    """
    PyWeather implements Weather using the python-weather library.

    It aims to map the incoming data via the python weather library to
    a universal class that will be the interface for all weather data.

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
        raise NotImplemented()

    def get_cdd(self, locations: List[location], start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplemented()

    def get_hdd(self, locations: List[location], start: datetime, end: datetime) -> dict:

        date_range = pd.date_range(start=start, end=end, freq="D")
        df = pd.DataFrame(date_range, columns=["Date"])
        res = df.merge(self.raw_df, on="Date", how="left")
        res = res[["Date", "hdd"]]
        res = res.rename(columns={"hdd": "HDD"})
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