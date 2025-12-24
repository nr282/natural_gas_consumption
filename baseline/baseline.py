"""
The following aims to be the baseline technique that I think is good enough to commercialize.

TODO: There are 8 functions that need to be implemented to implement the minimum viable
TODO: prototype.

TODO: What is the key goal here?
TODO: We will also need to put this in the AWS API.
TODO: We should look to keep track of the performance
TODO: characteristics of the API.
#TODO: As I write this code, what should we be able to consider:
    #TODO: (1) We want everything to follow strong schemas
    #TODO: (2) We would like to only produce a result if it is accurate.
        #TODO: (3) This will require checking of the input data for errors.
        #TODO: (4) This will require having weather data hooked up.
        #TODO: (5) We will need to handle how we calculate the consumption values.
        #TODO: (6) If the consumption values are in the future, then they will be
        #TODO: provided in the future.
        #TODO: As such, I think it will be good to parameterize the code, via
        #TODO: looking at the current day, and what divides the past, current and (2)
        #TODO:

"""
import matplotlib.pyplot as plt
import numpy as np

from data.weather import PrescientWeather
import datetime
from enum import Enum
from models.seasonality.seasonality import calculate_differences_for_df
from data.eia_consumption.eia_consumption import get_eia_consumption_data_in_pivot_format
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import calendar



class ComponentType(Enum):
    RESIDENTIAL="RESIDENTIAL"
    COMMERCIAL="COMMERCIAL"
    ELECTRIC="ELECTRIC"

component_to_type = dict()
component_to_type[ComponentType.RESIDENTIAL] = "HDD"
component_to_type[ComponentType.COMMERCIAL] = "HDD"
component_to_type[ComponentType.ELECTRIC] = "CDD"


def calculate_consumption_factor_diff(start_date,
                                      end_date,
                                      normal_weather,
                                      weather_values,
                                      component_type: ComponentType):



    weather_values = calculate_differences_for_df(weather_values, component_to_type[component_type])
    weather_values["diff"] = weather_values.apply(lambda row: row[component_to_type[component_type]]
                                                    - normal_weather[row["day_of_year"]], axis=1)

    weather_values["Day"] = weather_values["Date"].apply(lambda x: x.day)
    weather_values["Year"] = weather_values["Date"].apply(lambda x: x.year)
    weather_values["Month"] = weather_values["Date"].apply(lambda x: x.month)
    return weather_values[["Date", "Year", "Month", "Day", "diff"]]


def calculate_monthly_eia_values(start_date,
                                 end_date,
                                 consumption_factor_values,
                                 consumption_type: str):

    pass


def calculate_consumption_factor_via_weather(start_date,
                                             end_date,
                                             weather_normal,
                                             weather_values):


    pass

def create_daily_eia_via_weather(eia_monthly_value,
                                start_date,
                                end_date,
                                consumption_factor):
    """
    Given an eia_monthly_value such as accumulated consumption in a particular month
    and given the consumption_factor. On the basis of the consumption factor, form
    the daily values that aggregate up to the eia monthly value.
    """

    pass


def create_weather_values(start_date,
                          end_date,
                          current_date,
                          state,
                          component_type: ComponentType):

    weather = None
    if component_type in [ComponentType.COMMERCIAL, ComponentType.RESIDENTIAL]:
        prescient_weather = PrescientWeather([state])
        weather = prescient_weather.get_hdd([state], start_date, end_date)
    elif component_type in [ComponentType.ELECTRIC]:
        prescient_weather = PrescientWeather([state])
        weather = prescient_weather.get_cdd([state], start_date, end_date)
    else:
        pass
    return weather

def create_normal_weather_values(normal_start_date,
                                 normal_end_date,
                                 state,
                                 component_type: ComponentType) -> dict:
    """
    Creates the normal weather values.

    """

    weather = None
    if component_type in [ComponentType.COMMERCIAL, ComponentType.RESIDENTIAL]:
        prescient_weather = PrescientWeather([state])
        weather = prescient_weather.get_hdd([state], normal_start_date, normal_end_date)
    elif component_type in [ComponentType.ELECTRIC]:
        prescient_weather = PrescientWeather([state])
        weather = prescient_weather.get_cdd([state], normal_start_date, normal_end_date)
    else:
        pass

    weather = calculate_differences_for_df(weather, component_to_type[component_type])
    weather_dict = weather[["day_of_year", "avg_dd"]].set_index("day_of_year").to_dict()["avg_dd"]
    return weather_dict


def calculate_consumption_factor_to_eia_sensitivity(start_date,
                                                    end_date,
                                                    eia_monthly_values,
                                                    consumption_factor_values):
    """
    Calculate the eia sensitivity between (a) consumption factor and (b) eia_monthly_value
    between the start_date and end_date.


    """


    pass



def calculate_consumption_factor_to_eia_sensitivity_monthly(eia_start_date,
                                                            eia_end_date,
                                                            eia_monthly_values,
                                                            consumption_factor_values,
                                                            state,
                                                            component_type: ComponentType
                                                            ):
    """
    Calculate the consumption factor to eia sensitivity on a month-by-month basis.
    """


    consumption_factor_by_month = consumption_factor_values.groupby(["Year", "Month"])["diff"].sum().reset_index()


    comparison = eia_monthly_values.merge(consumption_factor_by_month,
                                           on=["Year", "Month"],
                                           how="inner", suffixes=("_eia", "_consumption_factor"))

    comparison.plot(x="diff_consumption_factor", y="diff_eia", kind="scatter")
    plt.savefig(f"{state}_{component_type}_diff.png")

    X = comparison["diff_consumption_factor"].values
    y = comparison["diff_eia"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    res = stats.linregress(X_train, y_train)
    y_predict = res.intercept + res.slope * X_test

    error = np.sum(np.abs(y_test - y_predict))
    total_mag = np.sum(np.abs(y_test))

    print(f"Error divided by total mag is {error/total_mag}")

    params = {"slope": res.slope, "intercept": res.intercept}
    return params

def calculate_eia_normal_monthly_values(normal_start_date,
                                        normal_end_date,
                                        state,
                                        component_type: ComponentType
                                        ) -> dict:
    """
    Calculate monthly eia values based on consumption factor values.

    """

    component_name = {ComponentType.RESIDENTIAL: "Residential",
                      ComponentType.COMMERCIAL: "Commercial",
                      ComponentType.ELECTRIC: "Electric"}[component_type]

    eia_values = get_eia_consumption_data_in_pivot_format(start_date=normal_start_date,
                                                         end_date=normal_end_date,
                                                         canonical_component_name=component_name)
    eia_values = eia_values[[state, "period"]]
    eia_values["Month"] = eia_values["period"].apply(lambda x: int(x[-2:]))
    eia_values["Year"] = eia_values["period"].apply(lambda x: int(x[:4]))
    eia_values = eia_values[["Month", state]]
    eia_values[state] = eia_values[state].astype(float)
    eia_month_values = eia_values.groupby(["Month"])[state].mean().to_dict()
    return eia_month_values


def calculate_eia_monthly_values(eia_start_date,
                                 eia_end_date,
                                 state,
                                 component_type: ComponentType) -> pd.DataFrame:
    component_name = {ComponentType.RESIDENTIAL: "Residential",
                      ComponentType.COMMERCIAL: "Commercial",
                      ComponentType.ELECTRIC: "Electric"}[component_type]

    eia_values = get_eia_consumption_data_in_pivot_format(start_date=eia_start_date,
                                                          end_date=eia_end_date,
                                                          canonical_component_name=component_name)
    eia_values = eia_values[[state, "period"]]
    eia_values["Month"] = eia_values["period"].apply(lambda x: int(x[-2:]))
    eia_values["Year"] = eia_values["period"].apply(lambda x: int(x[:4]))
    eia_values["Day"] = 1
    eia_values["Date"] = eia_values["period"].apply(lambda x: datetime.datetime(year=int(x[:4]),
                                                                                month=int(x[-2:]),
                                                                                day=1))

    eia_values = eia_values[["Date", "Year", "Month", "Day", state]]
    eia_values[state] = eia_values[state].astype(float)
    return eia_values

def calculate_eia_values_diff(start_date,
                             end_date,
                             eia_normal,
                             eia_monthly_values,
                             state):

    eia_monthly_values["diff"] = eia_monthly_values.apply(lambda row: row[state] - eia_normal[row["Month"]] , axis=1)
    return eia_monthly_values


def convert_date_str_to_datetime(date_str: str):

    return datetime.datetime.strptime(date_str, "%Y-%m-%d")

def check_preconditions(start_date: str,
                        end_date: str,
                        eia_start_date: str,
                        eia_end_date: str,
                        normal_start_date: str,
                        normal_end_date: str,
                        current_date: str):



    start_datetime = convert_date_str_to_datetime(start_date)
    end_datetime = convert_date_str_to_datetime(end_date)
    eia_start_datetime = convert_date_str_to_datetime(eia_start_date)
    eia_end_datetime = convert_date_str_to_datetime(eia_end_date)
    normal_start_datetime = convert_date_str_to_datetime(normal_start_date)
    normal_end_datetime = convert_date_str_to_datetime(normal_end_date)
    current_datetime = convert_date_str_to_datetime(current_date)

    primary_ordering = (start_datetime <= end_datetime)
    eia_ordering = (eia_start_datetime <= eia_end_datetime)
    normal_ordering = (normal_start_datetime <= normal_end_datetime)
    normal_end_less_than_start = (normal_end_datetime < start_datetime)
    start_less_than_current = (start_datetime < current_datetime)
    current_less_than_end = (current_datetime < end_datetime)
    start_less_than_eia_start = (start_datetime < eia_start_datetime)
    eia_start_less_than_current = (eia_start_datetime < current_datetime)


    precondition = [
        primary_ordering,
        eia_ordering,
        normal_ordering,
        normal_end_less_than_start,
        normal_end_less_than_start,
        start_less_than_current,
        current_less_than_end,
        start_less_than_eia_start,
        start_less_than_current
    ]

    checks = {"start_datetime <= end_datetime": primary_ordering,
              "eia_start_datetime <= eia_end_datetime": eia_ordering,
              "normal_start_datetime <= normal_end_datetime": normal_ordering,
              "normal_end_datetime < start_datetime": normal_end_less_than_start,
              "start_datetime < current_datetime": start_less_than_current,
              "current_datetime < end_datetime": current_less_than_end,
              "start_datetime < eia_start_datetime": start_less_than_eia_start,
              "eia_start_datetime < current_datetime": eia_start_less_than_current}

    return all(precondition), checks


def is_between(date, start_date, end_date):
    return (start_date <= date) and (date <= end_date)


def get_dates_in_month(month, year):

    return pd.date_range(start=datetime.date(year, month, 1),
                         end=datetime.date(year, month, calendar.monthrange(year, month)[1]))

def calculate_eia_daily_value(eia_monthly_value: float,
                              weather_values: pd.Series,
                              date: datetime.datetime,
                              component_type: ComponentType,
                              month,
                              year):

    dates_in_month = get_dates_in_month(month, year)
    weather_dates_in_month = weather_values["Date"].unique()

    if len(dates_in_month) != len(weather_dates_in_month):
        return float('nan')

    year = date.year
    month = date.month
    day = date.day

    degree_day_type = component_to_type[component_type]
    total_dd = weather_values[degree_day_type].sum()
    eia_per_dd = eia_monthly_value / total_dd
    weather_values["eia_daily"] = weather_values[degree_day_type].apply(lambda dd: eia_per_dd * dd)
    eia_daily_value = weather_values.query(f"Month == {month} "
                                           f"and Year == {year} "
                                           f"and Day == {day}")["eia_daily"].iloc[0]

    return float(eia_daily_value)

def calculate_eia_daily_values_with_params(eia_monthly_values,
                                            eia_normal_values,
                                            weather_normal_values,
                                            weather_values,
                                            start_date,
                                            end_date,
                                            eia_start_date,
                                            eia_end_date,
                                            normal_start_date,
                                            normal_end_date,
                                            current_date,
                                            params,
                                            state,
                                            component_type: ComponentType):
    """
    Calculate eia daily values with params.
    """

    start_date = convert_date_str_to_datetime(start_date)
    end_date = convert_date_str_to_datetime(end_date)
    eia_start_date = convert_date_str_to_datetime(eia_start_date)
    eia_end_date = convert_date_str_to_datetime(eia_end_date)
    current_date = convert_date_str_to_datetime(current_date)

    dates_to_predict = pd.date_range(start=start_date,
                                     end=end_date,
                                     freq="D")

    date_to_value = dict()
    for date in dates_to_predict:
        month = date.month
        year = date.year
        #NOTE: Weather values may not include all weather values for the provided month.
        weather_values_for_month = weather_values.query(f"Month == {month} and Year == {year}")
        if is_between(date, eia_start_date, eia_end_date):
            eia_monthly_values = eia_monthly_values.query(f"Month == {month} and Year == {year}")
            if len(eia_monthly_values) == 1:
                eia_monthly_value = float(eia_monthly_values[state].iloc[0])
                daily_value = calculate_eia_daily_value(eia_monthly_value,
                                                        weather_values_for_month,
                                                        date,
                                                        component_type,
                                                        month,
                                                        year)
            elif len(eia_monthly_values) == 0:
                eia_monthly_value = None
                daily_value = None
            else:
                print(f"EIA Monthly Values: There are more than one: {eia_monthly_values}")
                eia_monthly_value = float(eia_monthly_values[state].iloc[0])
                daily_value = calculate_eia_daily_value(eia_monthly_value,
                                                        weather_values_for_month,
                                                        date,
                                                        component_type,
                                                        month,
                                                        year)
        else:
            predicted_eia_monthly_value = (params["slope"] * weather_values_for_month["diff"].sum()
                                           + params["intercept"] +
                                           eia_normal_values[month])
            daily_value = calculate_eia_daily_value(predicted_eia_monthly_value,
                                                    weather_values_for_month,
                                                    date,
                                                    component_type,
                                                    month,
                                                    year)
        date_to_value[date] = daily_value

    result = pd.DataFrame.from_dict({"Date": [key for key in date_to_value],
                                     "Value": [date_to_value[key] for key in date_to_value]})
    return result



def calculate_eia_daily_values(start_date: str,
                               end_date: str,
                               eia_start_date: str,
                               eia_end_date: str,
                               normal_start_date: str,
                               normal_end_date: str,
                               current_date: str,
                               eia_monthly_values,
                               component_type: ComponentType,
                               state):
    """
    Calculates the eia daily values.

    The following diagram lays out how I would like the dates to divide time.

    There are 7 dates that are laid out above:
        1. start_date (std)
        2. end_date (etd)
        3. eia_start_date (eia_std)
        4. eia_end_date (eia_etd)
        5. normal_start_date (n_std)
        6. normal_end_date (n_etd)
        7. current_date (c_d)

                                        Timeline
        --------------------------------------------------------------------------------
            |           |       |     |            |      |                            |
        (n_std)     (n_etd)   (std)   |            |    (c_d)                        (etd)
                                      |            |
                                    (eia_std)     (eia_etd)


    Constraints:

        Normalization Dates must begin before the start date (std) for which we want to find the daily
        values. Hence, n_etd >= n_std, and n_etd < std.

        Likewise, EIA values must occur before the current date marker. Hence, eia_etd >= std, and eia_etd < c_d.

        Likewise, c_d <= etd.

    """

    #################################################################
    ################# CHECK PRECONDITIONS ###########################

    preconditions_satisfied, checks = check_preconditions(start_date,
                                                        end_date,
                                                        eia_start_date,
                                                        eia_end_date,
                                                        normal_start_date,
                                                        normal_end_date,
                                                        current_date)

    if not preconditions_satisfied:
        raise ValueError(f"Preconditions not satisfied. Checks are provided by: "
                         f"{checks}")

    #################################################################
    #Begin Weather Calculation
    #Create normal weather.
    weather_normal_values = create_normal_weather_values(normal_start_date,
                                                         normal_end_date,
                                                         state,
                                                         component_type)

    assert(type(weather_normal_values) == dict)

    #Get weather values.
    weather_values = create_weather_values(start_date,
                                           end_date,
                                           current_date,
                                           state,
                                           component_type)

    #Step 1. Calculate consumption factor normal values.
    consumption_factor_diff = calculate_consumption_factor_diff(start_date,
                                                                end_date,
                                                                weather_normal_values,
                                                                weather_values,
                                                                component_type)

    assert("Date" in consumption_factor_diff.columns)

    #################################################################
    #Begin EIA Calculation
    # Step 3: Calculate eia normal values.
    eia_normal_values = calculate_eia_normal_monthly_values(normal_start_date,
                                                          normal_end_date,
                                                          state,
                                                          component_type)

    assert(type(eia_normal_values) == dict)


    #Step 3: Calculate eia monthly values.
    eia_monthly_values = calculate_eia_monthly_values(eia_start_date,
                                                      eia_end_date,
                                                      state,
                                                      component_type)


    eia_monthly_diff = calculate_eia_values_diff(eia_start_date,
                                                eia_end_date,
                                                eia_normal_values,
                                                eia_monthly_values,
                                                 state)

    assert("Date" in eia_monthly_diff.columns)


    #################################################################
    #Calculate sensitivity.
    #Step 3. Calculate sensitivity between eia monthly,
    #and consumption factor values.
    params = calculate_consumption_factor_to_eia_sensitivity_monthly(eia_start_date,
                                                                    eia_end_date,
                                                                    eia_monthly_diff,
                                                                    consumption_factor_diff,
                                                                     state,
                                                                     component_type)


    #################################################################
    #Calculate EIA Daily Values.
    #Step 4: For dates in which no eia monthly exists, apply
    #sensitivity to weather. For dates, in which eia monthly
    #dates exist, form via weather calculation.
    daily_eia_values = calculate_eia_daily_values_with_params(eia_monthly_values,
                                                            eia_normal_values,
                                                            weather_normal_values,
                                                            weather_values,
                                                            start_date,
                                                            end_date,
                                                            eia_start_date,
                                                            eia_end_date,
                                                            normal_start_date,
                                                            normal_end_date,
                                                            current_date,
                                                            params,
                                                            state,
                                                            component_type)

    return daily_eia_values


if __name__ == "__main__":

    daily_values = calculate_eia_daily_values("2023-01-01",
                               "2024-12-31",
                               "2024-01-01",
                               "2024-12-01",
                               "2010-01-01",
                               "2022-12-01",
                               "2024-01-01",
                               "2020-01-01",
                               ComponentType.ELECTRIC,
                               "Virginia")
