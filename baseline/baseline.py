"""
The following aims to be the baseline technique that I think is good enough to commercialize.

TODO: There are 8 functions that need to be implemented to implement the minimum viable
TODO: prototype.

TODO: What is the key goal here?
TODO: We will also need to put this in the AWS API.
TODO: We should look to keep track of the performance
TODO: characteristics of the API.

"""


def calculate_consumption_factor_diff(start_date,
                                      end_date,
                                      normal_weather,
                                      weather_values):
    pass


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
                          state,
                          component_type):
    pass


def create_normal_weather_values(normal_start_date,
                                 normal_end_date,
                                 state,
                                 component_type: str):
    """
    Creates the normal weather values

    :param start_date:
    :param end_date:
    :param state:
    :return:
    """


    pass

def calculate_consumption_factor_to_eia_sensitivity(start_date,
                                                    end_date,
                                                    eia_monthly_values,
                                                    consumption_factor_values):
    """
    Calculate the eia sensitivity between (a) consumption factor and (b) eia_monthly_value
    between the start_date and end_date.


    """


    pass



def calculate_consumption_factor_to_eia_sensitivity_monthly(start_date,
                                                            end_date,
                                                            eia_monthly_values,
                                                            consumption_factor_values
                                                            ):
    """
    Calculate the consumption factor to eia sensitivity on a month-by-month basis.

    """

    pass


def calculate_eia_monthly_values(start_date,
                                 end_date,
                                 consumption_factor_values,
                                 sensitivity_parameters):
    """
    Calculate monthly eia values based on consumption factor values.

    """

    pass


def calculate_eia_values_diff(start_date,
                                      end_date,
                                      eia_normal,
                                      eia_monthly_values):
    pass

def calculate_eia_daily_values(start_date: str,
                               end_date: str,
                               eia_start_date: str,
                               eia_end_date: str,
                               normal_start_date: str,
                               normal_end_date: str,
                               eia_monthly_values,
                               component_type: str,
                               state,
                               params):
    """
    Calculates the eia daily values.

    """

    #################################################################
    #Begin Weather Calculation
    #Create normal weather.
    weather_normal_values = create_normal_weather_values(normal_start_date,
                                                         normal_end_date,
                                                         state,
                                                         component_type)

    #Get weather values.
    weather_values = create_weather_values(start_date,
                                           end_date,
                                           state,
                                           component_type)

    #Step 1. Calculate consumption factor normal values.
    consumption_factor_diff = calculate_consumption_factor_diff(start_date,
                                                                  end_date,
                                                                  weather_normal_values,
                                                                  weather_values)



    #################################################################
    #Begin EIA Calculation
    # Step 3: Calculate eia normal values.
    eia_normal_values = calculate_eia_monthly_values(normal_start_date,
                                                      normal_end_date,
                                                      state,
                                                      component_type)


    #Step 3: Calculate eia monthly values.
    eia_monthly_values = calculate_eia_monthly_values(eia_start_date,
                                                      eia_end_date,
                                                      state,
                                                      component_type)


    eia_monthly_diff = calculate_eia_values_diff(eia_start_date,
                                                eia_end_date,
                                                eia_normal_values,
                                                eia_monthly_values)


    #################################################################
    #Calculate sensitivity.
    #Step 3. Calculate sensitivity between eia monthly,
    #and consumption factor values.
    params = calculate_consumption_factor_to_eia_sensitivity_monthly(eia_start_date,
                                                                    eia_end_date,
                                                                    eia_monthly_diff,
                                                                    consumption_factor_diff)




    #################################################################
    #Calculate EIA Daily Values.
    #Step 4: For dates in which no eia monthly exists, apply
    #senstivity to weather. For dates, in which eia monthly
    #dates exist, form via weather calculation.
    daily_eia_values = calculate_eia_daily_values(eia_monthly_values,
                                                    eia_normal_values,
                                                    weather_normal_values,
                                                    start_date,
                                                    end_date,
                                                    eia_start_date,
                                                    eia_end_date,
                                                    normal_start_date,
                                                    normal_end_date,
                                                    params)

    return daily_eia_values