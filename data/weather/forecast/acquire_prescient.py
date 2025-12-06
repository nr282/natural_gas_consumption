"""
Acquire Prescient Weather Data via this module.

The module will aim to acquire the Prescient Weather Data.


TODO:
    1. See where the processes are in AWS EC2
    2. Create Forecast Weather Framework for
        1. Subseasonal Data
        2. Mid range Data
    3. Update the model to handle these new weather forecasts.
    4. Check the results for Virginia using 2022, 2023.

"""

import requests


def acquire_prescient_weather():

    bearer_token = "JV55O8qHDHKRr1pReHCexxIAc0DL7DAG"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "accept": "application/json"  # Optional: Include if sending JSON data
    }

    region = "NY"
    response = requests.get(f"https://fastapi.worldclimateservice.com/tm-api/v3/forecast/degreeday/mediumrange/daily/ecmwf00z/popcdd/{region}?climo=10&numfcst=1",
                            headers=headers)


    return response


if __name__ == "__main__":
    acquire_prescient_weather()