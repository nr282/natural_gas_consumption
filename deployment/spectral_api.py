"""
Module takes in data from EIA, and then runs the variational framework on this.

After running the variational framework on this dataset, we will have a newly
developed dataset.
"""

import pandas as pd
from baseline.baseline import ComponentType, calculate_eia_daily_values
import logging


def parse_lambda_event(event, context):
    """
    Parses lambda event and pass it to a handler for getting the
    results for eia dataset.

    """

    logging.basicConfig(level=logging.INFO)
    logging.info("Running parse_lambda event")

    #Add conditional logic to update s3 bucket.

    headers = event['headers']


    print("Headers")
    print(headers)
    print("Context")
    print(context)


    if (not "start_date" in headers
            or not "end_date" in headers
            or not "state" in headers
            or not "component_type"  in headers):
        raise ValueError(f"start_date and end_date are required headers. The headers provided are: "
                         f"{headers}")

    
    component_types_to_enum = dict()
    component_types_to_enum["residential"] = ComponentType.RESIDENTIAL
    component_types_to_enum["commercial"] = ComponentType.COMMERCIAL
    component_types_to_enum["electric"] = ComponentType.ELECTRIC

    current_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    result, pct_diff = calculate_eia_daily_values(headers["start_date"],
                                        headers["end_date"],
                                        "2021-01-01",
                                        "2025-06-30",
                                        "2016-01-01",
                                        "2020-12-31",
                                        current_date,
                                        component_types_to_enum[headers["component_type"]],
                                        headers["state"])

    import json
    lambda_result = {
                        'statusCode': 200,
                        'headers': {'Content-Type': 'application/json'},
                        'body': result.to_json(orient='records')
                    }

    return lambda_result