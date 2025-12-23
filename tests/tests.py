"""
Runs tests to ensure stability of the codebase.

The stability of the codebase is paramount to the success of the projects.

The core elements are:
    1. Unit Tests
    2. Integration Tests
    3. Component Tests

"""


import unittest
from data.weather import test_get_weather
from inference.inference import spectral_inference_engine
import datetime

class TestResidentialConsumption(unittest.TestCase):


    def test_run_inference(self):

        start_time = "2023-01-01"
        end_time = "2024-12-31"
        state = "Virginia"
        component_type = "residential"
        spectral_inference_engine(state,
                                  start_time,
                                  end_time,
                                  component_type,
                                  args={})




if __name__ == '__main__':
    unittest.main()



