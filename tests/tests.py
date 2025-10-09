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

class TestWeather(unittest.TestCase):

    def test_upper(self):
        weather = test_get_weather()
        self.assertTrue("Washington, DC CDD" in weather)


if __name__ == '__main__':
    unittest.main()



