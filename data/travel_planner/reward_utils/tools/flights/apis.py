import pandas as pd
from pandas import DataFrame
from typing import Optional
from data.travel_planner.reward_utils.tp_utils.func import extract_before_parenthesis
import os

FILE_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)  # flights/ -> tools/ -> reward_utils/ -> travel_planner/


class Flights:

    def __init__(
        self,
        path=f"{FILE_PATH}/assets/flights.csv",
    ):
        self.path = path
        self.data = None

        self.data = pd.read_csv(self.path)
        print("Flights API loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def run(
        self,
        origin: str,
        destination: str,
        departure_date: str,
    ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = self.data[self.data["OriginCityName"] == origin]
        results = results[results["DestCityName"] == destination]
        results = results[results["FlightDate"] == departure_date]
        if len(results) == 0:
            return "There is no flight from {} to {} on {}.".format(
                origin, destination, departure_date
            )
        return results

    def run_for_annotation(
        self,
        origin: str,
        destination: str,
        departure_date: str,
    ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = self.data[
            self.data["OriginCityName"] == extract_before_parenthesis(origin)
        ]
        results = results[
            results["DestCityName"] == extract_before_parenthesis(destination)
        ]
        results = results[results["FlightDate"] == departure_date]
        return results.to_string(index=False)

    def get_city_set(self):
        city_set = set()
        for unit in self.data["data"]:
            city_set.add(unit[5])
            city_set.add(unit[6])
