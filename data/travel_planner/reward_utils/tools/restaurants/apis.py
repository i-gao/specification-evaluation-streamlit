import pandas as pd
from pandas import DataFrame
from typing import Optional
from data.travel_planner.reward_utils.tp_utils.func import extract_before_parenthesis
import os

FILE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) # restaurants/ -> tools/ -> reward_utils/ -> travel_planner/

class Restaurants:
    def __init__(
        self,
        path=f"{FILE_PATH}/assets/restaurants.csv",
    ):
        self.path = path
        self.data = pd.read_csv(self.path)
        print("Restaurants loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def run(
        self,
        city: str,
    ) -> DataFrame:
        """Search for restaurant ."""
        results = self.data[self.data["City"] == city]
        if len(results) == 0:
            return "There is no restaurant in this city."
        return results

    def run_for_annotation(
        self,
        city: str,
    ) -> DataFrame:
        """Search for restaurant ."""
        results = self.data[self.data["City"] == extract_before_parenthesis(city)]
        return results
