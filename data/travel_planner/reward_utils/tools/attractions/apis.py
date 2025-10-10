import pandas as pd
from pandas import DataFrame
from typing import Optional
from tp_utils.func import extract_before_parenthesis
import os

FILE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) # attractions/ -> tools/ -> reward_utils/ -> travel_planner/


class Attractions:
    def __init__(
        self,
        path=f"{FILE_PATH}/assets/attractions.csv",
    ):
        self.path = path
        self.data = pd.read_csv(self.path)
        print("Attractions loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def run(
        self,
        city: str,
    ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == city]
        # the results should show the index
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return "There is no attraction in this city."
        return results

    def run_for_annotation(
        self,
        city: str,
    ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == extract_before_parenthesis(city)]
        # the results should show the index
        results = results.reset_index(drop=True)
        return results
