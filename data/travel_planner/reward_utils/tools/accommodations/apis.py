import pandas as pd
from pandas import DataFrame
from typing import Optional
from tp_utils.func import extract_before_parenthesis
import os

FILE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) # accomodations/ -> tools/ -> reward_utils/ -> travel_planner/


class Accommodations:
    def __init__(
        self,
        path=f"{FILE_PATH}/assets/accommodations.csv",
    ):
        self.path = path
        self.data = pd.read_csv(self.path)
        print("Accommodations loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def run(
        self,
        city: str,
    ) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == city]
        if len(results) == 0:
            return "There is no attraction in this city."

        return results

    def run_for_annotation(
        self,
        city: str,
    ) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == extract_before_parenthesis(city)]
        return results
