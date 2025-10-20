import json
import re
import os
import pandas as pd
from functools import cache


def extract_from_to(text: str):
    """
    Extracts 'A' and 'B' from the format "from A to B" in the given text, with B ending at a comma or the end of the string.

    Args:
    - text (str): The input string.

    Returns:
    - tuple: A tuple containing 'A' and 'B'. If no match is found, returns (None, None).
    """
    pattern = r"[fF]rom\s+(.+?)\s+[tT]o\s+([^,]+)(?=[,\s]|$)"
    matches = re.search(pattern, text)
    return matches.groups() if matches else (None, None)


FILE_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))
)  # func.py -> tp_utils/ -> reward_utils/ -> travel_planner/


@cache
def get_tools():
    from data.travel_planner.reward_utils.tools.flights.apis import Flights
    from data.travel_planner.reward_utils.tools.accommodations.apis import Accommodations
    from data.travel_planner.reward_utils.tools.restaurants.apis import Restaurants
    from data.travel_planner.reward_utils.tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
    from data.travel_planner.reward_utils.tools.attractions.apis import Attractions

    flight = Flights()
    accommodation = Accommodations()
    restaurants = Restaurants()
    googleDistanceMatrix = GoogleDistanceMatrix()
    attractions = Attractions()

    city_state_set = pd.read_csv(f"{FILE_PATH}/assets/city_state.csv")
    city_state_map = {row.city: row.state for row in city_state_set.itertuples()}
    return (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    )


def get_valid_name_city(info):
    # Modified the pattern to preserve spaces at the end of the name
    pattern = r"(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$"
    match = re.search(pattern, info)
    if match:
        return (
            match.group(1).strip(),
            extract_before_parenthesis(match.group(2).strip()).strip(),
        )
    else:
        print(f"{info} can not be parsed, '-' will be used instead.")
        return "-", "-"


def extract_before_parenthesis(s):
    if s is None:
        return None
    match = re.search(r"^(.*?)\([^)]*\)", s)
    return match.group(1) if match else s


def count_consecutive_values(lst):
    if not lst:
        return []

    result = []
    current_string = lst[0]
    count = 1

    for i in range(1, len(lst)):
        if lst[i] == current_string:
            count += 1
        else:
            result.append((current_string, count))
            current_string = lst[i]
            count = 1

    result.append((current_string, count))
    return result


def get_attractions(unit):
    if unit.get("attraction") and unit["attraction"] != "-":
        attractions_list = unit["attraction"].split(";")
        attractions_list = [
            attraction.strip() for attraction in attractions_list if "," in attraction
        ]
        return attractions_list
    return []
