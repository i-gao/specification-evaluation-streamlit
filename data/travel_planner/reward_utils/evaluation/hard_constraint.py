import os
import sys
import re
import math
import numpy as np

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tp_utils.func import (
    get_valid_name_city,
    extract_from_to,
    get_tools,
    get_attractions,
)



def get_total_cost(question, tested_data):
    """
    Calculates the total cost of the trip including:
    - Transportation costs (flights, self-driving, taxi) adjusted for number of people
    - Meal costs (breakfast, lunch, dinner) for each person
    - Accommodation costs adjusted for room occupancy limits
    Returns the total cost for the entire trip
    """
    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()
    total_cost = 0
    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]
        # transporation
        if unit["transportation"] and unit["transportation"] != "-":
            value = unit["transportation"]
            org_city, dest_city = extract_from_to(value)
            if org_city == None or dest_city == None:
                org_city, dest_city = extract_from_to(unit["current_city"])

            if org_city == None or dest_city == None:
                pass
            else:
                if "flight number" in value.lower():
                    res = flight.data[
                        flight.data["flight_number"]
                        == value.split("Flight Number: ")[1].split(",")[0]
                    ]
                    if len(res) > 0:
                        total_cost += res["price"].values[0] * question["people_number"]

                elif "self-driving" in value.lower() or "taxi" in value.lower():
                    if "self-driving" in value.lower():
                        cost = googleDistanceMatrix.run_for_evaluation(
                            org_city, dest_city, "self-driving"
                        )["cost"]
                        total_cost += cost * math.ceil(
                            question["people_number"] * 1.0 / 5
                        )
                    else:
                        cost = googleDistanceMatrix.run_for_evaluation(
                            org_city, dest_city, "taxi"
                        )["cost"]
                        total_cost += cost * math.ceil(
                            question["people_number"] * 1.0 / 4
                        )

        # breakfast
        if unit["breakfast"] and unit["breakfast"] != "-":
            name, city = get_valid_name_city(unit["breakfast"])
            res = restaurants.data[
                (restaurants.data["name"].astype(str).str.contains(re.escape(name)))
                & (restaurants.data["city"] == city)
            ]
            if len(res) > 0:
                total_cost += res["average_cost"].values[0] * question["people_number"]

        # lunch
        if unit["lunch"] and unit["lunch"] != "-":
            name, city = get_valid_name_city(unit["lunch"])
            res = restaurants.data[
                (restaurants.data["name"].astype(str).str.contains(re.escape(name)))
                & (restaurants.data["city"] == city)
            ]
            if len(res) > 0:
                total_cost += res["average_cost"].values[0] * question["people_number"]

        # dinner
        if unit["dinner"] and unit["dinner"] != "-":
            name, city = get_valid_name_city(unit["dinner"])
            res = restaurants.data[
                (restaurants.data["name"].astype(str).str.contains(re.escape(name)))
                & (restaurants.data["city"] == city)
            ]
            if len(res) > 0:
                total_cost += res["average_cost"].values[0] * question["people_number"]

        # accommodation
        if unit["accommodation"] and unit["accommodation"] != "-":
            name, city = get_valid_name_city(unit["accommodation"])
            res = accommodation.data[
                (accommodation.data["name"].astype(str).str.contains(re.escape(name)))
                & (accommodation.data["city"] == city)
            ]
            if len(res) > 0:
                total_cost += res["price"].values[0] * math.ceil(
                    question["people_number"] * 1.0 / res["maximum_occupancy"].values[0]
                )
    return total_cost


def is_valid_room_rule(question, tested_data):
    """
    Validates that accommodation choices follow house rules:
    - Checks if specified house rules (smoking, parties, children, visitors, pets)
      are compatible with the accommodation's rules
    - Returns None if no house rules are specified
    - Returns False if any accommodation violates the specified rules
    """
    if question["local_constraint"].get("house rule") is None:
        return None, None

    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]
        if unit["accommodation"] and unit["accommodation"] != "-":
            name, city = get_valid_name_city(unit["accommodation"])
            res = accommodation.data[
                (accommodation.data["name"].astype(str).str.contains(re.escape(name)))
                & (accommodation.data["city"] == city)
            ]
            if len(res) > 0:
                if question["local_constraint"][
                    "house rule"
                ] == "smoking" and "No smoking" in str(res["house_rules"].values[0]):
                    return (
                        False,
                        f"The house rule should be {question['local_constraint']['house rule']}.",
                    )
                if question["local_constraint"][
                    "house rule"
                ] == "parties" and "No parties" in str(res["house_rules"].values[0]):
                    return (
                        False,
                        f"The house rule should be {question['local_constraint']['house rule']}.",
                    )
                if question["local_constraint"][
                    "house rule"
                ] == "children under 10" and "No children under 10" in str(
                    res["house_rules"].values[0]
                ):
                    return (
                        False,
                        f"The house rule should be {question['local_constraint']['house rule']}.",
                    )
                if question["local_constraint"][
                    "house rule"
                ] == "visitors" and "No visitors" in str(res["house_rules"].values[0]):
                    return (
                        False,
                        f"The house rule should be {question['local_constraint']['house rule']}.",
                    )
                if question["local_constraint"][
                    "house rule"
                ] == "pets" and "No pets" in str(res["house_rules"].values[0]):
                    return (
                        False,
                        f"The house rule should be {question['local_constraint']['house rule']}.",
                    )

    return True, None


def is_valid_transportation(question, tested_data):
    """
    Validates that transportation choices follow specified restrictions:
    - Checks if driving is allowed if the plan specifies driving
    - Checks if transportation mode matches/excludes specified preferences
    - Handles restrictions like 'no flight' or 'no self-driving'
    - Returns None if no transportation preferences are specified
    - Returns False if transportation choices violate restrictions
    """
    if question["local_constraint"].get("transportation") is None:
        return None, None
    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]
        if unit["transportation"] and unit["transportation"] != "-":
            value = unit["transportation"]
            if (
                question["local_constraint"]["transportation"] == "no flight"
                and "Flight" in value
            ):
                return (
                    False,
                    f"The transportation should not be {question['local_constraint']['transportation']}.",
                )
            elif (
                question["local_constraint"]["transportation"] == "no self-driving"
                and "Self-driving" in value
            ):
                return (
                    False,
                    f"The transportation should not be {question['local_constraint']['transportation']}.",
                )

    return True, None


def is_valid_room_type(question, tested_data):
    """
    Validates that accommodation choices match room type preferences:
    - Checks if room types (e.g., private room, shared room) match specified preferences
    - Returns None if no room type preferences are specified
    - Returns False if any accommodation violates room type requirements
    """
    if question["local_constraint"].get("room_type") is None:
        return None, None

    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]
        if unit["accommodation"] and unit["accommodation"] != "-":
            name, city = get_valid_name_city(unit["accommodation"])
            res = accommodation.data[
                (accommodation.data["name"].astype(str).str.contains(re.escape(name)))
                & (accommodation.data["city"] == city)
            ]
            if len(res) > 0:
                if (
                    question["local_constraint"]["room_type"] == "not shared room"
                    and res["room_type"].values[0] == "Shared room"
                ):
                    return (
                        False,
                        f"The room type should be {question['local_constraint']['room type']}.",
                    )
                elif (
                    question["local_constraint"]["room_type"] == "shared room"
                    and res["room_type"].values[0] != "Shared room"
                ):
                    return (
                        False,
                        f"The room type should be {question['local_constraint']['room type']}.",
                    )
                elif (
                    question["local_constraint"]["room_type"] == "private room"
                    and res["room_type"].values[0] != "Private room"
                ):
                    return (
                        False,
                        f"The room type should be {question['local_constraint']['room type']}.",
                    )
                elif (
                    question["local_constraint"]["room_type"] == "entire room"
                    and res["room_type"].values[0] != "Entire home/apt"
                ):
                    return (
                        False,
                        f"The room type should be {question['local_constraint']['room type']}.",
                    )

    return True, None


def attractions_per_single_city_day(question, tested_data):
    """
    Validates the minimum and maximum number of attractions per day preference for non-travel days.
    """
    local_constraint = question.get("local_constraint", {})
    min_attractions_per_single_city_day = local_constraint.get(
        "min_attractions_per_single_city_day"
    )
    max_attractions_per_single_city_day = local_constraint.get(
        "max_attractions_per_single_city_day"
    )
    if (
        not min_attractions_per_single_city_day
        and not max_attractions_per_single_city_day
    ):
        return None, None

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]
        if "from" in unit["current_city"] and "to" in unit["current_city"]:
            # if not a single-city day, skip
            continue
        attractions_list = get_attractions(unit)
        if (
            min_attractions_per_single_city_day is not None
            and len(attractions_list) < min_attractions_per_single_city_day
        ):
            return (
                False,
                f"Since day {i} is a single-city day, the number of attractions should be at least {min_attractions_per_single_city_day}.",
            )
        if (
            max_attractions_per_single_city_day is not None
            and len(attractions_list) > max_attractions_per_single_city_day
        ):
            return (
                False,
                f"Since day {i} is a single-city day, the number of attractions should be at most {max_attractions_per_single_city_day}.",
            )
    return True, None


def attractions_per_travel_day(question, tested_data):
    """
    Validates the minimum and maximum number of attractions per day preference for travel days.
    """
    local_constraint = question.get("local_constraint", {})
    min_attractions_per_travel_day = local_constraint.get("min_attractions_per_travel_day")
    max_attractions_per_travel_day = local_constraint.get("max_attractions_per_travel_day")
    if not min_attractions_per_travel_day and not max_attractions_per_travel_day:
        return None, None
    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]
        if not ("from" in unit["current_city"] and "to" in unit["current_city"]):
            continue
        attractions_list = get_attractions(unit)
        if (
            min_attractions_per_travel_day is not None
            and len(attractions_list) < min_attractions_per_travel_day
        ):
            return (
                False,
                f"Since day {i} is a travel day, the number of attractions should be at least {min_attractions_per_travel_day}.",
            )
        if (
            max_attractions_per_travel_day is not None
            and len(attractions_list) > max_attractions_per_travel_day
        ):
            return (
                False,
                f"Since day {i} is a travel day, the number of attractions should be at most {max_attractions_per_travel_day}.",
            )
    return True, None


def evaluation(query_data, tested_data):
    """
    Main evaluation function that runs all hard constraint checks:
    - Validates total cost against budget
    - Validates room rules and types
    - Validates transportation restrictions
    - Validates attractions per day preferences
    Returns a dictionary with results of all constraint checks
    """
    return_info = {}
    return_info["valid_room_rule"] = is_valid_room_rule(query_data, tested_data)
    return_info["valid_transportation"] = is_valid_transportation(
        query_data, tested_data
    )
    return_info["valid_room_type"] = is_valid_room_type(query_data, tested_data)
    return_info["valid_attractions_per_single_city_day"] = (
        attractions_per_single_city_day(query_data, tested_data)
    )
    return_info["valid_attractions_per_travel_day"] = attractions_per_travel_day(
        query_data, tested_data
    )
    try:
        cost = get_total_cost(query_data, tested_data)
        return_info["valid_cost"] = (
            bool(cost <= query_data["budget"]),
            f"The total cost exceeds the budget: current cost is ${cost}.",
        )
    except Exception as e:
        pass
    return return_info
