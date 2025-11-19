# Extractor functions for travel planning constraints
# Each function takes a travel_plan and query_data as input and returns a tuple (value, detailed_message)
# where value is the value needed for the constraint and detailed_message describes the result

"""
Assume a travel plan is a list of dictionaries with the following structure:
[
    {
        "days": 1,
        "current_city": "from CityA to CityB" or "CityA",
        "transportation": "Flight Number: ..." or "Self-driving, ..." or "-",
        "breakfast": "Restaurant Name, City" or "-",
        "lunch": "Restaurant Name, City" or "-",
        "dinner": "Restaurant Name, City" or "-",
        "attraction": "Attraction1, City;Attraction2, City;" or "-",
        "accommodation": "Hotel Name, City" or "-",
    },
    ...
]
"""

import re
from collections import Counter
from datetime import datetime
from data.travel_planner.reward_utils.tp_utils.func import (
    get_valid_name_city,
    extract_from_to,
    extract_before_parenthesis,
    get_tools,
    get_attractions,
)
from data.travel_planner.reward_utils.evaluation.commonsense_constraint import (
    is_valid_accommodaton,
    normalize_string,
    transportation_match,
)
from data.travel_planner.reward_utils.evaluation.hard_constraint import (
    get_total_cost,
)


# ============================================================================
# Hard Constraint Extractors
# ============================================================================


def check_first_city_is_origin(travel_plan, **kwargs):
    """
    Check that first day's origin city matches query_data['org'].
    Returns (bool, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    if len(travel_plan) == 0:
        return False, "Travel plan is empty"
    city_value = travel_plan[0]["current_city"]
    if "from" in city_value:
        city1, city2 = extract_from_to(city_value)
        city1 = extract_before_parenthesis(city1)
        if city1 != query_data["org"]:
            return False, f"The first day's city should be {query_data['org']}."
    else:
        city = extract_before_parenthesis(city_value)
        if city != query_data["org"]:
            return False, f"The first day's city should be {query_data['org']}."
    return True, None


def check_closed_circle(travel_plan, **kwargs):
    """
    Check that trip forms a closed circle (starts and ends at same city).
    Returns (bool, str): (success, detailed_message)
    """
    if len(travel_plan) == 0:
        return False, "Travel plan is empty"
    
    # Get first city
    first_city_value = travel_plan[0]["current_city"]
    if "from" in first_city_value:
        first_city, _ = extract_from_to(first_city_value)
        first_city = extract_before_parenthesis(first_city)
    else:
        first_city = extract_before_parenthesis(first_city_value)
    
    # Get last city
    last_city_value = travel_plan[-1]["current_city"]
    if "from" in last_city_value:
        _, last_city = extract_from_to(last_city_value)
        last_city = extract_before_parenthesis(last_city)
    else:
        last_city = extract_before_parenthesis(last_city_value)
    
    if first_city != last_city:
        return False, "The trip should be a closed circle: the origin of the first day and the destination of the last day should be the same."
    return True, None


def check_cities_in_map(travel_plan, **kwargs):
    """
    Check that all cities exist in city_state_map.
    Returns (bool, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    (flight, accommodation, restaurants, googleDistanceMatrix, attractions, city_state_map) = get_tools()
    
    city_list = []
    for i in range(min(query_data["days"], len(travel_plan))):
        city_value = travel_plan[i]["current_city"]
        if "from" in city_value:
            city1, city2 = extract_from_to(city_value)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            city_list.extend([city1, city2])
        else:
            city_list.append(extract_before_parenthesis(city_value))
    
    for city in city_list:
        if city not in city_state_map:
            return False, f"{city} is not a valid city."
    return True, None


def check_intermediate_cities_in_destination_state(travel_plan, **kwargs):
    """
    Check that intermediate cities (for trips > 3 days) are in destination state.
    Returns (bool, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    if query_data["days"] <= 3:
        return True, None  # Not applicable
    
    (flight, accommodation, restaurants, googleDistanceMatrix, attractions, city_state_map) = get_tools()
    
    city_list = []
    for i in range(min(query_data["days"], len(travel_plan))):
        city_value = travel_plan[i]["current_city"]
        if "from" in city_value:
            city1, city2 = extract_from_to(city_value)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            city_list.extend([city1, city2])
        else:
            city_list.append(extract_before_parenthesis(city_value))
    
    for idx, city in enumerate(city_list):
        if idx not in [0, len(city_list) - 1] and city_state_map.get(city) != query_data["dest"]:
            return False, f"{city} is not in {query_data['dest']}."
    return True, None


def check_day_field_in_correct_city(travel_plan, **kwargs):
    """
    Check that a specific field for a specific day is in the correct city.
    Returns (bool, str): (success, detailed_message)
    """
    day_idx = kwargs["day_idx"]
    field = kwargs["field"]
    if day_idx >= len(travel_plan):
        return True, None  # Day doesn't exist
    
    unit = travel_plan[day_idx]
    current_city = unit["current_city"]
    final_city_list = []
    
    if "from" in current_city:
        city1, city2 = extract_from_to(current_city)
        city1 = extract_before_parenthesis(city1)
        city2 = extract_before_parenthesis(city2)
        final_city_list = [city1, city2]
    else:
        final_city_list = [extract_before_parenthesis(current_city)]
    
    if field not in unit or not unit[field] or unit[field] == "-":
        return True, None  # Field is empty, skip check
    
    if field == "transportation":
        for city in final_city_list:
            if city not in unit[field]:
                return False, f"The transportation in day {day_idx+1} is not in a valid city from the `current_city` field."
    else:
        # For meals, attractions, accommodation
        flag = False
        for city in final_city_list:
            if city in unit[field]:
                flag = True
                break
        if not flag:
            return False, f"The {field} in day {day_idx+1} is not in a valid city from the `current_city` field."
    
    return True, None


def check_day_field_exists_in_sandbox(travel_plan, **kwargs):
    """
    Check that a specific field for a specific day exists in the sandbox database.
    Returns (bool, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    day_idx = kwargs["day_idx"]
    field = kwargs["field"]
    if day_idx >= len(travel_plan):
        return True, None
    
    unit = travel_plan[day_idx]
    if field not in unit or not unit[field] or unit[field] == "-":
        return True, None  # Field is empty, skip check
    
    (flight, accommodation, restaurants, googleDistanceMatrix, attractions, city_state_map) = get_tools()
    
    if isinstance(query_data["date"], str):
        unit_date = eval(query_data["date"])[day_idx]
    else:
        unit_date = query_data["date"][day_idx]
    
    if field == "transportation":
        value = unit[field]
        org_city, dest_city = extract_from_to(value)
        if org_city is None or dest_city is None:
            org_city, dest_city = extract_from_to(unit["current_city"])
        
        if "flight number" in value.lower():
            try:
                org_city = extract_before_parenthesis(org_city)
                dest_city = extract_before_parenthesis(dest_city)
                flight_num = value.split("Flight Number: ")[1].split(",")[0]
                if len(flight.data[
                    (flight.data["flight_number"] == flight_num)
                    & (flight.data["departure_city"] == org_city)
                    & (flight.data["arrival_city"] == dest_city)
                    & (flight.data["flight_date"] == unit_date)
                ]) < 1:
                    return False, f"The flight in day {day_idx+1} is not a valid flight from the database."
            except Exception:
                return False, f"The flight in day {day_idx+1} cannot be parsed."
        elif "self-driving" in value.lower():
            org_city = extract_before_parenthesis(org_city)
            dest_city = extract_before_parenthesis(dest_city)
            if googleDistanceMatrix.run_for_evaluation(org_city, dest_city, mode="self-driving")["cost"] is None:
                return False, f"The self-driving in day {day_idx+1} is not a valid route."
        elif "taxi" in value.lower():
            org_city = extract_before_parenthesis(org_city)
            dest_city = extract_before_parenthesis(dest_city)
            if googleDistanceMatrix.run_for_evaluation(org_city, dest_city, mode="taxi")["cost"] is None:
                return False, f"The taxi in day {day_idx+1} is not a valid route."
    
    elif field in ["breakfast", "lunch", "dinner"]:
        name, city = get_valid_name_city(unit[field])
        if len(restaurants.data[
            (restaurants.data["name"].astype(str).apply(normalize_string).str.contains(normalize_string(name), regex=False))
            & (restaurants.data["city"] == city)
        ]) < 1:
            return False, f"The {field} in day {day_idx+1} is not a valid restaurant."
    
    elif field == "attraction":
        attractions_list = unit[field].split(";")
        attractions_list = [a.strip() for a in attractions_list if a.strip()]
        for attraction in attractions_list:
            name, city = get_valid_name_city(attraction)
            if len(attractions.data[
                (attractions.data["name"].astype(str).apply(normalize_string).str.contains(normalize_string(name), regex=False))
                & (attractions.data["city"] == city)
            ]) < 1:
                return False, f"The attraction {attraction} in day {day_idx+1} is not a valid attraction."
    
    elif field == "accommodation":
        name, city = get_valid_name_city(unit[field])
        if len(accommodation.data[
            (accommodation.data["name"].astype(str).apply(normalize_string).str.contains(normalize_string(name), regex=False))
            & (accommodation.data["city"] == city)
        ]) < 1:
            return False, f"The accommodation in day {day_idx+1} is not a valid accommodation."
    
    return True, None


def check_day_field_present(travel_plan, **kwargs):
    """
    Check that a required field is present for a specific day.
    Returns (bool, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    day_idx = kwargs["day_idx"]
    field = kwargs["field"]
    if day_idx >= len(travel_plan):
        return False, f"Day {day_idx+1} is missing from travel plan"
    
    unit = travel_plan[day_idx]
    if field not in unit:
        return False, f"No {field} info for day {day_idx+1}."
    
    # Additional checks based on field and day type
    if field == "transportation":
        if ("from " in unit.get("current_city", "") or "to " in unit.get("current_city", "")) and unit[field] in ["", "-"]:
            return False, f"No transportation in day {day_idx+1} is not allowed."
    
    if field == "attraction":
        if ("from " not in unit.get("current_city", "") and " to " not in unit.get("current_city", "")) and unit[field] in ["", "-"]:
            return False, f"No attraction in day {day_idx+1} is not allowed."
    
    if field == "accommodation":
        if day_idx != query_data["days"] - 1 and unit[field] in ["", "-"]:
            return False, f"No accommodation in day {day_idx+1} is not allowed."
    
    if field in ["breakfast", "lunch", "dinner"]:
        if "from " not in unit.get("current_city", "") and unit[field] in ["", "-"]:
            return False, f"No meal in day {day_idx+1} is not allowed."
    
    return True, None


def check_accommodation_minimum_nights(travel_plan, **kwargs):
    """
    Check that accommodations are booked for minimum required nights.
    Returns (bool, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    result = is_valid_accommodaton(query_data, travel_plan)
    return result


def check_budget(travel_plan, **kwargs):
    """
    Check that total cost is within budget.
    Returns (bool, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    try:
        cost = get_total_cost(query_data, travel_plan)
        if cost > query_data["budget"]:
            return False, f"The total cost exceeds the budget: current cost is ${cost}."
        return True, None
    except Exception as e:
        return False, f"Error calculating cost: {str(e)}"


def check_transportation_first_day(travel_plan, **kwargs):
    """
    Validates that transportation choices follow basic constraints:
    - First day must have transportation specified
    Returns (bool, str): (success, detailed_message)
    """
    if len(travel_plan) == 0:
        return False, "The trip should have at least one day."

    if travel_plan[0]["transportation"] and travel_plan[0]["transportation"] != "-":
        transportation_match(travel_plan[0]["transportation"])  # Validate transportation type
    else:
        return False, "The transportation in day 1 should not be empty."

    return True, None


def check_valid_days(travel_plan, **kwargs):
    """
    Validates that the number of days matches the required number:
    - Counts only days with valid city information
    - Must match the specified number of days
    Returns (bool, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    lens = 0
    for i in range(min(query_data["days"], len(travel_plan))):
        if (
            travel_plan[i] != {}
            and travel_plan[i]["current_city"]
            != "You don't need to fill in the information for this or later days."
        ):
            lens += 1

    if lens != query_data["days"]:
        return False, f"The number of days should be {query_data['days']}."
    else:
        return True, None


def check_visiting_city_number(travel_plan, **kwargs):
    """
    Validates that the number of unique cities visited matches the required number:
    - Counts unique cities excluding the origin city
    - Must match the specified visiting_city_number
    Returns (bool, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    city_set = set()

    for i in range(min(query_data["days"], len(travel_plan))):
        city_value = travel_plan[i]["current_city"]

        if "from" in city_value:
            city1, city2 = extract_from_to(city_value)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            if i == 0 and city1 != query_data["org"]:
                return False, f"The first day's city should be {query_data['org']}."

            city_set.add(city1)
            city_set.add(city2)
        else:
            city_set.add(extract_before_parenthesis(city_value))

    city_set.discard(query_data["org"])

    if len(city_set) != query_data["visiting_city_number"]:
        return (
            False,
            f"The number of visiting cities should be {query_data['visiting_city_number']}.",
        )

    return True, None


def check_meals_match_travel_time(travel_plan, **kwargs):
    """
    Validates that the meals of the first / last day reflects the travel time.
    For example, if the flight on day 1 to the location arrives at 5p, breakfast and
    lunch should be empty in the plan.
    Returns (bool, str): (success, detailed_message)
    """
    if len(travel_plan) == 0:
        return False, "The trip should have at least one day."

    # Check first day
    unit = travel_plan[0]
    city1, city2 = extract_from_to(unit["current_city"])
    city1 = extract_before_parenthesis(city1)
    city2 = extract_before_parenthesis(city2)
    if (
        unit["transportation"]
        and unit["transportation"] != "-"
        and "Arrival Time" in unit["transportation"]
    ):
        # Extract arrival time more robustly - handle format "Arrival Time: HH:MM, ..." or "Arrival Time: HH:MM"
        arrival_time_str = unit["transportation"].split("Arrival Time: ")[1]
        # Take only the time part (before comma or end of string)
        arrival_time_str = arrival_time_str.split(",")[0].strip()
        arrival_time_str = arrival_time_str.replace(" ", "")
        arrival_time = datetime.strptime(arrival_time_str, "%H:%M")
        # If we arrive after 12p, breakfast should be empty
        if arrival_time.hour >= 12:
            if (
                unit["breakfast"]
                and unit["breakfast"] != "-"
                and get_valid_name_city(unit["breakfast"])[1] == city2
            ):
                return (
                    False,
                    f"We arrive in {city2} after 12p, so we should not have breakfast there.",
                )

        # If we arrive after 3p, lunch should be empty
        if arrival_time.hour >= 15:
            if (
                unit["lunch"]
                and unit["lunch"] != "-"
                and get_valid_name_city(unit["lunch"])[1] == city2
            ):
                return (
                    False,
                    f"We arrive in {city2} after 3p, so we should not have lunch there.",
                )

        # If we arrive after 11p, dinner should be empty
        if arrival_time.hour >= 23:
            if (
                unit["dinner"]
                and unit["dinner"] != "-"
                and get_valid_name_city(unit["dinner"])[1] == city2
            ):
                return (
                    False,
                    f"We arrive in {city2} after 11p, so we should not have dinner there.",
                )

    if (
        unit["transportation"]
        and unit["transportation"] != "-"
        and "duration" in unit["transportation"]
    ):
        duration = unit["transportation"].split("duration: ")[1].split(",")[0]
        duration_hours = int(duration.split("hour")[0])
        # a reasonable leaving time is like 6a
        # if we arrive after 12p, breakfast should be empty
        # this means duration should be < 6 hours
        if duration_hours >= 6:
            if (
                unit["breakfast"]
                and unit["breakfast"] != "-"
                and get_valid_name_city(unit["breakfast"])[1] == city2
            ):
                return (
                    False,
                    f"We arrive in {city2} after 12p, so we should not have breakfast there.",
                )
        # if we arrive after 3p, lunch should be empty
        # this means duration should be < 9 hours
        if duration_hours >= 9:
            if (
                unit["lunch"]
                and unit["lunch"] != "-"
                and get_valid_name_city(unit["lunch"])[1] == city2
            ):
                return (
                    False,
                    f"We arrive in {city2} after 3p, so we should not have lunch there.",
                )

    # Check last day
    unit = travel_plan[-1]
    city1, city2 = extract_from_to(unit["current_city"])
    city1 = extract_before_parenthesis(city1)
    city2 = extract_before_parenthesis(city2)
    if (
        unit["transportation"]
        and unit["transportation"] != "-"
        and "Departure Time" in unit["transportation"]
    ):
        # Extract departure time more robustly - handle format "Departure Time: HH:MM, Arrival Time: ..." or "Departure Time: HH:MM"
        departure_time_str = unit["transportation"].split("Departure Time: ")[1]
        # Take only the time part (before comma or end of string)
        departure_time_str = departure_time_str.split(",")[0].strip()
        departure_time_str = departure_time_str.replace(" ", "")
        departure_time = datetime.strptime(departure_time_str, "%H:%M")
        # If we leave before 9a, breakfast should be empty
        if departure_time.hour < 9:
            if (
                unit["breakfast"]
                and unit["breakfast"] != "-"
                and get_valid_name_city(unit["breakfast"])[1] == city1
            ):
                return (
                    False,
                    f"We leave {city1} before 9a, so we should not have breakfast there.",
                )

        # If we leave before 12p, lunch should be empty
        if departure_time.hour < 12:
            if (
                unit["lunch"]
                and unit["lunch"] != "-"
                and get_valid_name_city(unit["lunch"])[1] == city1
            ):
                return (
                    False,
                    f"We leave {city1} before 12p, so we should not have lunch there.",
                )

        # If we leave before 5p, dinner should be empty
        if departure_time.hour < 17:
            if (
                unit["dinner"]
                and unit["dinner"] != "-"
                and get_valid_name_city(unit["dinner"])[1] == city1
            ):
                return (
                    False,
                    f"We leave {city1} before 5p, so we should not have dinner there.",
                )

    if (
        unit["transportation"]
        and unit["transportation"] != "-"
        and "duration" in unit["transportation"]
    ):
        duration = unit["transportation"].split("duration: ")[1].split(",")[0]
        duration_hours = int(duration.split("hour")[0])
        # a reasonable leaving time is like 6a
        # if we leave before 9a, breakfast should be empty
        # this means duration should be < 3 hours
        if duration_hours < 3:
            if (
                unit["breakfast"]
                and unit["breakfast"] != "-"
                and get_valid_name_city(unit["breakfast"])[1] == city1
            ):
                return (
                    False,
                    f"We leave {city1} before 9a, so we should not have breakfast there.",
                )
        # if we leave before 12p, lunch should be empty
        # this means duration should be < 6 hours
        if duration_hours < 6:
            if (
                unit["lunch"]
                and unit["lunch"] != "-"
                and get_valid_name_city(unit["lunch"])[1] == city1
            ):
                return (
                    False,
                    f"We leave {city1} before 12p, so we should not have lunch there.",
                )

    return True, None


def check_room_rule(travel_plan, **kwargs):
    """
    Validates that accommodation choices follow house rules:
    - Checks if specified house rules (smoking, parties, children, visitors, pets)
      are compatible with the accommodation's rules
    - Returns None if no house rules are specified
    - Returns False if any accommodation violates the specified rules
    Returns (bool|None, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    if query_data["local_constraint"].get("house rule") is None:
        return None, None

    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()

    for i in range(min(query_data["days"], len(travel_plan))):
        unit = travel_plan[i]
        if unit["accommodation"] and unit["accommodation"] != "-":
            name, city = get_valid_name_city(unit["accommodation"])
            res = accommodation.data[
                (accommodation.data["name"].astype(str).str.contains(re.escape(name)))
                & (accommodation.data["city"] == city)
            ]
            if len(res) > 0:
                if query_data["local_constraint"][
                    "house rule"
                ] == "smoking" and "No smoking" in str(res["house_rules"].values[0]):
                    return (
                        False,
                        f"The house rule should be {query_data['local_constraint']['house rule']}.",
                    )
                if query_data["local_constraint"][
                    "house rule"
                ] == "parties" and "No parties" in str(res["house_rules"].values[0]):
                    return (
                        False,
                        f"The house rule should be {query_data['local_constraint']['house rule']}.",
                    )
                if query_data["local_constraint"][
                    "house rule"
                ] == "children under 10" and "No children under 10" in str(
                    res["house_rules"].values[0]
                ):
                    return (
                        False,
                        f"The house rule should be {query_data['local_constraint']['house rule']}.",
                    )
                if query_data["local_constraint"][
                    "house rule"
                ] == "visitors" and "No visitors" in str(res["house_rules"].values[0]):
                    return (
                        False,
                        f"The house rule should be {query_data['local_constraint']['house rule']}.",
                    )
                if query_data["local_constraint"][
                    "house rule"
                ] == "pets" and "No pets" in str(res["house_rules"].values[0]):
                    return (
                        False,
                        f"The house rule should be {query_data['local_constraint']['house rule']}.",
                    )

    return True, None


def check_transportation_restriction(travel_plan, **kwargs):
    """
    Validates that transportation choices follow specified restrictions:
    - Checks if driving is allowed if the plan specifies driving
    - Checks if transportation mode matches/excludes specified preferences
    - Handles restrictions like 'no flight' or 'no self-driving'
    - Returns None if no transportation preferences are specified
    - Returns False if transportation choices violate restrictions
    Returns (bool|None, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    if query_data["local_constraint"].get("transportation") is None:
        return None, None
    for i in range(min(query_data["days"], len(travel_plan))):
        unit = travel_plan[i]
        if unit["transportation"] and unit["transportation"] != "-":
            value = unit["transportation"]
            if (
                query_data["local_constraint"]["transportation"] == "no flight"
                and "Flight" in value
            ):
                return (
                    False,
                    f"The transportation should not be {query_data['local_constraint']['transportation']}.",
                )
            elif (
                query_data["local_constraint"]["transportation"] == "no self-driving"
                and "Self-driving" in value
            ):
                return (
                    False,
                    f"The transportation should not be {query_data['local_constraint']['transportation']}.",
                )

    return True, None


def check_room_type(travel_plan, **kwargs):
    """
    Validates that accommodation choices match room type preferences:
    - Checks if room types (e.g., private room, shared room) match specified preferences
    - Returns None if no room type preferences are specified
    - Returns False if any accommodation violates room type requirements
    Returns (bool|None, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    if query_data["local_constraint"].get("room_type") is None:
        return None, None

    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()

    for i in range(min(query_data["days"], len(travel_plan))):
        unit = travel_plan[i]
        if unit["accommodation"] and unit["accommodation"] != "-":
            name, city = get_valid_name_city(unit["accommodation"])
            res = accommodation.data[
                (accommodation.data["name"].astype(str).str.contains(re.escape(name)))
                & (accommodation.data["city"] == city)
            ]
            if len(res) > 0:
                if (
                    query_data["local_constraint"]["room_type"] == "not shared room"
                    and res["room_type"].values[0] == "Shared room"
                ):
                    return (
                        False,
                        f"The room type should be {query_data['local_constraint']['room type']}.",
                    )
                elif (
                    query_data["local_constraint"]["room_type"] == "shared room"
                    and res["room_type"].values[0] != "Shared room"
                ):
                    return (
                        False,
                        f"The room type should be {query_data['local_constraint']['room type']}.",
                    )
                elif (
                    query_data["local_constraint"]["room_type"] == "private room"
                    and res["room_type"].values[0] != "Private room"
                ):
                    return (
                        False,
                        f"The room type should be {query_data['local_constraint']['room type']}.",
                    )
                elif (
                    query_data["local_constraint"]["room_type"] == "entire room"
                    and res["room_type"].values[0] != "Entire home/apt"
                ):
                    return (
                        False,
                        f"The room type should be {query_data['local_constraint']['room type']}.",
                    )

    return True, None


def check_attractions_per_single_city_day(travel_plan, **kwargs):
    """
    Validates the minimum and maximum number of attractions per day preference for non-travel days.
    Returns (bool|None, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    local_constraint = query_data.get("local_constraint", {})
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

    for i in range(min(query_data["days"], len(travel_plan))):
        unit = travel_plan[i]
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


def check_attractions_per_travel_day(travel_plan, **kwargs):
    """
    Validates the minimum and maximum number of attractions per day preference for travel days.
    Returns (bool|None, str): (success, detailed_message)
    """
    query_data = kwargs["query_data"]
    local_constraint = query_data.get("local_constraint", {})
    min_attractions_per_travel_day = local_constraint.get("min_attractions_per_travel_day")
    max_attractions_per_travel_day = local_constraint.get("max_attractions_per_travel_day")
    if not min_attractions_per_travel_day and not max_attractions_per_travel_day:
        return None, None
    for i in range(min(query_data["days"], len(travel_plan))):
        unit = travel_plan[i]
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


def check_all_days_fields_in_correct_city(travel_plan, **kwargs):
    """Check that all fields for all days are in the correct city."""
    query_data = kwargs["query_data"]
    days = query_data.get("days", 1)
    fields = ["transportation", "breakfast", "lunch", "dinner", "attraction", "accommodation"]
    
    for day_idx in range(days):
        for field in fields:
            result = check_day_field_in_correct_city(travel_plan, day_idx=day_idx, field=field, **kwargs)
            if not result[0]:  # If any check fails, return False
                return False, result[1]  # Return the first failure message
    return True, None


def check_all_days_fields_exist_in_sandbox(travel_plan, **kwargs):
    """Check that all fields for all days exist in the sandbox database."""
    query_data = kwargs["query_data"]
    days = query_data.get("days", 1)
    fields = ["transportation", "breakfast", "lunch", "dinner", "attraction", "accommodation"]
    
    for day_idx in range(days):
        for field in fields:
            result = check_day_field_exists_in_sandbox(travel_plan, day_idx=day_idx, field=field, **kwargs)
            if not result[0]:  # If any check fails, return False
                return False, result[1]  # Return the first failure message
    return True, None


def check_all_days_fields_present(travel_plan, **kwargs):
    """Check that all required fields for all days are present."""
    query_data = kwargs["query_data"]
    days = query_data.get("days", 1)
    fields = ["transportation", "breakfast", "lunch", "dinner", "attraction", "accommodation"]
    
    for day_idx in range(days):
        for field in fields:
            result = check_day_field_present(travel_plan, day_idx=day_idx, field=field, **kwargs)
            if not result[0]:  # If any check fails, return False
                return False, result[1]  # Return the first failure message
    return True, None


# ============================================================================
# Preference Extractors
# ============================================================================


def extract_all_restaurant_tags(travel_plan, **kwargs):
    """
    Extract all restaurant tags from all meals in the travel plan.
    Returns (list, str): (list of all tags found, message)
    """
    query_data = kwargs["query_data"]
    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()
    
    all_tags = []
    
    def get_tags(meal_field):
        if not meal_field or meal_field == "-":
            return []
        name, city = get_valid_name_city(meal_field)
        res = restaurants.data[
            (restaurants.data["name"].astype(str).str.contains(re.escape(name)))
            & (restaurants.data["city"] == city)
        ]
        if len(res) > 0:
            tags = res.iloc[0]["tags"]
            if isinstance(tags, str):
                # Tags might be a string representation of a list
                import ast
                try:
                    tags = ast.literal_eval(tags)
                except (ValueError, SyntaxError):
                    tags = [tags]
            if isinstance(tags, list):
                return tags
            return []
        return []
    
    for i in range(min(query_data.get("days", 1), len(travel_plan))):
        unit = travel_plan[i]
        for meal in ["breakfast", "lunch", "dinner"]:
            if unit.get(meal) and unit[meal] != "-":
                tags = get_tags(unit[meal])
                all_tags.extend(tags)
    
    return all_tags, f"Actual tags: {Counter(all_tags)}"

def extract_all_room_types(travel_plan, **kwargs):
    """
    Extract all room types from all accommodations in the travel plan.
    Returns (list, str): (list of all room types found, message)
    """
    query_data = kwargs["query_data"]
    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()
    
    all_room_types = []
    
    def get_room_type(accommodation_field):
        if not accommodation_field or accommodation_field == "-":
            return None
        name, city = get_valid_name_city(accommodation_field)
        res = accommodation.data[
            (accommodation.data["name"].astype(str).str.contains(re.escape(name)))
            & (accommodation.data["city"] == city)
        ]
        if len(res) > 0:
            return res.iloc[0]["room_type"]
        return None
    
    for i in range(min(query_data.get("days", 1) - 1, len(travel_plan))):
        unit = travel_plan[i]
        if unit.get("accommodation") and unit["accommodation"] != "-":
            room_type = get_room_type(unit["accommodation"])
            if room_type:
                all_room_types.append(room_type)
    
    return all_room_types, f"Found {len(all_room_types)} room types across all accommodations"


def extract_all_attraction_types(travel_plan, **kwargs):
    """
    Extract all attraction types from all attractions in the travel plan.
    Returns (list, str): (list of all attraction types found, message)
    """
    query_data = kwargs["query_data"]
    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()
    
    all_attraction_types = []
    
    def get_attraction_types(attraction_name, city):
        res = attractions.data[
            (attractions.data["name"].astype(str).str.contains(re.escape(attraction_name)))
            & (attractions.data["city"] == city)
        ]
        if len(res) > 0:
            attr_type_str = res.iloc[0]["attraction_type"]
            if isinstance(attr_type_str, str):
                try:
                    attr_types = eval(attr_type_str)
                    if isinstance(attr_types, list):
                        return attr_types
                    elif attr_types:
                        return [attr_types]
                except (ValueError, SyntaxError, NameError):
                    return []
            elif isinstance(attr_type_str, list):
                return attr_type_str
        return []
    
    for i in range(min(query_data.get("days", 1), len(travel_plan))):
        unit = travel_plan[i]
        attractions_list = get_attractions(unit)
        for attraction_str in attractions_list:
            if attraction_str and attraction_str != "-":
                name, city = get_valid_name_city(attraction_str)
                attr_types = get_attraction_types(name, city)
                all_attraction_types.extend(attr_types)
    
    return all_attraction_types, f"Found {len(all_attraction_types)} attraction types across all attractions"


def extract_all_restaurant_names(travel_plan, **kwargs):
    """
    Extract all restaurant names from all meals in the travel plan.
    Returns (list, str): (list of all restaurant names found, message)
    """
    query_data = kwargs["query_data"]
    all_restaurant_names = []
    
    for i in range(min(query_data.get("days", 1), len(travel_plan))):
        unit = travel_plan[i]
        for meal in ["breakfast", "lunch", "dinner"]:
            if unit.get(meal) and unit[meal] != "-":
                name, city = get_valid_name_city(unit[meal])
                all_restaurant_names.append(name)
    
    return all_restaurant_names, f"Found {len(all_restaurant_names)} restaurant names across all meals"


def extract_all_accommodation_names(travel_plan, **kwargs):
    """
    Extract all accommodation names from all days in the travel plan.
    Returns (list, str): (list of all accommodation names found, message)
    """
    query_data = kwargs["query_data"]
    all_accommodation_names = []
    
    for i in range(min(query_data.get("days", 1) - 1, len(travel_plan))):
        unit = travel_plan[i]
        if unit.get("accommodation") and unit["accommodation"] != "-":
            name, city = get_valid_name_city(unit["accommodation"])
            all_accommodation_names.append(name)
    
    return all_accommodation_names, f"Found {len(all_accommodation_names)} accommodation names across all days"


def extract_all_attraction_names(travel_plan, **kwargs):
    """
    Extract all attraction names from all attractions in the travel plan.
    Returns (list, str): (list of all attraction names found, message)
    """
    query_data = kwargs["query_data"]
    all_attraction_names = []
    
    for i in range(min(query_data.get("days", 1), len(travel_plan))):
        unit = travel_plan[i]
        attractions_list = get_attractions(unit)
        for attraction_str in attractions_list:
            if attraction_str and attraction_str != "-":
                name, city = get_valid_name_city(attraction_str)
                all_attraction_names.append(name)
    
    return all_attraction_names, f"Found {len(all_attraction_names)} attraction names across all attractions"


def extract_all_restaurant_attributes(travel_plan, **kwargs):
    """
    Extract all restaurant attributes from all meals in the travel plan.
    Returns (list, str): (list of all restaurant attributes found, message)
    """
    query_data = kwargs["query_data"]
    preferences = query_data.get("preferences", {})
    liked_attributes = preferences.get("liked_attributes", [])
    disliked_attributes = preferences.get("disliked_attributes", [])
    
    if not liked_attributes and not disliked_attributes:
        return [], "No restaurant attribute preferences specified"
    
    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()
    
    all_attributes = []
    
    def get_attributes(meal_field):
        if not meal_field or meal_field == "-":
            return []
        name, city = get_valid_name_city(meal_field)
        res = restaurants.data[
            (restaurants.data["name"].astype(str).str.contains(re.escape(name)))
            & (restaurants.data["city"] == city)
        ]
        if len(res) > 0:
            # Get all attribute columns
            attr_cols = [col for col in restaurants.data.columns if col.startswith("attributes_")]
            found_attrs = []
            for col in attr_cols:
                if res.iloc[0][col] == 1:
                    attr_name = col.replace("attributes_", "").replace("_", " ")
                    found_attrs.append(attr_name)
            return found_attrs
        return []
    
    for i in range(min(query_data.get("days", 1), len(travel_plan))):
        unit = travel_plan[i]
        for meal in ["breakfast", "lunch", "dinner"]:
            if unit.get(meal) and unit[meal] != "-":
                attrs = get_attributes(unit[meal])
                all_attributes.extend(attrs)
    
    return all_attributes, f"Found {len(all_attributes)} restaurant attributes across all meals"


def extract_restaurant_repeat_flags(travel_plan, **kwargs):
    """
    Extract a list of booleans indicating whether each meal slot is a repeat.
    Returns (list, str): (list of booleans, one per meal slot, True if repeat, False otherwise)
    """
    query_data = kwargs["query_data"]
    restaurants_seen = {}  # {restaurant_name: first_occurrence}
    repeat_flags = []
    
    for i in range(min(query_data.get("days", 1), len(travel_plan))):
        unit = travel_plan[i]
        for meal in ["breakfast", "lunch", "dinner"]:
            if unit.get(meal) and unit[meal] != "-":
                name, city = get_valid_name_city(unit[meal])
                if name not in restaurants_seen:
                    restaurants_seen[name] = (i, meal)
                    repeat_flags.append(False)  # First occurrence, not a repeat
                else:
                    repeat_flags.append(True)  # Repeat
            else:
                repeat_flags.append(False)  # No restaurant, not a repeat
    
    num_repeats = sum(repeat_flags)
    message = f"Found {num_repeats} repeated restaurants out of {len(repeat_flags)} meal slots"
    return repeat_flags, message


def extract_attraction_repeat_flags(travel_plan, **kwargs):
    """
    Extract a list of booleans indicating whether each attraction slot is a repeat.
    Returns (list, str): (list of booleans, one per attraction slot, True if repeat, False otherwise)
    """
    query_data = kwargs["query_data"]
    attractions_seen = {}  # {attraction_name: first_occurrence}
    repeat_flags = []
    
    for i in range(min(query_data.get("days", 1), len(travel_plan))):
        unit = travel_plan[i]
        attractions_list = get_attractions(unit)
        for attraction_ix in range(5):  # Up to 5 attractions per day
            if attraction_ix < len(attractions_list):
                attraction_str = attractions_list[attraction_ix]
                if attraction_str and attraction_str != "-":
                    name, city = get_valid_name_city(attraction_str)
                    if name not in attractions_seen:
                        attractions_seen[name] = (i, attraction_ix)
                        repeat_flags.append(False)  # First occurrence, not a repeat
                    else:
                        repeat_flags.append(True)  # Repeat
                else:
                    repeat_flags.append(False)  # No attraction, not a repeat
            else:
                repeat_flags.append(False)  # Slot not filled, not a repeat
    
    num_repeats = sum(repeat_flags)
    message = f"Found {num_repeats} repeated attractions out of {len(repeat_flags)} attraction slots"
    return repeat_flags, message


def extract_restaurant_rating_below_minimum_flags(travel_plan, **kwargs):
    """
    Extract a list of booleans indicating whether each meal slot has a rating below the minimum.
    Returns (list, str): (list of booleans, one per meal slot, True if rating < min_rating, False otherwise)
    """
    query_data = kwargs["query_data"]
    preferences = query_data.get("preferences", {})
    min_rating = preferences.get("min_rating_restaurants")
    
    if min_rating is None:
        return [], "No minimum restaurant rating specified"
    
    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()
    
    below_minimum_flags = []
    
    def get_restaurant_rating(meal_field):
        if not meal_field or meal_field == "-":
            return None
        name, city = get_valid_name_city(meal_field)
        res = restaurants.data[
            (restaurants.data["name"].astype(str).str.contains(re.escape(name)))
            & (restaurants.data["city"] == city)
        ]
        if len(res) > 0:
            return res.iloc[0]["rating"]
        return None
    
    for i in range(min(query_data.get("days", 1), len(travel_plan))):
        unit = travel_plan[i]
        for meal in ["breakfast", "lunch", "dinner"]:
            if unit.get(meal) and unit[meal] != "-":
                rating = get_restaurant_rating(unit[meal])
                if rating is not None:
                    below_minimum_flags.append(rating < min_rating)
                else:
                    below_minimum_flags.append(False)  # No rating available, not counted as below minimum
            else:
                below_minimum_flags.append(False)  # No restaurant, not counted as below minimum
    
    num_below = sum(below_minimum_flags)
    message = f"Found {num_below} meals with ratings below {min_rating} out of {len(below_minimum_flags)} meal slots"
    return below_minimum_flags, message


def extract_accommodation_reviews_below_minimum_flags(travel_plan, **kwargs):
    """
    Extract a list of booleans indicating whether each accommodation slot has reviews below the minimum.
    Returns (list, str): (list of booleans, one per accommodation slot, True if num_reviews < min_reviews, False otherwise)
    """
    query_data = kwargs["query_data"]
    preferences = query_data.get("preferences", {})
    min_reviews = preferences.get("min_num_ratings_accommodations")
    
    if min_reviews is None:
        return [], "No minimum accommodation reviews specified"
    
    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()
    
    below_minimum_flags = []
    
    def get_accommodation_reviews(accommodation_field):
        if not accommodation_field or accommodation_field == "-":
            return None
        name, city = get_valid_name_city(accommodation_field)
        res = accommodation.data[
            (accommodation.data["name"].astype(str).str.contains(re.escape(name)))
            & (accommodation.data["city"] == city)
        ]
        if len(res) > 0:
            return res.iloc[0]["num_reviews"]
        return None
    
    # Accommodations are for days 0 to days-2 (last day doesn't need accommodation)
    for i in range(min(query_data.get("days", 1), len(travel_plan)) - 1):
        unit = travel_plan[i]
        if unit.get("accommodation") and unit["accommodation"] != "-":
            num_reviews = get_accommodation_reviews(unit["accommodation"])
            if num_reviews is not None:
                below_minimum_flags.append(num_reviews < min_reviews)
            else:
                below_minimum_flags.append(False)  # No reviews available, not counted as below minimum
        else:
            below_minimum_flags.append(False)  # No accommodation, not counted as below minimum
    
    num_below = sum(below_minimum_flags)
    message = f"Found {num_below} accommodations with reviews below {min_reviews} out of {len(below_minimum_flags)} accommodation slots"
    return below_minimum_flags, message

