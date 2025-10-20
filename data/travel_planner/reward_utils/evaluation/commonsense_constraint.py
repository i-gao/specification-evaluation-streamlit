import os
import sys
import re
import numpy as np
import string
from datetime import datetime

from data.travel_planner.reward_utils.tp_utils.func import (
    get_valid_name_city,
    extract_from_to,
    extract_before_parenthesis,
    get_tools,
    count_consecutive_values,
)


def transportation_match(text: str):
    """
    Determines the type of transportation from a text string.
    Returns 'Taxi', 'Self-driving', or 'Flight' based on keywords in the text.
    """
    if "taxi" in text.lower():
        return "Taxi"
    elif "self-driving" in text.lower():
        return "Self-driving"
    elif "flight" in text.lower():
        return "Flight"


def is_valid_city_sequence(city_list):
    """
    Validates that a sequence of cities follows reasonable travel patterns:
    - Must have at least 3 cities (at minimum: origin, destination, origin)
    - Cities cannot be repeated in the middle of the sequence
    - Single-day visits are not allowed in the middle of the sequence
    - First and last cities can be repeated (for round trips)
    """
    if len(city_list) < 3:
        return False

    visited_cities = set()

    i = 0
    while i < len(city_list):
        city = city_list[i]

        if city in visited_cities and (i != 0 and i != len(city_list) - 1):
            return False

        count = 0
        while i < len(city_list) and city_list[i] == city:
            count += 1
            i += 1

        if count == 1 and 0 < i - 1 < len(city_list) - 1:
            return False

        visited_cities.add(city)

    return True


def is_reasonable_visiting_city(question, tested_data):
    """
    Validates that the city sequence follows basic travel constraints:
    - First city must match the origin city
    - Trip must form a closed circle (start and end at same city)
    # - City sequence must be valid according to is_valid_city_sequence
    - All cities must exist in the city-state map
    - For trips > 3 days, intermediate cities must be in the destination state
    """
    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()
    city_list = []

    for i in range(min(question["days"], len(tested_data))):
        city_value = tested_data[i]["current_city"]

        if "from" in city_value:
            city1, city2 = extract_from_to(city_value)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            if i == 0 and city1 != question["org"]:
                return False, f"The first day's city should be {question['org']}."

            city_list += [city1, city2]
        else:
            city_list.append(extract_before_parenthesis(city_value))

    if len(city_list) == 0:
        return False, "The trip should have at least one city."

    if city_list[0] != city_list[-1]:
        return (
            False,
            "The trip should be a closed circle: the origin of the first day and the destination of the last day should be the same.",
        )

    for idx, city in enumerate(city_list):
        if city not in city_state_map:
            return False, f"{city} is not a valid city."
        if (
            idx not in [0, len(city_list) - 1]
            and question["days"] > 3
            and city_state_map[city] != question["dest"]
        ):
            return False, f"{city} is not in {question['dest']}."

    return True, None


# def is_valid_restaurants(question, tested_data):
#     """
#     Validates that restaurant choices follow basic constraints:
#     - No restaurant can be repeated across different days
#     - Validates breakfast, lunch, and dinner choices
#     """
#     restaurants_list = []

#     for i in range(min(question["days"], len(tested_data))):
#         unit = tested_data[i]

#         if "breakfast" in unit and unit["breakfast"] and unit["breakfast"] != "-":
#             if unit["breakfast"] not in restaurants_list:
#                 restaurants_list.append(unit["breakfast"])
#             else:
#                 return False, f"The restaurant in day {i+1} breakfast is repeated."

#         if "lunch" in unit and unit["lunch"] and unit["lunch"] != "-":
#             if unit["lunch"] not in restaurants_list:
#                 restaurants_list.append(unit["lunch"])
#             else:
#                 return (
#                     False,
#                     f"The restaurant in day {i+1} lunch {unit['lunch']} is repeated.",
#                 )

#         if "dinner" in unit and unit["dinner"] and unit["dinner"] != "-":
#             if unit["dinner"] not in restaurants_list:
#                 restaurants_list.append(unit["dinner"])
#             else:
#                 return False, f"The restaurant in day {i+1} dinner is repeated."

#     return True, None


# def is_valid_attractions(question, tested_data):
#     """
#     Validates that attraction choices follow basic constraints:
#     - No attraction can be repeated across different days
#     - Handles multiple attractions per day (separated by semicolons)
#     """
#     attractions_list = []

#     for i in range(min(question["days"], len(tested_data))):
#         unit = tested_data[i]

#         if "attraction" in unit and unit["attraction"] and unit["attraction"] != "-":
#             for attraction in unit["attraction"].split(";")[:-1]:
#                 if attraction not in attractions_list:
#                     attractions_list.append(attraction)
#                 else:
#                     return (
#                         False,
#                         f"The attraction '{attraction}' in day {i+1} is repeated.",
#                     )

#     return True, None


def is_valid_transportation(question, tested_data):
    """
    Validates that transportation choices follow basic constraints:
    - First day must have transportation specified
    # - Cannot mix conflicting transportation types:
    #   - Cannot use both self-driving and flights
    #   - Cannot use both taxi and self-driving
    """
    if len(tested_data) == 0:
        return False, "The trip should have at least one day."

    if tested_data[0]["transportation"] and tested_data[0]["transportation"] != "-":
        transportation_list = [transportation_match(tested_data[0]["transportation"])]
    else:
        return False, "The transportation in day 1 should not be empty."

    # for i in range(min(question["days"], len(tested_data))):
    #     unit = tested_data[i]
    #     if (
    #         "transportation" in unit
    #         and unit["transportation"]
    #         and unit["transportation"] != "-"
    #     ):
    #         transportation_list.append(transportation_match(unit["transportation"]))

    # if (
    #     ("Self-driving" in transportation_list) and ("Flight" in transportation_list)
    # ) or (("Taxi" in transportation_list) and ("Self-driving" in transportation_list)):
    #     return False, "The transportation is conflicting."

    return True, None


def normalize_string(s):
    """
    Normalizes a string for comparison by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Normalizing whitespace
    """
    if not isinstance(s, str):
        return ""
    # Convert to lowercase
    s = s.lower()
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Normalize whitespace
    s = " ".join(s.split())
    return s


def is_valid_information_in_current_city(question, tested_data):
    """
    Validates that all activities (transportation, meals, attractions, accommodation)
    are in the correct city for each day:
    - Transportation must mention the cities being traveled between
    - Meals (breakfast, lunch, dinner) must be in the current city
    - Attractions must be in the current city
    - Accommodation must be in the current city
    """
    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]
        current_city = unit["current_city"]
        final_city_list = []

        if "from" in current_city:
            city1, city2 = extract_from_to(current_city)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            final_city_list = [city1, city2]
        else:
            final_city_list = extract_before_parenthesis(current_city)

        if (
            "transportation" in unit
            and unit["transportation"]
            and unit["transportation"] != "-"
        ):
            for city in final_city_list:
                if city not in unit["transportation"]:
                    return (
                        False,
                        f"The transportation in day {i+1} is not in a valid city from the `current_city` field.",
                    )

        if "breakfast" in unit and unit["breakfast"] and unit["breakfast"] != "-":
            flag = False
            for city in final_city_list:
                if city in unit["breakfast"]:
                    flag = True
            if not flag:
                return (
                    False,
                    f"The breakfast in day {i+1} is not in a valid city from the `current_city` field.",
                )

        if "lunch" in unit and unit["lunch"] and unit["lunch"] != "-":
            flag = False
            for city in final_city_list:
                if city in unit["lunch"]:
                    flag = True

            if not flag:
                return (
                    False,
                    f"The lunch in day {i+1} is not in a valid city from the `current_city` field.",
                )

        if "dinner" in unit and unit["dinner"] and unit["dinner"] != "-":
            flag = False
            for city in final_city_list:
                if city in unit["dinner"]:
                    flag = True

            if not flag:
                return (
                    False,
                    f"The dinner in day {i+1} is not in a valid city from the `current_city` field.",
                )

        if "attraction" in unit and unit["attraction"] and unit["attraction"] != "-":
            flag = False
            for city in final_city_list:
                if city in unit["attraction"]:
                    flag = True

            if not flag:
                return (
                    False,
                    f"The attraction in day {i+1} is not in a valid city from the `current_city` field.",
                )

        if (
            "accommodation" in unit
            and unit["accommodation"]
            and unit["accommodation"] != "-"
        ):
            flag = False
            for city in final_city_list:
                if city in unit["accommodation"]:
                    flag = True

            if not flag:
                return (
                    False,
                    f"The accommodation in day {i+1} is not in a valid city from the `current_city` field.",
                )

    return True, None


def is_valid_information_in_sandbox(question, tested_data):
    """
    Validates that all activities exist in the sandbox data:
    - Flights must have valid flight numbers and city pairs and exist on the right date
    - Self-driving and taxi routes must have valid distances
    - Restaurants must exist in the specified cities
    - Attractions must exist in the specified cities
    - Accommodations must exist in the specified cities
    """
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
        if isinstance(question["date"], str):
            unit_date = eval(question["date"])[i]
        else:
            unit_date = question["date"][i]

        if unit["transportation"] and unit["transportation"] != "-":
            value = unit["transportation"]
            org_city, dest_city = extract_from_to(value)
            if org_city == None or dest_city == None:
                org_city, dest_city = extract_from_to(unit["current_city"])
            if "flight number" in value.lower():
                try:
                    org_city = extract_before_parenthesis(org_city)
                    dest_city = extract_before_parenthesis(dest_city)
                except TypeError:
                    raise ValueError(
                        "The transportation {} in day {} can not be parsed. Make sure the flight is formatted as 'Flight Number: [number], from [city] to [city], Departure Time: [time], Arrival Time: [time]'.".format(
                            value, i + 1
                        )
                    )
                if (
                    len(
                        flight.data[
                            (
                                flight.data["flight_number"]
                                == value.split("Flight Number: ")[1].split(",")[0]
                            )
                            & (flight.data["departure_city"] == org_city)
                            & (flight.data["arrival_city"] == dest_city)
                            & (flight.data["flight_date"] == unit_date)
                        ]
                    )
                    < 1
                ):
                    return (
                        False,
                        f"The flight in day {i+1} is not a valid flight from the database. Did you correctly filter the flight database to reflect the desired date and cities?",
                    )

            elif "self-driving" in value.lower() or "taxi" in value.lower():
                try:
                    org_city = extract_before_parenthesis(org_city)
                    dest_city = extract_before_parenthesis(dest_city)
                except TypeError:
                    org_city = "-"
                    dest_city = "-"
                    print(
                        "The transportation {} in day {} can not be parsed and '-' will be used instead.".format(
                            value, i + 1
                        )
                    )

                if "self-driving" in value.lower():
                    if (
                        googleDistanceMatrix.run_for_evaluation(
                            org_city, dest_city, mode="self-driving"
                        )["cost"]
                        == None
                    ):
                        return (
                            False,
                            f"The self-driving in day {i+1} is not a valid route in the driving options. Did you correctly use the get_driving_options tool?",
                        )
                else:
                    if (
                        googleDistanceMatrix.run_for_evaluation(
                            org_city, dest_city, mode="taxi"
                        )["cost"]
                        == None
                    ):
                        return (
                            False,
                            f"The taxi in day {i+1} is not a valid route in the driving options. Did you correctly use the get_driving_options tool?",
                        )

        if "breakfast" in unit and unit["breakfast"] and unit["breakfast"] != "-":
            name, city = get_valid_name_city(unit["breakfast"])
            if (
                len(
                    restaurants.data[
                        (
                            restaurants.data["name"]
                            .astype(str)
                            .apply(normalize_string)
                            .str.contains(normalize_string(name), regex=False)
                        )
                        & (restaurants.data["city"] == city)
                    ]
                )
                < 1
            ):
                return (
                    False,
                    f"The breakfast in day {i+1} is not a valid restaurant in that city in the database. Make sure to format the restaurant as 'Restaurant Name, City'.",
                )

        if "lunch" in unit and unit["lunch"] and unit["lunch"] != "-":
            name, city = get_valid_name_city(unit["lunch"])
            if (
                len(
                    restaurants.data[
                        (
                            restaurants.data["name"]
                            .astype(str)
                            .apply(normalize_string)
                            .str.contains(normalize_string(name), regex=False)
                        )
                        & (restaurants.data["city"] == city)
                    ]
                )
                < 1
            ):
                return (
                    False,
                    f"The lunch in day {i+1} is not a valid restaurant in that city in the database. Make sure to format the restaurant as 'Restaurant Name, City'.",
                )

        if "dinner" in unit and unit["dinner"] and unit["dinner"] != "-":
            name, city = get_valid_name_city(unit["dinner"])
            if (
                len(
                    restaurants.data[
                        (
                            restaurants.data["name"]
                            .astype(str)
                            .apply(normalize_string)
                            .str.contains(normalize_string(name), regex=False)
                        )
                        & (restaurants.data["city"] == city)
                    ]
                )
                < 1
            ):
                return (
                    False,
                    f"The dinner in day {i+1} is not a valid restaurant in that city in the database. Make sure to format the restaurant as 'Restaurant Name, City'.",
                )

        if "attraction" in unit and unit["attraction"] and unit["attraction"] != "-":
            attractions_list = unit["attraction"].split(";")
            attractions_list = [
                attraction for attraction in attractions_list if attraction != ""
            ]
            for attraction in attractions_list:
                name, city = get_valid_name_city(attraction)
                if (
                    len(
                        attractions.data[
                            (
                                attractions.data["name"]
                                .astype(str)
                                .apply(normalize_string)
                                .str.contains(normalize_string(name), regex=False)
                            )
                            & (attractions.data["city"] == city)
                        ]
                    )
                    < 1
                ):
                    return (
                        False,
                        f"The attraction {attraction} in day {i+1} is not a valid attraction in that city in the database. Make sure to format the attraction as 'Name, City' and separate multiple attractions with semicolons.",
                    )

        if (
            "accommodation" in unit
            and unit["accommodation"]
            and unit["accommodation"] != "-"
        ):
            name, city = get_valid_name_city(unit["accommodation"])
            if (
                len(
                    accommodation.data[
                        (
                            accommodation.data["name"]
                            .astype(str)
                            .apply(normalize_string)
                            .str.contains(normalize_string(name), regex=False)
                        )
                        & (accommodation.data["city"] == city)
                    ]
                )
                < 1
            ):
                return (
                    False,
                    f"The accommodation in day {i+1} is not a valid accommodation in that city in the database. Make sure to format the accommodation as 'Hotel/lodging name, City'.",
                )

    return True, None


def is_valid_accommodaton(question, tested_data):
    """
    Validates accommodation choices follow basic constraints:
    - Must have accommodation information for each day
    - Must stay at each accommodation for the minimum required nights
    - Accommodations must exist in the sandbox data
    """
    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()
    data = []
    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]

        if "accommodation" not in unit:
            return False, f"No Accommodation Info."

        data.append(unit["accommodation"])

    consectutive_accommodation = count_consecutive_values(data)
    for unit in consectutive_accommodation:
        if unit and unit[0] not in ["-", ""]:
            name, city = get_valid_name_city(unit[0])
            if (
                len(
                    accommodation.data[
                        (
                            accommodation.data["name"]
                            .astype(str)
                            .str.contains(re.escape(name))
                        )
                        & (accommodation.data["city"] == city)
                    ]
                )
                == 1
            ):
                min_nights = accommodation.data[
                    (
                        accommodation.data["name"]
                        .astype(str)
                        .str.contains(re.escape(name))
                    )
                    & (accommodation.data["city"] == city)
                ].iloc[0]["minimum_nights"]
                if unit[1] < min_nights:
                    return (
                        False,
                        f"The accommodation {unit[0]} does not obey the minumum nights rule: plan stays for {unit[1]} nights, but the minimum nights is {min_nights} nights.",
                    )

    return True, None


def is_valid_visiting_city_number(question, tested_data):
    """
    Validates that the number of unique cities visited matches the required number:
    - Counts unique cities excluding the origin city
    - Must match the specified visiting_city_number
    """
    city_set = set()

    for i in range(min(question["days"], len(tested_data))):
        city_value = tested_data[i]["current_city"]

        if "from" in city_value:
            city1, city2 = extract_from_to(city_value)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            if i == 0 and city1 != question["org"]:
                return False, f"The first day's city should be {question['org']}."

            city_set.add(city1)
            city_set.add(city2)
        else:
            city_set.add(extract_before_parenthesis(city_value))

    city_set.discard(question["org"])

    if len(city_set) != question["visiting_city_number"]:
        return (
            False,
            f"The number of visiting cities should be {question['visiting_city_number']}.",
        )

    return True, None


def is_valid_days(question, tested_data):
    """
    Validates that the number of days matches the required number:
    - Counts only days with valid city information
    - Must match the specified number of days
    """
    lens = 0
    for i in range(min(question["days"], len(tested_data))):
        if (
            tested_data[i] != {}
            and tested_data[i]["current_city"]
            != "You don't need to fill in the information for this or later days."
        ):
            lens += 1

    if lens != question["days"]:
        return False, f"The number of days should be {question['days']}."
    else:
        return True, None


def is_not_absent(question, tested_data):
    """
    Validates that required information is present for each day:
    - Must have transportation, meals, attractions, and accommodation info
    - Transportation required when traveling between cities
    - Attractions required for single-city days
    - Accommodation required for all days except the last
    - Meals required for single-city days
    - At least 50% of required information must be present
    """
    needed_info = 6 * question["days"]
    total_valid_info = 0

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]

        if "transportation" not in unit:
            return False, f"No transportation info for day {i+1}."

        if "breakfast" not in unit:
            return False, f"No breakfast info for day {i+1}."

        if "lunch" not in unit:
            return False, f"No lunch info for day {i+1}."

        if "dinner" not in unit:
            return False, f"No dinner info for day {i+1}."

        if "attraction" not in unit:
            return False, f"No attraction info for day {i+1}."

        if "accommodation" not in unit:
            return False, f"No accommodation info for day {i+1}."

        if ("from " in unit["current_city"] or "to " in unit["current_city"]) and unit[
            "transportation"
        ] in ["", "-"]:
            return False, f"No transportation in day {i+1} is not allowed."

        if (
            "from " not in unit["current_city"] and " to " not in unit["current_city"]
        ) and unit["attraction"] in ["", "-"]:
            return False, f"No attaction in day {i+1} is not allowed."

        if i != question["days"] - 1 and unit["accommodation"] in ["", "-"]:
            return False, f"No accommodation in day {i+1} is not allowed."

        if (
            unit["breakfast"] in ["", "-"]
            or unit["lunch"] in ["", "-"]
            or unit["dinner"] in ["", "-"]
        ) and "from " not in unit["current_city"]:
            return False, f"No meal in day {i+1} is not allowed."

        for key in unit:
            if unit[key] and unit[key] != "-":
                total_valid_info += 1

    # if total_valid_info * 1.0 / needed_info < 0.5:
    #     return False, f"More than 50% of the fields are empty."

    return True, None


def meals_match_travel_time(question, tested_data):
    """
    Validates that the meals of the first / last day reflects the travel time.
    For example, if the flight on day 1 to the location arrives at 5p, breakfast and
    lunch should be empty in the plan.
    """
    if len(tested_data) == 0:
        return False, "The trip should have at least one day."

    # Check first day
    unit = tested_data[0]
    city1, city2 = extract_from_to(unit["current_city"])
    city1 = extract_before_parenthesis(city1)
    city2 = extract_before_parenthesis(city2)
    if (
        unit["transportation"]
        and unit["transportation"] != "-"
        and "Arrival Time" in unit["transportation"]
    ):
        arrival_time = unit["transportation"].split("Arrival Time: ")[1]
        arrival_time = arrival_time.replace(" ", "")
        arrival_time = datetime.strptime(arrival_time, "%H:%M")
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
    unit = tested_data[-1]
    city1, city2 = extract_from_to(unit["current_city"])
    city1 = extract_before_parenthesis(city1)
    city2 = extract_before_parenthesis(city2)
    if (
        unit["transportation"]
        and unit["transportation"] != "-"
        and "Departure Time" in unit["transportation"]
    ):

        departure_time = (
            unit["transportation"].split("Departure Time: ")[1].split(",")[0]
        )
        departure_time = departure_time.replace(" ", "")
        departure_time = datetime.strptime(departure_time, "%H:%M")
        # If we depart before 2p, dinner should be empty
        if departure_time.hour < 14:
            if (
                unit["dinner"]
                and unit["dinner"] != "-"
                and get_valid_name_city(unit["dinner"])[1] == city1
            ):
                return (
                    False,
                    f"We depart from {city1} before 12p, so we should not have dinner there.",
                )

        # If we depart before 9a, lunch and dinner should be empty
        if departure_time.hour < 9:
            if (
                unit["lunch"]
                and unit["lunch"] != "-"
                and get_valid_name_city(unit["lunch"])[1] == city1
            ):
                return (
                    False,
                    f"We depart from {city1} before 9a, so we should not have lunch or dinner there.",
                )
            if (
                unit["dinner"]
                and unit["dinner"] != "-"
                and get_valid_name_city(unit["dinner"])[1] == city1
            ):
                return (
                    False,
                    f"We depart from {city1} before 9a, so we should not have lunch or dinner there.",
                )
            
    if (
        unit["transportation"]
        and unit["transportation"] != "-"
        and "duration" in unit["transportation"]
    ):
        duration = unit["transportation"].split("duration: ")[1].split(",")[0]
        duration_hours = int(duration.split("hour")[0])
        # a reasonable arriving time is like midnight
        # if we depart before 2p, dinner should be empty
        # this means duration should be < 10 hours
        if duration_hours >= 10:
            if (
                unit["dinner"]
                and unit["dinner"] != "-"
                and get_valid_name_city(unit["dinner"])[1] == city1
            ):
                return (
                    False,
                    f"We depart from {city1} before 2p, so we should not have dinner there.",
                )
        # if we depart before 9a, lunch and dinner should be empty
        # this means duration should be < 15 hours
        if duration_hours >= 15:
            if (
                unit["lunch"]
                and unit["lunch"] != "-"
                and get_valid_name_city(unit["lunch"])[1] == city1
            ):
                return (
                    False,
                    f"We depart from {city1} before 9a, so we should not have lunch or dinner there.",
                )
            if (
                unit["dinner"]
                and unit["dinner"] != "-"
                and get_valid_name_city(unit["dinner"])[1] == city1
            ):
                return (
                    False,
                    f"We depart from {city1} before 9a, so we should not have lunch or dinner there.",
                )

    return True, None


def evaluation(query_data, tested_data):
    """
    Main evaluation function that runs all constraint checks:
    - Validates city sequence and visiting patterns
    - Validates restaurant and attraction choices
    - Validates accommodation rules
    - Validates transportation choices
    - Validates city-specific information
    - Validates sandbox data existence
    - Validates information completeness
    Returns a dictionary with results of all constraint checks
    """
    return_info = {}
    return_info["is_reasonable_visiting_city"] = is_reasonable_visiting_city(
        query_data, tested_data
    )
    return_info["is_valid_accommodation"] = is_valid_accommodaton(
        query_data, tested_data
    )
    return_info["is_valid_transportation"] = is_valid_transportation(
        query_data, tested_data
    )
    return_info["is_valid_information_in_current_city"] = (
        is_valid_information_in_current_city(query_data, tested_data)
    )
    return_info["is_valid_information_in_sandbox"] = is_valid_information_in_sandbox(
        query_data, tested_data
    )
    return_info["is_not_absent"] = is_not_absent(query_data, tested_data)
    return_info["is_valid_days"] = is_valid_days(query_data, tested_data)
    return_info["is_valid_visiting_city_number"] = is_valid_visiting_city_number(
        query_data, tested_data
    )
    return_info["meals_match_travel_time"] = meals_match_travel_time(
        query_data, tested_data
    )
    return return_info
