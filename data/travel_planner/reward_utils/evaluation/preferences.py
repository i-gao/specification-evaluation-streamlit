import os
import sys
import re
import numpy as np
from typing import Dict, List, Any, Tuple

from data.travel_planner.reward_utils.tp_utils.func import (
    get_valid_name_city,
    get_tools,
    get_attractions,
)


def hinge_reward(
    value,
    *,  # keyword-only arguments
    lower,
    upper,
    lower_val=-1.0,
    upper_val=1.0,
    min_val=None,
    max_val=None,
    none_val=0.0,
):
    """
    Hinge reward function that gives lower_val below lower, upper_val above upper,
    and linear interpolation in between.
    The line is bounded by min_val and max_val.
    """
    if value is None:
        return none_val
    if value <= lower:
        return lower_val
    elif value >= upper:
        return upper_val
    else:
        if min_val is None:
            min_val = min(lower_val, upper_val)
        if max_val is None:
            max_val = max(lower_val, upper_val)
        # Linear interpolation
        return min(
            max_val,
            max(
                min_val,
                lower_val + (upper_val - lower_val) * (value - lower) / (upper - lower),
            ),
        )


def evaluate_tag_preferences(query_data, tested_data):
    """
    Evaluates tag preferences for meals.
    For every meal, we earn:
    - Positive reward (+1) if the restaurant has any liked tags
    - Negative reward (-1) if the restaurant has any disliked tags
    - 0 if the meal is unscheduled or the restaurant has no tags
    - 0 if the restaurant has both a liked tag and a disliked tag

    Returns:
        List of tuples: ((day, meal), reward, actual_restaurant_tags)
            Empty list [] if there are no tag preferences
        Summary string: f"Tag preferences evaluated: liked={liked_tags}, disliked={disliked_tags}"
    """
    preferences = query_data.get("preferences", {})
    liked_tags = preferences.get("liked_tags")
    disliked_tags = preferences.get("disliked_tags")

    if not liked_tags and not disliked_tags:
        return None, "No tag preferences specified"

    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()

    results = []

    def get_tags(meal_field):
        name, city = get_valid_name_city(meal_field)
        res = restaurants.data[
            (restaurants.data["name"].astype(str).str.contains(re.escape(name)))
            & (restaurants.data["city"] == city)
        ]
        if len(res) > 0:
            return res.iloc[0]["tags"]
        return []

    for i in range(min(query_data["days"], len(tested_data))):
        unit = tested_data[i]

        for meal in ["breakfast", "lunch", "dinner"]:
            if unit.get(meal) and unit[meal] != "-":
                restaurant_tags = get_tags(unit[meal])
                reward = 0
                # (both cases can happen if a restaurant has multiple tags)
                if liked_tags and any(tag in restaurant_tags for tag in liked_tags):
                    reward += 1
                if disliked_tags and any(
                    tag in restaurant_tags for tag in disliked_tags
                ):
                    reward -= 1
                results.append(((i, meal), reward, restaurant_tags))
            else:
                results.append(((i, meal), 0, None))

    return (
        results,
        f"Tag preferences evaluated: liked={liked_tags}, disliked={disliked_tags}",
    )


def evaluate_specific_liked_restaurants(query_data, tested_data):
    """
    Evaluates specific liked restaurants.
    For every meal, we earn:
    - (+1) if the restaurant is in the specific liked restaurants list
    - 0 if the meal is unscheduled or not in the specific liked restaurants list

    Returns:
        List of tuples: ((day, meal), reward, actual_restaurant_name)
            Empty list [] if there are no specific liked restaurants
        Summary string: f"Specific liked restaurants evaluated: specific_liked={specific_liked_restaurants}"
    """
    preferences = query_data.get("preferences", {})
    specific_liked_restaurants = preferences.get("specific_liked_restaurants")
    if not specific_liked_restaurants:
        return None, "No specific liked restaurants specified"

    results = []

    def get_name(meal_field):
        name, city = get_valid_name_city(meal_field)
        return name

    for i in range(min(query_data["days"], len(tested_data))):
        unit = tested_data[i]
        for meal in ["breakfast", "lunch", "dinner"]:
            if unit.get(meal) and unit[meal] != "-":
                restaurant_name = get_name(unit[meal])
                if restaurant_name in specific_liked_restaurants:
                    results.append(((i, meal), 1, restaurant_name))
                else:
                    results.append(((i, meal), 0, restaurant_name))
            else:
                results.append(((i, meal), 0, None))
    return (
        results,
        f"Specific liked restaurants evaluated: specific_liked={specific_liked_restaurants}",
    )


def evaluate_specific_disliked_restaurants(query_data, tested_data):
    """
    Evaluates specific disliked restaurants.
    For every meal, we earn:
    - (-1) if the restaurant is in the specific disliked restaurants list
    - 0 if the meal is unscheduled or not in the specific disliked restaurants list

    Returns:
        List of tuples: ((day, meal), reward, actual_restaurant_name)
            Empty list [] if there are no specific disliked restaurants
        Summary string: f"Specific disliked restaurants evaluated: specific_disliked={specific_disliked_restaurants}"
    """
    preferences = query_data.get("preferences", {})
    specific_disliked_restaurants = preferences.get("specific_disliked_restaurants")
    if not specific_disliked_restaurants:
        return None, "No specific disliked restaurants specified"

    results = []

    def get_name(meal_field):
        name, city = get_valid_name_city(meal_field)
        return name

    for i in range(min(query_data["days"], len(tested_data))):
        unit = tested_data[i]
        for meal in ["breakfast", "lunch", "dinner"]:
            if unit.get(meal) and unit[meal] != "-":
                restaurant_name = get_name(unit[meal])
                if restaurant_name in specific_disliked_restaurants:
                    results.append(((i, meal), -1, restaurant_name))
                else:
                    results.append(((i, meal), 0, restaurant_name))
            else:
                results.append(((i, meal), 0, None))
    return (
        results,
        f"Specific disliked restaurants evaluated: specific_disliked={specific_disliked_restaurants}",
    )


def evaluate_restaurant_ratings(query_data, tested_data):
    """
    Evaluates restaurant ratings.
    For every meal, we earn:
    - a linear reward between 0 and 1 if the restaurant rating is >= min_rating_restaurants (max 1)
    - a linear reward between 0 and -1 if the restaurant rating is < min_rating_restaurants (min -1)
    - 0 if the meal is unscheduled or the restaurant has no rating

    Returns:
        List of tuples: ((day, meal), reward, rating)
        Summary string: f"Restaurant ratings evaluated: min_rating_restaurants={min_rating_restaurants}"
    """
    preferences = query_data.get("preferences", {})
    min_rating_restaurants = preferences.get("min_rating_restaurants")

    if not min_rating_restaurants:
        return None, "No restaurant rating threshold specified"

    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()

    results = []

    def get_restaurant_rating(meal_field):
        name, city = get_valid_name_city(meal_field)
        res = restaurants.data[
            (restaurants.data["name"].astype(str).str.contains(re.escape(name)))
            & (restaurants.data["city"] == city)
        ]
        if len(res) > 0:
            return res.iloc[0]["rating"]
        return None

    # Check restaurant ratings with hinge reward
    for i in range(min(query_data["days"], len(tested_data))):
        unit = tested_data[i]

        for meal in ["breakfast", "lunch", "dinner"]:
            if unit.get(meal) and unit[meal] != "-":
                rating = get_restaurant_rating(unit[meal])
                if rating is not None:
                    reward = hinge_reward(
                        rating,
                        lower=min_rating_restaurants,
                        upper=5,
                        lower_val=0,
                        upper_val=1,
                        min_val=-1,
                        max_val=1,
                    )
                    results.append(((i, meal), reward, rating))
                else:
                    results.append(((i, meal), 0, None))

    return (
        results,
        f"Restaurant ratings evaluated: min_rating_restaurants={min_rating_restaurants}",
    )


def evaluate_restaurant_repeats(question, tested_data):
    """
    Evaluates restaurant repeats.
    For every meal, we earn:
    - (-1) if the restaurant is a repeat from a previous meal
    - 0 if the restaurant is not a repeat from a previous meal or the meal is unscheduled

    Returns:
        List of tuples: ((day, meal), reward, first_occurrence)
            Empty list [] if there are no restaurant repeats
        Summary string: f"Evaluated for restaurant repeats"
    """
    restaurants_list = {}  # {restaurant_name: first_occurrence}
    results = []

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]

        if "breakfast" in unit and unit["breakfast"] and unit["breakfast"] != "-":
            if unit["breakfast"] not in restaurants_list:
                restaurants_list[unit["breakfast"]] = (i, "breakfast")
                results.append(((i, "breakfast"), 0, None))
            else:
                results.append(
                    ((i, "breakfast"), -1, restaurants_list[unit["breakfast"]])
                )

        if "lunch" in unit and unit["lunch"] and unit["lunch"] != "-":
            if unit["lunch"] not in restaurants_list:
                restaurants_list[unit["lunch"]] = (i, "lunch")
                results.append(((i, "lunch"), 0, None))
            else:
                results.append(((i, "lunch"), -1, restaurants_list[unit["lunch"]]))

        if "dinner" in unit and unit["dinner"] and unit["dinner"] != "-":
            if unit["dinner"] not in restaurants_list:
                restaurants_list[unit["dinner"]] = (i, "dinner")
                results.append(((i, "dinner"), 0, None))
            else:
                results.append(((i, "dinner"), -1, restaurants_list[unit["dinner"]]))

    return results, "Evaluated for restaurant repeats"


def evaluate_restaurant_attributes(query_data, tested_data):
    """
    Evaluates the restaurant attributes preference.
    For every meal, we earn:
    - 1 if the restaurant has all the attributes
    - -1 if the restaurant has none of the attributes
    - a linear function in between -1 and 1 if the restaurant has some of the attributes
    - 0 if the meal is unscheduled

    Returns:
        List of tuples: ((day, meal), reward, (pct_fulfilled, actual_restaurant_attributes))
        Summary string: f"Restaurant attributes evaluated: restaurant_attributes={restaurant_attributes}"
    """
    preferences = query_data.get("preferences", {})
    restaurant_attributes = preferences.get("restaurant_attributes")
    if not restaurant_attributes or len(restaurant_attributes) == 0:
        return None, "No restaurant attributes specified"
    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()

    def get_pct_of_attributes(meal_field):
        name, city = get_valid_name_city(meal_field)
        res = restaurants.data[
            (restaurants.data["name"].astype(str).str.contains(re.escape(name)))
            & (restaurants.data["city"] == city)
        ]
        if len(res) == 0:
            # cannot find the restaurant
            return 0, None

        row = res.iloc[0]
        actual_attributes = [row.get(att) for att in restaurant_attributes]
        pct_of_attributes = sum(
            1
            for i, att in enumerate(restaurant_attributes)
            if actual_attributes[i] == restaurant_attributes[att]
        ) / len(restaurant_attributes)
        return pct_of_attributes, actual_attributes

    results = []
    for i in range(min(query_data["days"], len(tested_data))):
        unit = tested_data[i]
        if "breakfast" in unit and unit["breakfast"] and unit["breakfast"] != "-":
            pct_fulfilled, actual_attributes = get_pct_of_attributes(unit["breakfast"])
            reward = hinge_reward(
                pct_fulfilled,
                lower=0,
                upper=1,
                lower_val=-1,
                upper_val=1,
                none_val=0,
            )
            results.append(
                ((i, "breakfast"), reward, (pct_fulfilled, actual_attributes))
            )
        if "lunch" in unit and unit["lunch"] and unit["lunch"] != "-":
            pct_fulfilled, actual_attributes = get_pct_of_attributes(unit["lunch"])
            reward = hinge_reward(
                pct_fulfilled,
                lower=0,
                upper=1,
                lower_val=-1,
                upper_val=1,
                none_val=0,
            )
            results.append(((i, "lunch"), reward, (pct_fulfilled, actual_attributes)))
        if "dinner" in unit and unit["dinner"] and unit["dinner"] != "-":
            pct_fulfilled, actual_attributes = get_pct_of_attributes(unit["dinner"])
            reward = hinge_reward(
                pct_fulfilled,
                lower=0,
                upper=1,
                lower_val=-1,
                upper_val=1,
                none_val=0,
            )
            results.append(((i, "dinner"), reward, (pct_fulfilled, actual_attributes)))
    return (
        results,
        f"Restaurant attributes evaluated: restaurant_attributes={restaurant_attributes}",
    )


def evaluate_accommodation_reviews(query_data, tested_data):
    """
    Evaluates accommodation review counts.
    For every day except the last day, we earn:
    - -1 if the accommodation review count is < min_num_ratings_accommodations
    - 1 if the accommodation review count is >= min_num_ratings_accommodations
    - 0 if the day's accommodation is unscheduled or the accommodation has no review count

    Returns:
        List of tuples: ((day, accommodation), reward, num_reviews)
        Summary string: f"Accommodation reviews evaluated: min_num_ratings_accommodations={min_num_ratings_accommodations}"
    """
    preferences = query_data.get("preferences", {})
    min_num_ratings_accommodations = preferences.get("min_num_ratings_accommodations")

    if not min_num_ratings_accommodations:
        return None, "No accommodation review threshold specified"

    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()

    results = []

    def get_accommodation_reviews(accommodation_field):
        name, city = get_valid_name_city(accommodation_field)
        res = accommodation.data[
            (accommodation.data["name"].astype(str).str.contains(re.escape(name)))
            & (accommodation.data["city"] == city)
        ]
        if len(res) > 0:
            return res.iloc[0]["num_reviews"]
        return None

    # Check accommodation review counts with hinge reward
    for i in range(min(query_data["days"], len(tested_data)) - 1):
        unit = tested_data[i]

        if unit.get("accommodation") and unit["accommodation"] != "-":
            num_reviews = get_accommodation_reviews(unit["accommodation"])
            if num_reviews is not None:
                reward = 1 if num_reviews >= min_num_ratings_accommodations else -1
                results.append(((i, "accommodation"), reward, num_reviews))
            else:
                results.append(((i, "accommodation"), 0, None))

    return (
        results,
        f"Accommodation reviews evaluated: min_num_ratings_accommodations={min_num_ratings_accommodations}",
    )


def evaluate_room_type_preferences(query_data, tested_data):
    """
    Evaluates room type preferences for accommodations.
    For every day except the last day, we earn:
    - Positive reward (+1) if the day's accommodation fits any liked room types
    - Negative reward (-1) if the day's accommodation fits any disliked room types
    - 0 if the day's accommodation is unscheduled

    Returns:
        List of tuples: (day, reward, actual_room_type)
            Empty list [] if there are no room type preferences
        Summary string: f"Room type preferences evaluated: liked={liked_room_types}, disliked={disliked_room_types}"
    """
    preferences = query_data.get("preferences", {})
    liked_room_types = preferences.get("liked_room_types")
    disliked_room_types = preferences.get("disliked_room_types")

    if not liked_room_types and not disliked_room_types:
        return None, "No room type preferences specified"

    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()

    results = []

    def get_room_type(accommodation_field):
        name, city = get_valid_name_city(accommodation_field)
        res = accommodation.data[
            (accommodation.data["name"].astype(str).str.contains(re.escape(name)))
            & (accommodation.data["city"] == city)
        ]
        if len(res) > 0:
            return res.iloc[0]["room_type"]
        return None

    for i in range(min(query_data["days"], len(tested_data)) - 1):
        unit = tested_data[i]

        if unit.get("accommodation") and unit["accommodation"] != "-":
            room_type = get_room_type(unit["accommodation"])
            if liked_room_types and room_type in liked_room_types:
                reward = 1
            elif disliked_room_types and room_type in disliked_room_types:
                reward = -1
            else:
                reward = 0
            results.append((i, reward, room_type))
        else:
            results.append((i, 0, None))

    return (
        results,
        f"Room type preferences evaluated: liked={liked_room_types}, disliked={disliked_room_types}",
    )


def evaluate_specific_liked_accommodations(query_data, tested_data):
    """
    Evaluates specific liked accommodations.
    For every day except the last day, we earn:
    - (+1) if the day's accommodation is in the specific liked accommodations list
    - 0 if the day's accommodation is unscheduled or not in the specific liked accommodations list

    Returns:
        List of tuples: (day, reward, actual_accommodation_name)
            Empty list [] if there are no specific liked accommodations
        Summary string: f"Specific liked accommodations evaluated: specific_liked={specific_liked_accommodations}"
    """
    preferences = query_data.get("preferences", {})
    specific_liked_accommodations = preferences.get("specific_liked_accommodations")
    if not specific_liked_accommodations:
        return None, "No specific liked accommodations specified"

    results = []

    def get_name(accommodation_field):
        name, city = get_valid_name_city(accommodation_field)
        return name

    for i in range(min(query_data["days"], len(tested_data)) - 1):
        unit = tested_data[i]
        if unit.get("accommodation") and unit["accommodation"] != "-":
            accommodation_name = get_name(unit["accommodation"])
            if accommodation_name in specific_liked_accommodations:
                results.append((i, 1, accommodation_name))
            else:
                results.append((i, 0, accommodation_name))
        else:
            results.append((i, 0, None))
    return (
        results,
        f"Specific liked accommodations evaluated: specific_liked={specific_liked_accommodations}",
    )


def evaluate_specific_disliked_accommodations(query_data, tested_data):
    """
    Evaluates specific disliked accommodations.
    For every day except the last day, we earn:
    - (-1) if the day's accommodation is in the specific disliked accommodations list
    - 0 if the day's accommodation is unscheduled or not in the specific disliked accommodations list

    Returns:
        List of tuples: (day, reward, actual_accommodation_name)
            Empty list [] if there are no specific disliked accommodations
        Summary string: f"Specific disliked accommodations evaluated: specific_disliked={specific_disliked_accommodations}"
    """
    preferences = query_data.get("preferences", {})
    specific_disliked_accommodations = preferences.get(
        "specific_disliked_accommodations"
    )
    if not specific_disliked_accommodations:
        return None, "No specific disliked accommodations specified"

    results = []

    def get_name(accommodation_field):
        name, city = get_valid_name_city(accommodation_field)
        return name

    for i in range(min(query_data["days"], len(tested_data)) - 1):
        unit = tested_data[i]
        if unit.get("accommodation") and unit["accommodation"] != "-":
            accommodation_name = get_name(unit["accommodation"])
            if accommodation_name in specific_disliked_accommodations:
                results.append((i, -1, accommodation_name))
            else:
                results.append((i, 0, accommodation_name))
        else:
            results.append((i, 0, None))
    return (
        results,
        f"Specific disliked accommodations evaluated: specific_disliked={specific_disliked_accommodations}",
    )


def evaluate_attraction_type_preferences(query_data, tested_data):
    """
    Evaluates attraction type preferences.
    For every attraction, we earn:
    - Positive reward (+1) if the attraction has any liked attraction types
    - Negative reward (-1) if the attraction has any disliked attraction types
    - 0 if the attraction is unscheduled or the attraction has no attraction types
    - 0 if the attraction has both a liked attraction type and a disliked attraction type

    This assumes the global maximum number of attractions per day is 5.

    Returns:
        List of tuples: ((day, attraction_ix), reward, actual_attraction_type)
            Empty list [] if there are no attraction type preferences
            attraction_ix varies from 0 to 4 for each day (5 maximum attractions per day)
        Summary string: f"Attraction type preferences evaluated: liked={liked_attraction_types}, disliked={disliked_attraction_types}"
    """
    preferences = query_data.get("preferences", {})
    liked_attraction_types = preferences.get("liked_attraction_types")
    disliked_attraction_types = preferences.get("disliked_attraction_types")

    if not liked_attraction_types and not disliked_attraction_types:
        return None, "No attraction type preferences specified"

    (
        flight,
        accommodation,
        restaurants,
        googleDistanceMatrix,
        attractions,
        city_state_map,
    ) = get_tools()

    results = []

    def get_attraction_types(attraction_field):
        name, city = get_valid_name_city(attraction_field)
        res = attractions.data[
            (attractions.data["name"].astype(str).str.contains(re.escape(name)))
            & (attractions.data["city"] == city)
        ]
        if len(res) > 0:
            return eval(res.iloc[0]["attraction_type"])
        return []

    for i in range(min(query_data["days"], len(tested_data))):
        unit = tested_data[i]
        if unit.get("attraction") and unit["attraction"] != "-":
            attractions_list = get_attractions(unit)
            for attraction_ix in range(5):
                if attraction_ix < len(attractions_list):
                    attraction = attractions_list[attraction_ix]
                    attraction_types = get_attraction_types(attraction)
                    reward = 0
                    # (both cases can happen if an attraction has multiple types)
                    if liked_attraction_types and any(
                        attraction_type in liked_attraction_types
                        for attraction_type in attraction_types
                    ):
                        reward += 1
                    if disliked_attraction_types and any(
                        attraction_type in disliked_attraction_types
                        for attraction_type in attraction_types
                    ):
                        reward -= 1
                    results.append(((i, attraction_ix), reward, attraction_types))
                else:
                    results.append(((i, attraction_ix), 0, None))
        else:
            for attraction_ix in range(5):
                results.append(((i, attraction_ix), 0, None))

    return (
        results,
        f"Attraction type preferences evaluated: liked={liked_attraction_types}, disliked={disliked_attraction_types}",
    )


def evaluate_specific_liked_attractions(query_data, tested_data):
    """
    Evaluates specific liked attractions.
    For every attraction, we earn:
    - (+1) if the attraction is in the specific liked attractions list
    - 0 if the attraction is unscheduled or not in the specific liked attractions list

    Returns:
        List of tuples: ((day, attraction_ix), reward, actual_attraction_name)
            Empty list [] if there are no specific liked attractions
            attraction_ix varies from 0 to 4 for each day (5 maximum attractions per day)
        Summary string: f"Specific liked attractions evaluated: specific_liked={specific_liked_attractions}"
    """
    preferences = query_data.get("preferences", {})
    specific_liked_attractions = preferences.get("specific_liked_attractions")
    if not specific_liked_attractions:
        return None, "No specific liked attractions specified"

    results = []

    def get_name(attraction_field):
        name, city = get_valid_name_city(attraction_field)
        return name

    for i in range(min(query_data["days"], len(tested_data))):
        unit = tested_data[i]
        if unit.get("attraction") and unit["attraction"] != "-":
            attractions_list = get_attractions(unit)
            for attraction_ix in range(5):
                if attraction_ix < len(attractions_list):
                    attraction = attractions_list[attraction_ix]
                    attraction_name = get_name(attraction)
                    if attraction_name in specific_liked_attractions:
                        results.append(((i, attraction_ix), 1, attraction_name))
                    else:
                        results.append(((i, attraction_ix), 0, attraction_name))
                else:
                    results.append(((i, attraction_ix), 0, None))
        else:
            for attraction_ix in range(5):
                results.append(((i, attraction_ix), 0, None))
    return (
        results,
        f"Specific liked attractions evaluated: specific_liked={specific_liked_attractions}",
    )


def evaluate_specific_disliked_attractions(query_data, tested_data):
    """
    Evaluates specific disliked attractions.
    For every attraction, we earn:
    - (-1) if the attraction is in the specific disliked attractions list
    - 0 if the attraction is unscheduled or not in the specific disliked attractions list

    Returns:
        List of tuples: ((day, attraction_ix), reward, actual_attraction_name)
            Empty list [] if there are no specific disliked attractions
            attraction_ix varies from 0 to 4 for each day (5 maximum attractions per day)
        Summary string: f"Specific disliked attractions evaluated: specific_disliked={specific_disliked_attractions}"
    """
    preferences = query_data.get("preferences", {})
    specific_disliked_attractions = preferences.get("specific_disliked_attractions")
    if not specific_disliked_attractions:
        return None, "No specific disliked attractions specified"

    results = []

    def get_name(attraction_field):
        name, city = get_valid_name_city(attraction_field)
        return name

    for i in range(min(query_data["days"], len(tested_data))):
        unit = tested_data[i]
        if unit.get("attraction") and unit["attraction"] != "-":
            attractions_list = get_attractions(unit)
            for attraction_ix in range(5):
                if attraction_ix < len(attractions_list):
                    attraction = attractions_list[attraction_ix]
                    attraction_name = get_name(attraction)
                    if attraction_name in specific_disliked_attractions:
                        results.append(((i, attraction_ix), -1, attraction_name))
                    else:
                        results.append(((i, attraction_ix), 0, attraction_name))
                else:
                    results.append(((i, attraction_ix), 0, None))
        else:
            for attraction_ix in range(5):
                results.append(((i, attraction_ix), 0, None))
    return (
        results,
        f"Specific disliked attractions evaluated: specific_disliked={specific_disliked_attractions}",
    )


def evaluate_attraction_repeats(question, tested_data):
    """
    Evaluates attraction repeats.
    For every attraction, we earn:
    - (-1) if the attraction is a repeat from a previous day
    - 0 if the attraction is not a repeat from a previous day or the attraction is unscheduled

    Returns:
        List of tuples: ((day, attraction_ix), reward, first_occurrence)
            Empty list [] if there are no attraction repeats
        Summary string: f"Attraction repeats evaluated"
    """
    attractions_list = {}  # {attraction_name: first_occurrence}
    results = []

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]

        if "attraction" in unit and unit["attraction"] and unit["attraction"] != "-":
            attractions = get_attractions(unit)
            for attraction_ix in range(5):
                if attraction_ix < len(attractions):
                    attraction = attractions[attraction_ix]
                    if attraction not in attractions_list:
                        attractions_list[attraction] = (i, attraction_ix)
                        results.append(((i, attraction_ix), 0, None))
                    else:
                        results.append(
                            ((i, attraction_ix), -1, attractions_list[attraction])
                        )
                else:
                    results.append(((i, attraction_ix), 0, None))
        else:
            for attraction_ix in range(5):
                results.append(((i, attraction_ix), 0, None))

    return results, "Evaluated for attraction repeats"


def evaluation(query_data, tested_data):
    """
    Main evaluation function that runs all preference checks with set-based constraints:
    - Evaluates tag preferences (liked/disliked) with boolean rewards/penalties
    - Evaluates room type preferences (liked/disliked) with boolean rewards/penalties
    - Evaluates attraction type preferences (liked/disliked) with boolean rewards/penalties
    - Evaluates activity level preferences with distance-based rewards
    - Evaluates restaurant ratings with hinge rewards
    - Evaluates accommodation reviews with hinge rewards
    - Evaluates specific liked/disliked restaurants, accommodations, and attractions

    Args:
        query_data: Query data containing preferences
        tested_data: Test data to evaluate

    Returns a dictionary with results of all preference checks and total score
    """
    return_info = {}

    # Run all preference evaluations
    return_info["tag_results"] = evaluate_tag_preferences(query_data, tested_data)
    return_info["specific_liked_restaurant_results"] = (
        evaluate_specific_liked_restaurants(query_data, tested_data)
    )
    return_info["specific_disliked_restaurant_results"] = (
        evaluate_specific_disliked_restaurants(query_data, tested_data)
    )
    return_info["room_type_results"] = evaluate_room_type_preferences(
        query_data, tested_data
    )
    return_info["attraction_type_results"] = evaluate_attraction_type_preferences(
        query_data, tested_data
    )
    return_info["restaurant_ratings_results"] = evaluate_restaurant_ratings(
        query_data, tested_data
    )
    return_info["accommodation_reviews_results"] = evaluate_accommodation_reviews(
        query_data, tested_data
    )
    return_info["specific_liked_accommodations_results"] = (
        evaluate_specific_liked_accommodations(query_data, tested_data)
    )
    return_info["specific_disliked_accommodations_results"] = (
        evaluate_specific_disliked_accommodations(query_data, tested_data)
    )
    return_info["specific_liked_attractions_results"] = (
        evaluate_specific_liked_attractions(query_data, tested_data)
    )
    return_info["specific_disliked_attractions_results"] = (
        evaluate_specific_disliked_attractions(query_data, tested_data)
    )
    return_info["restaurant_repeats_results"] = evaluate_restaurant_repeats(
        query_data, tested_data
    )
    return_info["attraction_repeats_results"] = evaluate_attraction_repeats(
        query_data, tested_data
    )
    return_info["restaurant_attributes_results"] = evaluate_restaurant_attributes(
        query_data, tested_data
    )
    return return_info


def compute_linear_reward(
    weights: Dict[str, float], preferences_info: Dict[str, List[Tuple[str, float, Any]]]
):
    """
    Compute the linear reward for the preferences.

    Each preference type is evaluated to a list of tuples, each containing (instance_id, reward, metadata)
    This is associated with a weight.

    The score is the sum of the product of the weight and the reward for each preference type.
    The maximum possible score is the sum of the product of the weight and the oracle reward for each preference type.

    Args:
        weights: A dictionary mapping preference keys to their weights
        preferences_info: A dictionary mapping preference keys to their evaluation results

    Returns:
        A tuple containing the score and the maximum possible score
    """
    assert preferences_info is not None

    # Compute the score
    all_weights = []
    all_rewards = []
    all_oracle_rewards = []
    worst_case_rewards = []

    INFO_KEY_TO_WEIGHT_KEY = {
        "tag_results": "tags",
        "attraction_type_results": "attraction_types",
        "room_type_results": "room_types",
        "restaurant_ratings_results": "restaurant_ratings",
        "accommodation_reviews_results": "accommodation_reviews",
        "specific_liked_restaurant_results": "specific_liked_restaurants",
        "specific_disliked_restaurant_results": "specific_disliked_restaurants",
        "specific_liked_accommodations_results": "specific_liked_accommodations",
        "specific_disliked_accommodations_results": "specific_disliked_accommodations",
        "specific_liked_attractions_results": "specific_liked_attractions",
        "specific_disliked_attractions_results": "specific_disliked_attractions",
        "restaurant_repeats_results": "restaurant_repeats",
        "attraction_repeats_results": "attraction_repeats",
        "restaurant_attributes_results": "restaurant_attributes",
    }
    INFO_KEY_TO_ORACLE_REWARD = {  # what is the best score? 1 if you can earn a positive reward; 0 if you are just trying to avoid a negative reward
        "tag_results": 1,
        "attraction_type_results": 1,
        "room_type_results": 1,
        "restaurant_ratings_results": 1,
        "accommodation_reviews_results": 1,
        "specific_liked_restaurant_results": 1,
        "specific_disliked_restaurant_results": 0,
        "specific_liked_accommodations_results": 1,
        "specific_disliked_accommodations_results": 0,
        "specific_liked_attractions_results": 1,
        "specific_disliked_attractions_results": 0,
        "restaurant_repeats_results": 0,
        "attraction_repeats_results": 0,
        "restaurant_attributes_results": 1,
    }
    INFO_KEY_TO_WORST_CASE_REWARD = {  # what is the worst possible score? 0 if you can earn a positive reward; -1 if you are just trying to avoid a negative reward
        "tag_results": -1,  # disliked tags possible
        "attraction_type_results": -1,  # disliked attraction types possible
        "room_type_results": -1,  # disliked room types possible
        "restaurant_ratings_results": -1,  # disliked restaurant ratings possible
        "accommodation_reviews_results": -1,  # -1 if < min_reviews
        "specific_liked_restaurant_results": 0,
        "specific_disliked_restaurant_results": -1,
        "specific_liked_accommodations_results": 0,
        "specific_disliked_accommodations_results": -1,
        "specific_liked_attractions_results": 0,
        "specific_disliked_attractions_results": -1,
        "restaurant_repeats_results": -1,
        "attraction_repeats_results": -1,
        "restaurant_attributes_results": -1,
    }

    for inf_key, weight_key in INFO_KEY_TO_WEIGHT_KEY.items():
        if inf_key in preferences_info and preferences_info[inf_key][0] is not None:
            inf_results = preferences_info[inf_key][0]
            if weights.get(weight_key) is not None:
                _w = weights[weight_key]
                all_weights.extend([_w] * len(inf_results))
                all_rewards.extend([r[1] for r in inf_results])
                all_oracle_rewards.extend(
                    [INFO_KEY_TO_ORACLE_REWARD[inf_key]] * len(inf_results)
                )
                worst_case_rewards.extend(
                    [INFO_KEY_TO_WORST_CASE_REWARD[inf_key]] * len(inf_results)
                )

    all_weights = np.array(all_weights)
    score = np.sum(all_weights * all_rewards)
    min_score = np.sum(all_weights * worst_case_rewards)
    max_score = np.sum(all_weights * all_oracle_rewards)
    return score, min_score, max_score
