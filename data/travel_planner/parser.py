from data.travel_planner.db import TravelDB
from utils.misc import parse_json
from data.travel_planner.reward_utils.tp_utils.func import (
    extract_from_to,
    extract_before_parenthesis,
    get_valid_name_city,
)
from data.travel_planner.reward_utils.evaluation.hard_constraint import get_total_cost
from typing import Dict, Any, List, Union
import re


def parse_travel_plan(
    yhat: str,
    include_info: bool = False,
    db: TravelDB = None,
    people_number: int = 1,
    driving_info: Union[Dict[str, str], List[Dict[str, str]]] = None,
) -> dict:
    """
    Parse a travel plan from a JSON string.
    """
    from utils.misc import parse_for_answer_tags

    # First try to parse from <travel_plan> tags
    travel_plan_content = parse_for_answer_tags(
        yhat, keyword="travel_plan", return_none_if_not_found=True
    )
    if travel_plan_content:
        plan = parse_json(travel_plan_content)
    else:
        # Fall back to parsing JSON directly (for backward compatibility)
        plan = parse_json(yhat)

    if plan is None:
        print(f"Error parsing travel plan: {yhat}")
        return None

    if not isinstance(plan, list) or not all(isinstance(day, dict) for day in plan):
        print(f"Error parsing travel plan: {plan}")
        return None

    # Do some automatic correction for missing fields
    for day in plan:
        for field in [
            "breakfast",
            "lunch",
            "dinner",
            "attraction",
            "accommodation",
            "transportation",
        ]:
            if field not in day:
                day[field] = "-"

    if not include_info:
        return plan

    # Add info to the plan
    for day in plan:
        day["total_cost"] = _calculate_daily_cost(day, people_number)

        # Restaurants
        for meal in ["breakfast", "lunch", "dinner"]:
            if day[meal] != "-":
                restaurant_name, city = get_valid_name_city(day[meal])
                restaurant_info = db.get_restaurant_info(restaurant_name, city)
                if restaurant_info is None:
                    day[meal] = {
                        "name": restaurant_name,
                        "city": city,
                        "invalid": True,
                    }
                else:
                    day[meal] = {**restaurant_info, "invalid": False}
            else:
                day[meal] = None

        # Attractions
        attractions = day["attraction"]
        if attractions != "-":
            attraction_list = [
                attr.strip() for attr in attractions.split(";") if attr.strip()
            ]
            attraction_infos = []
            for ix, attraction in enumerate(attraction_list):
                attraction_name, city = get_valid_name_city(attraction)
                try:
                    attraction_info = db.get_attraction_info(attraction_name, city)
                    assert attraction_info is not None
                    invalid = False
                except Exception:
                    attraction_info = {"name": attraction_name, "city": city}
                    invalid = True
                finally:
                    attraction_infos.append({**attraction_info, "invalid": invalid})
            day["attraction"] = attraction_infos
        else:
            day["attraction"] = []

        # Accommodations
        accommodation = day["accommodation"]
        if accommodation != "-":
            accommodation_name, city = get_valid_name_city(accommodation)
            accommodation_info = db.get_accommodation_info(accommodation_name, city)
            if accommodation_info is None:
                day["accommodation"] = {
                    "name": accommodation_name,
                    "city": city,
                    "invalid": True,
                }
            else:
                day["accommodation"] = {**accommodation_info, "invalid": False}
        else:
            day["accommodation"] = None

        # Transportation
        transportation = day["transportation"]
        if transportation != "-":
            city1, city2 = extract_from_to(day["current_city"])
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)

            # Determine transportation type
            transport_type = "flight"
            if "self-driving" in transportation.lower():
                transport_type = "self-driving"
            elif "taxi" in transportation.lower():
                transport_type = "taxi"

            # Validate transportation
            invalid = False
            if transport_type == "flight":
                # Extract flight number from transportation string
                flight_number = None
                if "Flight Number:" in transportation:
                    try:
                        flight_number = (
                            transportation.split("Flight Number:")[1]
                            .split(",")[0]
                            .strip()
                        )
                    except (IndexError, ValueError):
                        pass

                # Validate flight exists in database
                if flight_number and db is not None:
                    flights = db.get_flight_info(city1, city2)
                    if flights:
                        # Check if flight number matches any flight
                        flight_found = any(
                            str(flight.get("flight_number", "")).strip()
                            == flight_number
                            for flight in flights
                        )
                        if not flight_found:
                            invalid = True
                    else:
                        # No flights found for this route
                        invalid = True
                elif db is None:
                    # Can't validate without database
                    invalid = False
                else:
                    # Flight number not found in string
                    invalid = True
            else:
                # Validate self-driving or taxi against driving_info
                if driving_info is not None:
                    invalid = not any(
                        d["Content"].lower() == transportation.lower()
                        for d in driving_info
                    )
                # If driving_info is None, we can't validate, so assume valid
            day["transportation"] = {
                "name": transportation,
                "time": _get_transportation_time_slot(
                    transportation, day["current_city"]
                ),
                "invalid": invalid,
                "city1": city1,
                "city2": city2,
                "type": transport_type,
            }
        else:
            day["transportation"] = None

    _update_dicts_with_icons(plan, db)
    return plan


def _remove_duration_from_transportation(transportation: str) -> str:
    """
    Remove duration information from transportation string.

    Examples:
    - "self-driving, from Kansas City to Pensacola, duration: 14 hours 4 mins, distance: 1,433 km, cost: 71"
      -> "self-driving, from Kansas City to Pensacola, distance: 1,433 km, cost: 71"
    - "taxi, from A to B, duration: 2 hours, distance: 100 km, cost: 50"
      -> "taxi, from A to B, distance: 100 km, cost: 50"
    """
    if not transportation:
        return transportation

    # Use regex to remove duration: ... pattern (case-insensitive)
    # Pattern matches "duration:" followed by any characters until the next comma or end of string
    # Handles variations like "duration: 14 hours 4 mins" or "duration:2 hours"
    # Matches both ", duration: ..." (middle) and ", duration: ..." (end of string)
    pattern = r",\s*duration:\s*[^,]+(?=,|$)"
    result = re.sub(pattern, "", transportation, flags=re.IGNORECASE)

    # Also handle case where duration is at the start (before first comma)
    # Pattern: "duration: ... ," at the beginning
    pattern_start = r"^duration:\s*[^,]+,\s*"
    result = re.sub(pattern_start, "", result, flags=re.IGNORECASE)

    # Clean up any double commas or trailing/leading commas
    result = re.sub(r",\s*,", ",", result)  # Remove double commas
    result = result.strip().rstrip(",")  # Remove trailing comma

    return result


def _calculate_daily_cost(day_data: Dict[str, Any], people_number: int) -> float:
    question = {"days": 1, "people_number": people_number}
    return get_total_cost(question, [day_data])


def _get_transportation_time_slot(transportation: str, current_city: str) -> str:
    from datetime import datetime

    if "Departure Time" in transportation:
        try:
            departure_time = transportation.split("Departure Time: ")[1].split(",")[0]
            departure_time = departure_time.replace(" ", "")
            departure_time = datetime.strptime(departure_time, "%H:%M")
            if departure_time.hour < 12:
                return "morning"
            elif departure_time.hour < 18:
                return "afternoon"
            else:
                return "evening"
        except (ValueError, IndexError):
            pass
    if "duration" in transportation:
        try:
            duration = transportation.split("duration: ")[1].split(",")[0]
            duration_hours = int(duration.split("hour")[0])
            if "from" in current_city and "to" in current_city:
                if duration_hours < 6:
                    return "evening"
                elif duration_hours < 9:
                    return "afternoon"
                else:
                    return "morning"
        except (ValueError, IndexError):
            pass
    return None


def _update_dicts_with_icons(
    travel_plan: List[Dict[str, Any]], travel_db: TravelDB
) -> List[dict]:
    if not travel_plan:
        return
    for day_data in travel_plan:
        # Assumes day_data was parsed with include_info=True
        if day_data["transportation"]:
            transport_time = day_data["transportation"]["time"]
            day_data["transportation"].update(
                {"icon": _get_transportation_icon(day_data["transportation"]["name"])}
            )
        else:
            transport_time = None
        is_first_day = day_data.get("days", "Unknown") == 1

        attractions = day_data["attraction"]
        if not attractions:
            continue
        for ix, d in enumerate(attractions):
            icon = ":material/attractions:"
            if not d["invalid"]:
                if "outdoor" in d["attraction_type"]:
                    icon = ":material/park:"
                elif (
                    "shopping" in d["attraction_type"]
                    or "dining" in d["attraction_type"]
                ):
                    icon = ":material/local_mall:"
                elif (
                    "amuseument" in d["attraction_type"]
                    or "zoo" in d["attraction_type"]
                ):
                    icon = ":material/attractions:"
                else:
                    icon = ":material/museum:"
            d.update(
                {
                    "time": "morning"
                    if (ix == 0 and (not is_first_day or transport_time != "morning"))
                    else "afternoon",
                    "icon": icon,
                }
            )


def _get_transportation_icon(transportation: str) -> str:
    if "flight" in transportation.lower():
        return ":material/flight:"
    elif "self-driving" in transportation.lower():
        return ":material/local_taxi:"
    elif "taxi" in transportation.lower():
        return ":material/local_taxi:"
    else:
        return ":material/local_taxi:"
