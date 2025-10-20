from data.travel_planner.db import TravelDB
from utils.misc import parse_json
from data.travel_planner.reward_utils.tp_utils.func import (
    extract_from_to,
    extract_before_parenthesis,
    get_valid_name_city,
)
from data.travel_planner.reward_utils.evaluation.hard_constraint import get_total_cost
from typing import Dict, Any, List


def parse_travel_plan(
    yhat: str, include_info: bool = False, db: TravelDB = None, people_number: int = 1
) -> dict:
    """
    Parse a travel plan from a JSON string.
    """
    plan = parse_json(yhat)
    if plan is None:
        print(f"Error parsing travel plan: {yhat}")
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
            day["transportation"] = {
                "name": transportation,
                "time": _get_transportation_time_slot(
                    transportation, day["current_city"]
                ),
                "invalid": False,
                "city1": city1,
                "city2": city2,
                "type": "flight"
                if "flight" in transportation.lower()
                else "self-driving"
                if "self-driving" in transportation.lower()
                else "taxi",
            }
        else:
            day["transportation"] = None

    _update_dicts_with_icons(plan, db)
    return plan


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
