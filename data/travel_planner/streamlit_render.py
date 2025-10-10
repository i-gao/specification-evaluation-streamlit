from typing import List, Dict, Any, Optional
import streamlit as st


def render_eval_travel(
    *, final_prediction: str, y0: Optional[str], db, people_number: int
):
    """
    Render evaluation UI for Travel Planner custom specs and return (completed, feedback).
    """
    from utils.streamlit_types import FormElement, form_element_to_streamlit
    from data.travel_planner.data import output_to_streamlit, output_to_streamlit_comparison

    st.markdown("## Evaluate the assistant's travel plan")
    with st.container(key="travel_eval_display", width="stretch"):
        try:
            if y0 is not None:
                output_to_streamlit_comparison(y0, final_prediction, db, people_number)
            else:
                output_to_streamlit(final_prediction, db, people_number)
        except Exception as e:
            st.write("Error rendering plans:", str(e))

    form_elements: List[FormElement] = [
        FormElement(
            input_type="text_area",
            label="Describe the pros and cons of the plan in a few sentences.",
            height=120,
        ),
        FormElement(
            input_type="radio",
            label="Do you think more exploration could have led to a better plan?",
            options=["Yes", "Maybe", "No"],
        ),
    ]

    with st.form(key="travel_custom_eval_form"):
        feedback: Dict[str, Any] = {}
        for element in form_elements:
            st_fn, st_kwargs, required = form_element_to_streamlit(element)
            value = st_fn(**st_kwargs)
            label = element.get("label", "question")
            feedback[label] = value
        submit = st.form_submit_button("Submit", type="primary")
        if submit:
            for element in form_elements:
                if element.get("required", False):
                    label = element.get("label")
                    if not feedback.get(label):
                        st.error("Please fill in all required fields.")
                        return False, None
            return True, feedback

    return False, None
from typing import Any, Dict, List
import streamlit as st
import os
import sys
import uuid
from data.travel_planner.db import TravelDB

# Add TravelPlanner to path
travel_planner_path = os.path.join(os.path.dirname(__file__), "reward_utils")
if travel_planner_path not in sys.path:
    sys.path.append(travel_planner_path)

from tp_utils.func import (
    extract_from_to,
    extract_before_parenthesis,
    get_valid_name_city,
)
from evaluation.hard_constraint import get_total_cost


def render_travel_plan_streamlit(
    travel_plan: List[Dict[str, Any]], travel_db: TravelDB, people_number: int
) -> None:
    if not travel_plan:
        st.markdown("*No travel plan data available*")
        return

    st.markdown("### 🗓️ Itinerary at a glance")
    st.markdown("This is a quick overview of your daily schedule.")
    st.markdown(_render_round_trip_check(travel_plan))

    # Calendar/summary table
    st.markdown(
        _render_travel_summary_table(
            travel_plan, travel_db, people_number, header=True, footer=True
        ),
        unsafe_allow_html=True,
    )

    # One expander per day with detailed breakdown
    for day_data in travel_plan:
        day_num = day_data.get("days", "?")
        current_city = day_data.get("current_city", "")
        title = f"Day {day_num} — {current_city}"
        with st.expander(title, expanded=False):
            st.markdown(
                _render_travel_summary_table(
                    [day_data], travel_db, people_number, header=False, footer=False
                ),
                unsafe_allow_html=True,
            )
            # Render detailed sections for this single day by reusing multi-day renderer
            _render_detailed_travel_sections([day_data], travel_db, people_number)


def output_to_streamlit_comparison(
    parsed1: List[Dict[str, Any]],
    parsed2: List[Dict[str, Any]],
    travel_db: TravelDB,
    people_number: int,
    valid1: bool,
    valid2: bool,
    metadata1: Dict[str, Any],
    metadata2: Dict[str, Any],
) -> None:
    tab1, tab2 = st.tabs(["Plan A", "Plan B"])

    with tab1:
        if valid1 is not None:
            if valid1:
                st.markdown(":small[:green[:material/check: Plan A is valid]]")
            else:
                st.markdown(":small[:red[:material/close: Plan A invalid]]\n\n")
                constraints_md = "\n\n".join(
                    [
                        f":small[:red[- {constraint}]]"
                        for constraint in (metadata1 or {}).get(
                            "violated_constraints", []
                        )
                        if constraint is not None
                    ]
                )
                if constraints_md:
                    st.markdown(constraints_md)
        render_travel_plan_streamlit(parsed1, travel_db, people_number)

    with tab2:
        if valid2 is not None:
            if valid2:
                st.markdown(":small[:green[:material/check: Plan B is valid]]")
            else:
                st.markdown(":small[:red[:material/close: Plan B invalid]]\n\n")
                constraints_md = "\n\n".join(
                    [
                        f":small[:red[- {constraint}]]"
                        for constraint in (metadata2 or {}).get(
                            "violated_constraints", []
                        )
                        if constraint is not None
                    ]
                )
                if constraints_md:
                    st.markdown(constraints_md)
        render_travel_plan_streamlit(parsed2, travel_db, people_number)


# ===== Helpers (ported from data.py to avoid circular import) =====


def _is_round_trip(travel_plan: List[Dict[str, Any]]) -> bool:
    day1 = travel_plan[0]
    last_day = travel_plan[-1]
    if "from" not in day1["current_city"]:
        return False, "Should plan to travel to destination on first day"
    if "from" not in last_day["current_city"]:
        return False, "Should plan to travel to origin on last day"
    city1, city2 = extract_from_to(day1["current_city"])
    org = extract_before_parenthesis(city1)
    city2, city3 = extract_from_to(last_day["current_city"])
    dest = extract_before_parenthesis(city3)
    if org != dest:
        return False, f"Must return to origin {org} on last day"
    return True, ""


def _render_round_trip_check(travel_plan: List[Dict[str, Any]]) -> str:
    if not travel_plan:
        return ""
    is_round_trip, error_message = _is_round_trip(travel_plan)
    if is_round_trip:
        return ":green[:material/check: Round trip]"
    else:
        return f":red[:material/close: Not a round trip: {error_message}]"


def _get_transportation_icon(transportation: str) -> str:
    if "flight" in transportation.lower():
        return ":material/flight:"
    elif "self-driving" in transportation.lower():
        return ":material/local_taxi:"
    elif "taxi" in transportation.lower():
        return ":material/local_taxi:"
    else:
        return ":material/local_taxi:"


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


def _render_travel_summary_table(
    travel_plan: List[Dict[str, Any]],
    travel_db: TravelDB,
    people_number: int,
    header: bool = True,
    footer: bool = True,
) -> str:
    if not travel_plan:
        return "*No travel plan data available*"
    lines: List[str] = []
    num_days = len(travel_plan)
    if header:
        header_cells = [""] + [f"**Day {i + 1}**" for i in range(num_days)]
        header_row = "| " + " | ".join(header_cells) + " |"
    else:
        header_row = "| " + "| " * (num_days) + "|"
    separator = "|" + "|".join(["------"] * (num_days + 1)) + "|"
    lines.append(header_row)
    lines.append(separator)

    cities_row = ["**Cities**"]
    for day in travel_plan:
        if "from" in day.get("current_city", ""):
            city1, city2 = extract_from_to(day.get("current_city", ""))
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            cities_row.append(f"{city1}, {city2}")
        else:
            cities_row.append(day.get("current_city", ""))
    lines.append("| " + " | ".join(cities_row) + " |")

    morning_row = ["**Morning**"]
    for day in travel_plan:
        items: List[str] = []
        transportation = day.get("transportation", "-")
        if transportation != "-":
            t_slot = _get_transportation_time_slot(
                transportation, day.get("current_city", "")
            )
            if t_slot == "morning":
                items.append(
                    f"{_get_transportation_icon(transportation)} {transportation}"
                )
        breakfast = day.get("breakfast", "-")
        if breakfast != "-" and "," in breakfast:
            restaurant_name, city = get_valid_name_city(breakfast)
            restaurant_info = travel_db.get_restaurant_info(restaurant_name, city)
            if restaurant_info is None:
                items.append(f":red-background[:material/error: {restaurant_name} ({city})]")
            else:
                items.append(f":material/restaurant: {restaurant_name} ({city})")
        attractions = _get_attractions(day, travel_db)
        for att in attractions:
            if att["time"] == "morning":
                if att.get("invalid", False):
                    items.append(f":red-background[:material/error: {att['name']} ({att['city']})]")
                else:
                    items.append(f"{att['icon']} {att['name']} ({att['city']})")
        morning_row.append("<br>".join(items) if items else "*No morning activities*")
    lines.append("| " + " | ".join(morning_row) + " |")

    afternoon_row = ["**Afternoon**"]
    for day in travel_plan:
        items: List[str] = []
        transportation = day.get("transportation", "-")
        if transportation != "-":
            t_slot = _get_transportation_time_slot(
                transportation, day.get("current_city", "")
            )
            if t_slot == "afternoon":
                items.append(
                    f"{_get_transportation_icon(transportation)} {transportation}"
                )
        lunch = day.get("lunch", "-")
        if lunch != "-" and "," in lunch:
            restaurant_name, city = get_valid_name_city(lunch)
            restaurant_info = travel_db.get_restaurant_info(restaurant_name, city)
            if restaurant_info is None:
                items.append(f":red-background[:material/error: {restaurant_name} ({city})]")
            else:
                items.append(f":material/restaurant: {restaurant_name} ({city})")
        attractions = _get_attractions(day, travel_db)
        for att in attractions:
            if att["time"] == "afternoon":
                if att.get("invalid", False):
                    items.append(f":red-background[:material/error: {att['name']} ({att['city']})]")
                else:
                    items.append(f"{att['icon']} {att['name']} ({att['city']})")
        afternoon_row.append(
            "<br>".join(items) if items else "*No afternoon activities*"
        )
    lines.append("| " + " | ".join(afternoon_row) + " |")

    evening_row = ["**Evening**"]
    for day in travel_plan:
        items: List[str] = []
        transportation = day.get("transportation", "-")
        if transportation != "-":
            t_slot = _get_transportation_time_slot(
                transportation, day.get("current_city", "")
            )
            if t_slot == "evening":
                items.append(
                    f"{_get_transportation_icon(transportation)} {transportation}"
                )
        dinner = day.get("dinner", "-")
        if dinner != "-" and "," in dinner:
            restaurant_name, city = get_valid_name_city(dinner)
            restaurant_info = travel_db.get_restaurant_info(restaurant_name, city)
            if restaurant_info is None:
                items.append(f":red-background[:material/error: {restaurant_name} ({city})]")
            else:
                items.append(f":material/restaurant: {restaurant_name} ({city})")
        evening_row.append("<br>".join(items) if items else "*No evening activities*")
    lines.append("| " + " | ".join(evening_row) + " |")

    accommodation_row = ["**Accommodation**"]
    for day in travel_plan:
        accommodation = day.get("accommodation", "-")
        if accommodation != "-" and "," in accommodation:
            hotel_name, city = get_valid_name_city(accommodation)
            hotel_info = travel_db.get_accommodation_info(hotel_name, city)
            if hotel_info is None:
                accommodation_row.append(f":red-background[:material/error: {hotel_name} ({city})]")
            else:
                accommodation_row.append(f":material/hotel: {hotel_name} ({city})")
        else:
            accommodation_row.append("*No accommodation*")
    lines.append("| " + " | ".join(accommodation_row) + " |")

    daily_costs_row = ["**Estimated Spend**"]
    total_cost = 0
    for day in travel_plan:
        day_cost = _calculate_daily_cost(day, people_number)
        total_cost += day_cost
        daily_costs_row.append(f"**\${day_cost:.2f}**")
    lines.append("| " + " | ".join(daily_costs_row) + " |")

    if footer:
        total_row = (
            [""]
            + [""] * (num_days - 1)
            + [
                f"**Total Cost: \${total_cost:.2f}**<br>This does not include the cost of attractions."
            ]
        )
        lines.append("| " + " | ".join(total_row) + " |")
    return "\n".join(lines)


def _render_detailed_travel_sections(
    travel_plan: List[Dict[str, Any]], travel_db: TravelDB, people_number: int
) -> None:
    if not travel_plan:
        st.markdown("*No travel plan data available*")
        return

    for day_data in travel_plan:
        try:
            day_num = day_data.get("days", "Unknown")
            # Extract city information (currently unused but kept for potential future use)
            if "from" in day_data.get("current_city", ""):
                city1, city2 = extract_from_to(day_data.get("current_city", ""))
                city1 = extract_before_parenthesis(city1)
                city2 = extract_before_parenthesis(city2)
                # cities = [city1, city2]  # Unused for now
            else:
                # cities = [day_data.get("current_city", "")]  # Unused for now
                pass

            # Collect all items for this day
            all_items = []

            # Add meals
            for meal_type in ["breakfast", "lunch", "dinner"]:
                meal = day_data.get(meal_type, "-")
                if meal != "-":
                    restaurant_name, city = get_valid_name_city(meal)
                    restaurant = travel_db.get_restaurant_info(restaurant_name, city)
                    if restaurant:
                        all_items.append(
                            {
                                "type": "restaurant",
                                "title": f":material/restaurant: {meal_type.capitalize()}: {restaurant['name']} ({_get_dollar_signs(restaurant['average_cost'])})",
                                "data": restaurant,
                                "meal_type": meal_type,
                            }
                        )
                    else:
                        all_items.append(
                            {
                                "type": "restaurant",
                                "title": f":material/restaurant: {meal_type.capitalize()}: Invalid restaurant",
                                "data": {"name": meal, "invalid": True},
                                "meal_type": meal_type,
                            }
                        )

            # Add attractions
            attractions = day_data.get("attraction", "-")
            if attractions != "-":
                for attraction in _get_attractions(day_data, travel_db):
                    all_items.append(
                        {
                            "type": "attraction",
                            "title": f"{attraction['icon']} {attraction['name']}",
                            "data": attraction,
                        }
                    )

            # Add accommodation
            accommodation = day_data.get("accommodation", "-")
            if accommodation != "-":
                if "," in accommodation:
                    hotel_name, city = [
                        part.strip() for part in accommodation.split(",", 1)
                    ]
                    hotel_info = travel_db.get_accommodation_info(hotel_name, city)
                    if hotel_info:
                        all_items.append(
                            {
                                "type": "accommodation",
                                "title": f":material/hotel: Accommodation: {hotel_info['name']}",
                                "data": hotel_info,
                            }
                        )
                    else:
                        all_items.append(
                            {
                                "type": "accommodation",
                                "title": ":material/hotel: Accommodation: Invalid accommodation",
                                "data": {"name": accommodation, "invalid": True},
                            }
                        )

            # Render items in 3-column grid
            if all_items:
                st.markdown("### Details")
                st.markdown("Click on any item to view its details.")

                cols = st.columns(3)
                for i, item in enumerate(all_items):
                    with cols[i % 3]:
                        if item["type"] == "restaurant":
                            _render_restaurant_button(item, day_num)
                        elif item["type"] == "attraction":
                            _render_attraction_button(item, day_num)
                        elif item["type"] == "accommodation":
                            _render_accommodation_button(item, day_num)

        except Exception as e:
            st.markdown(
                f"*The assistant output was not formatted correctly for day {day_data.get('days', 'Unknown')}: {str(e)}*"
            )
        st.markdown("---")


def _get_attractions(day_data: Dict[str, Any], travel_db: TravelDB) -> List[dict]:
    attractions = day_data.get("attraction", "-")
    if attractions == "-":
        return []
    attraction_list = [attr.strip() for attr in attractions.split(";") if attr.strip()]
    transport_time = _get_transportation_time_slot(
        day_data.get("transportation", "-"), day_data.get("current_city", "")
    )
    is_first_day = day_data.get("days", "Unknown") == 1
    res: List[dict] = []
    for ix, attraction in enumerate(attraction_list):
        attraction_name, city = get_valid_name_city(attraction)
        try:
            attraction_info = travel_db.get_attraction_info(attraction_name, city)
            invalid = False
            attraction_type = attraction_info["attraction_type"]
            if "outdoor" in attraction_type:
                icon = ":material/park:"
            elif "shopping" in attraction_type or "dining" in attraction_type:
                icon = ":material/local_mall:"
            elif "amuseument" in attraction_type or "zoo" in attraction_type:
                icon = ":material/attractions:"
            else:
                icon = ":material/museum:"
        except Exception:
            attraction_info = None
            invalid = True
            icon = ":material/attractions:"
        finally:
            res.append(
                {
                    "string": attraction,
                    "name": attraction_name,
                    "city": city,
                    "info": attraction_info,
                    "time": "morning"
                    if (ix == 0 and (not is_first_day or transport_time != "morning"))
                    else "afternoon",
                    "icon": icon,
                    "invalid": invalid,
                }
            )
    return res


def _calculate_daily_cost(day_data: Dict[str, Any], people_number: int) -> float:
    question = {"days": 1, "people_number": people_number}
    return get_total_cost(question, [day_data])


def _clean_camelcase(s: str) -> str:
    if not s:
        return ""
    s = s.replace("_", " ")
    import re as _re

    s = _re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    parts = s.split()
    parts = [p.lower() for p in parts]
    s = " ".join(parts).title()
    return s.capitalize()


def _containsDigit(s: str) -> bool:
    return any(char.isdigit() for char in s)


def _fmt(v: Any) -> str:
    try:
        v = eval(v)
    except Exception:
        pass
    if isinstance(v, str):
        return v.replace("_", " ")
    elif isinstance(v, list):
        return ", ".join(v)
    elif isinstance(v, dict):
        return ", ".join([ki for ki, vi in v.items() if vi is True])
    else:
        return str(v)


ICON_TO_EMOJI = {
    ":material/park:": "🌳",
    ":material/attractions:": "🎠",
    ":material/museum:": "🏛️",
    ":material/local_mall:": "🛍️",
}

def _render_restaurant_button(item: Dict[str, Any], day_num: str) -> None:
    """Render a restaurant button with dialog."""

    @st.dialog(f"Restaurant Details - Day {day_num}", width="large")
    def _show_restaurant_dialog(
        restaurant_data: Dict[str, Any], meal_type: str
    ) -> None:
        st.markdown(
            _restaurant_to_dialog_content(restaurant_data, meal_type),
            unsafe_allow_html=True,
        )

    st.button(
        item["title"],
        on_click=_show_restaurant_dialog,
        args=(item["data"], item["meal_type"]),
        key=f"restaurant_{item['data'].get('name', 'unknown')}_{day_num}",
        use_container_width=True,
    )


def _render_attraction_button(item: Dict[str, Any], day_num: str) -> None:
    """Render an attraction button with dialog."""

    @st.dialog(f"Attraction Details - Day {day_num}", width="large")
    def _show_attraction_dialog(attraction_data: Dict[str, Any]) -> None:
        st.markdown(
            _attraction_to_dialog_content(attraction_data),
            unsafe_allow_html=True,
        )

    st.button(
        item["title"],
        on_click=_show_attraction_dialog,
        args=(item["data"],),
        key=f"attraction_{item['data'].get('name', 'unknown')}_{day_num}",
        use_container_width=True,
    )


def _render_accommodation_button(item: Dict[str, Any], day_num: str) -> None:
    """Render an accommodation button with dialog."""

    @st.dialog(f"Accommodation Details - Day {day_num}", width="large")
    def _show_accommodation_dialog(accommodation_data: Dict[str, Any]) -> None:
        st.markdown(
            _accommodation_to_dialog_content(accommodation_data),
            unsafe_allow_html=True,
        )

    st.button(
        item["title"],
        on_click=_show_accommodation_dialog,
        args=(item["data"],),
        key=f"accommodation_{item['data'].get('name', 'unknown')}_{day_num}",
        use_container_width=True,
    )


def _restaurant_to_dialog_content(restaurant: Dict[str, Any], meal_type: str) -> str:
    """Convert restaurant data to dialog content (similar to collapsible but without details tags)."""
    if restaurant.get("invalid"):
        return f"<b>:material/restaurant: {meal_type.capitalize()}: Invalid restaurant</b><br><br>:material/error: '{restaurant['name']}' is an invalid restaurant that is not in the database."

    lines: List[str] = []
    lines.append(
        f"<b>:material/restaurant: {meal_type.capitalize()}: {restaurant['name']} ({_get_dollar_signs(restaurant['average_cost'])})</b>"
    )
    lines.append("")

    basic_info: List[str] = []
    if restaurant.get("rating"):
        basic_info.append(f"**Rating:** {restaurant['rating']}⭐")
    if restaurant.get("review_count"):
        basic_info.append(f"**Reviews:** {restaurant['review_count']}")
    if restaurant.get("average_cost"):
        basic_info.append(f"**Cost per person:** \${restaurant['average_cost']:.2f}")

    tags: List[str] = []
    if restaurant.get("tags"):
        try:
            tags.append(f"**Tags:** {', '.join(eval(restaurant['tags']))}")
        except Exception:
            pass

    features: List[str] = []
    for feat_key, label in [
        ("has_takeout", "Takeout"),
        ("has_delivery", "Delivery"),
        ("has_reservations", "Reservations"),
        ("has_outdoor_seating", "Outdoor Seating"),
        ("has_wifi", "WiFi"),
        ("good_for_groups", "Good for Groups"),
        ("good_for_kids", "Kid-Friendly"),
    ]:
        if restaurant.get(feat_key):
            features.append(label)

    attributes: List[str] = []
    if restaurant.get("attributes"):
        try:
            parsed_attributes = eval(restaurant["attributes"])
        except Exception:
            parsed_attributes = {}
        if parsed_attributes:
            attributes.append("**Attributes:**")
            attributes.append("")
            for k, v in parsed_attributes.items():
                if not _containsDigit(k) and _fmt(v).strip() != "":
                    attributes.append(f"  * {_clean_camelcase(k)}: {_fmt(v)}")

    if basic_info:
        lines.append(" | ".join(basic_info))
        lines.append("")
    if tags:
        lines.append("\n\n".join(tags))
        lines.append("")
    if features:
        lines.append("**Features:** " + ", ".join(features))
        lines.append("")
    if attributes:
        lines.append("\n".join(attributes))
        lines.append("")

    return "\n".join(lines)


def _attraction_to_dialog_content(attraction: Dict[str, Any]) -> str:
    """Convert attraction data to dialog content (similar to collapsible but without details tags)."""
    if attraction.get("invalid"):
        return f"<b>🎯 Invalid attraction</b><br><br>:material/error: '{attraction['string']}' is an invalid attraction that is not in the database."

    lines: List[str] = []
    basic_info: List[str] = []
    if attraction["info"] and attraction["info"].get("attraction_type"):
        types_val = attraction["info"]["attraction_type"]
        types_str = (
            ", ".join([_fmt(t).capitalize() for t in types_val])
            if isinstance(types_val, list)
            else _fmt(types_val)
        )
        basic_info.append(types_str)
    if attraction["info"] and attraction["info"].get("activity_level"):
        basic_info.append(f"**Activity Level:** {attraction['info']['activity_level']}")

    lines.append(f"<b>{attraction['icon']} {attraction['name']}</b>")
    lines.append("")

    if basic_info:
        lines.append(" | ".join(basic_info))
        lines.append("")
    if attraction["info"] and attraction["info"].get("address"):
        lines.append("**Address:**")
        lines.append(f"{attraction['info']['address']}")
        lines.append("")
    if attraction["info"] and attraction["info"].get("description"):
        lines.append(f"{attraction['info']['description']}")
        lines.append("")

    return "\n".join(lines)


def _accommodation_to_dialog_content(accommodation: Dict[str, Any]) -> str:
    """Convert accommodation data to dialog content (similar to collapsible but without details tags)."""
    if accommodation.get("invalid"):
        return f"<b>:material/hotel: Invalid accommodation</b><br><br>:material/error: '{accommodation['name']}' is an invalid accommodation that is not in the database."

    lines: List[str] = []
    basic_info: List[str] = []
    if accommodation.get("room_type"):
        basic_info.append(f"**Stay type:** {accommodation['room_type']}")
    if accommodation.get("price"):
        basic_info.append(f"**Price per night:** \${accommodation['price']:.2f}")
    if accommodation.get("num_reviews"):
        basic_info.append(f"**Reviews:** {accommodation['num_reviews']}")

    additional_info: List[str] = []
    if accommodation.get("minimum_nights"):
        additional_info.append(
            f"**Minimum nights stay to book:** {accommodation['minimum_nights']}"
        )
    if accommodation.get("maximum_occupancy"):
        additional_info.append(
            f"**Maximum Occupancy:** {accommodation['maximum_occupancy']}"
        )

    lines.append(f"<b>:material/hotel: {accommodation['name']}</b>")
    lines.append("")

    if basic_info:
        lines.append(" | ".join(basic_info))
        lines.append("")
    if additional_info:
        lines.append(" | ".join(additional_info))
        lines.append("")
    if accommodation.get("house_rules"):
        lines.append("**House Rules:**")
        lines.append(f"{accommodation['house_rules']}")
        lines.append("")

    return "\n".join(lines)


def _get_dollar_signs(price: float) -> str:
    if price < 15:
        return "$"
    elif price < 30:
        return "$$"
    elif price < 50:
        return "$$$"
    else:
        return "$$$$"


def render_travel_mentions(travel_names: List[str], db: TravelDB) -> None:
    """
    Render a section showing mentioned travel items (restaurants, attractions, accommodations) with their details.
    Uses a grid of buttons that launch dialogs, similar to workout planning exercises and meal planning recipes.
    """
    if not travel_names:
        return
    
    st.markdown('<div id="options-mentioned-in-message"></div>', unsafe_allow_html=True)
    st.markdown("Click on an item to view its details, including location, ratings, prices, and other information.")
    
    # Get unique travel items (first occurrence only as requested)
    seen_items = set()
    unique_items = []
    for travel_name in travel_names:
        if travel_name not in seen_items:
            seen_items.add(travel_name)
            unique_items.append(travel_name)
    
    if not unique_items:
        return
    
    # Create a grid of buttons (3 columns)
    cols = st.columns(3)
    for i, travel_name in enumerate(unique_items):
        with cols[i % 3]:
            try:
                # Try to find the travel item in the database
                travel_item = db.get_travel_item_by_name(travel_name)
                icon = ":material/restaurant:"
                if travel_item['type'] == 'attraction':
                    icon = ":material/attractions:"
                elif travel_item['type'] == 'accommodation':
                    icon = ":material/hotel:"

                if travel_item is not None:
                    # Create dialog for valid travel item
                    @st.dialog(f"{icon} {travel_name}", width="large")
                    def _show_travel_dialog(travel_item: Dict[str, Any], travel_name: str) -> None:
                        info = travel_item['info']
                        if travel_item['type'] == 'restaurant':
                            _render_restaurant_details(info)
                        elif travel_item['type'] == 'attraction':
                            _render_attraction_details(info)
                        elif travel_item['type'] == 'accommodation':
                            _render_accommodation_details(info)
                    
                    st.button(
                        f"{icon} {travel_name}",
                        on_click=_show_travel_dialog,
                        args=(travel_item, travel_name),
                        key=f"travel_{travel_name}_{uuid.uuid4().hex[:8]}",
                        use_container_width=True,
                    )
                else:
                    # Travel item not found in database - show disabled button
                    st.button(
                        f":material/error: {travel_name} (not found)",
                        disabled=True,
                        key=f"travel_invalid_{travel_name}_{uuid.uuid4().hex[:8]}",
                        use_container_width=True,
                    )
            except Exception:
                # Error finding travel item - show disabled button
                st.button(
                    f":material/error: {travel_name} (error)",
                    disabled=True,
                    key=f"travel_error_{travel_name}_{uuid.uuid4().hex[:8]}",
                    use_container_width=True,
                )


def _render_restaurant_details(restaurant_info: Dict[str, Any]) -> None:
    """Render restaurant details in a dialog."""
    if restaurant_info.get('name'):
        st.markdown(f"**Name:** {restaurant_info['name']}")
    if restaurant_info.get('city'):
        st.markdown(f"**City:** {restaurant_info['city']}")
    if restaurant_info.get('state'):
        st.markdown(f"**State:** {restaurant_info['state']}")
    if restaurant_info.get('rating'):
        st.markdown(f"**Rating:** {restaurant_info['rating']}/5")
    if restaurant_info.get('price_range'):
        st.markdown(f"**Price Range:** {restaurant_info['price_range']}")
    if restaurant_info.get('cuisine'):
        st.markdown(f"**Cuisine:** {restaurant_info['cuisine']}")
    if restaurant_info.get('address'):
        st.markdown(f"**Address:** {restaurant_info['address']}")
    if restaurant_info.get('phone'):
        st.markdown(f"**Phone:** {restaurant_info['phone']}")


def _render_attraction_details(attraction_info: Dict[str, Any]) -> None:
    """Render attraction details in a dialog."""
    if attraction_info.get('name'):
        st.markdown(f"**Name:** {attraction_info['name']}")
    if attraction_info.get('city'):
        st.markdown(f"**City:** {attraction_info['city']}")
    if attraction_info.get('state'):
        st.markdown(f"**State:** {attraction_info['state']}")
    if attraction_info.get('rating'):
        st.markdown(f"**Rating:** {attraction_info['rating']}/5")
    if attraction_info.get('type'):
        st.markdown(f"**Type:** {attraction_info['type']}")
    if attraction_info.get('address'):
        st.markdown(f"**Address:** {attraction_info['address']}")
    if attraction_info.get('description'):
        st.markdown("---")
        st.markdown(attraction_info['description'])


def _render_accommodation_details(accommodation_info: Dict[str, Any]) -> None:
    """Render accommodation details in a dialog."""
    if accommodation_info.get('name'):
        st.markdown(f"**Name:** {accommodation_info['name']}")
    if accommodation_info.get('city'):
        st.markdown(f"**City:** {accommodation_info['city']}")
    if accommodation_info.get('state'):
        st.markdown(f"**State:** {accommodation_info['state']}")
    if accommodation_info.get('rating'):
        st.markdown(f"**Rating:** {accommodation_info['rating']}/5")
    if accommodation_info.get('price_range'):
        st.markdown(f"**Price Range:** {accommodation_info['price_range']}")
    if accommodation_info.get('address'):
        st.markdown(f"**Address:** {accommodation_info['address']}")
    if accommodation_info.get('phone'):
        st.markdown(f"**Phone:** {accommodation_info['phone']}")
    if accommodation_info.get('amenities'):
        st.markdown(f"**Amenities:** {accommodation_info['amenities']}")
