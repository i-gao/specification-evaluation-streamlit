from typing import List, Dict, Any, Optional, Callable
import streamlit as st
from evaluation.qualitative_eval import COMPARISON_LIKERT, INSTRUMENT_LIKERT
import os
import sys
import uuid
from data.travel_planner.db import TravelDB
from utils.misc import parse_json
from data.travel_planner.parser import parse_travel_plan
import random
from utils.streamlit_types import FormElement, form_element_to_streamlit
from data.reward import pairwise_win_rate

from data.travel_planner.reward_utils.tp_utils.func import (
    extract_from_to,
    extract_before_parenthesis,
    get_valid_name_city,
)


def render_eval(
    *,
    final_prediction: str,
    y0: Optional[str],
    db,
    people_number: int,
    dest: str,
):
    """
    Render evaluation UI for shopping custom specs and return (completed, feedback).
    """
    ranking_done = render_eval_first_page(
        final_prediction=final_prediction,
        y0=y0,
        db=db,
        people_number=people_number,
        dest=dest,
    )
    print(ranking_done)
    if not ranking_done:
        return False, None

    return True, None


def render_eval_first_page(
    *,
    final_prediction: str,
    y0: Optional[str],
    db,
    people_number: int,
    dest: str,
):
    """
    For each kind of thing, display the carousel with some decoys, and ask user to rank options among valid options in the actual dest
    """
    parsed_prediction = parse_travel_plan(
        final_prediction, include_info=True, db=db, people_number=people_number
    )
    parsed_y0 = parse_travel_plan(
        y0, include_info=True, db=db, people_number=people_number
    )

    # 1. Transportation (to)
    done = "transportation_to" in st.session_state.form_results["final_evaluation"]
    if not done:
        if len(parsed_prediction) == 0:
            predicted_transportation_to = []
        else:
            predicted_transportation_to = [parsed_prediction[0]["transportation"]]
        y0_transportation_to = [parsed_y0[0]["transportation"]]
        _render_carousel(
            predicted_transportation_to,
            y0_transportation_to,
            "transportation_to",
            _transportation_to_dialog_content,
            db,
            filter_fn=lambda d: dest == d["city2"],
        )
        return

    # 2. Transportation (from)
    done = "transportation_from" in st.session_state.form_results["final_evaluation"]
    if not done:
        if len(parsed_prediction) == 0:
            predicted_transportation_from = []
        else:
            predicted_transportation_from = [parsed_prediction[-1]["transportation"]]
        y0_transportation_from = [parsed_y0[-1]["transportation"]]
        _render_carousel(
            predicted_transportation_from,
            y0_transportation_from,
            "transportation_from",
            _transportation_to_dialog_content,
            db,
            filter_fn=lambda d: dest == d["city1"],
        )
        return

    # 3. Restaurants
    done = "restaurant" in st.session_state.form_results["final_evaluation"]
    if not done:
        predicted_restaurants = (
            [day["breakfast"] for day in parsed_prediction]
            + [day["lunch"] for day in parsed_prediction]
            + [day["dinner"] for day in parsed_prediction]
        )
        y0_restaurants = (
            [day["breakfast"] for day in parsed_y0]
            + [day["lunch"] for day in parsed_y0]
            + [day["dinner"] for day in parsed_y0]
        )
        _render_carousel(
            predicted_restaurants,
            y0_restaurants,
            "restaurant",
            _restaurant_to_dialog_content,
            db,
            filter_fn=lambda d: dest == d["city"],
        )
        return

    # 4. Attractions
    done = "attraction" in st.session_state.form_results["final_evaluation"]
    if not done:
        predicted_attractions = [
            a for day in parsed_prediction for a in day["attraction"]
        ]
        y0_attractions = [a for day in parsed_y0 for a in day["attraction"]]
        _render_carousel(
            predicted_attractions,
            y0_attractions,
            "attraction",
            _attraction_to_dialog_content,
            db,
            filter_fn=lambda d: dest == d["city"],
        )
        return

    # 5. Accommodations
    done = "accommodation" in st.session_state.form_results["final_evaluation"]
    if not done:
        predicted_accommodations = [day["accommodation"] for day in parsed_prediction]
        y0_accommodations = [day["accommodation"] for day in parsed_y0]
        _render_carousel(
            predicted_accommodations,
            y0_accommodations,
            "accommodation",
            _accommodation_to_dialog_content,
            db,
            filter_fn=lambda d: dest == d["city"],
        )
        return

    # Compute the final score
    prediction_rankings = [
        list(
            st.session_state.form_results["final_evaluation"][name][
                "predicted_ranks"
            ].values()
        )
        for name in [
            "transportation_to",
            "transportation_from",
            "restaurant",
            "attraction",
            "accommodation",
        ]
    ]
    y0_rankings = [
        list(
            st.session_state.form_results["final_evaluation"][name]["y0_ranks"].values()
        )
        for name in [
            "transportation_to",
            "transportation_from",
            "restaurant",
            "attraction",
            "accommodation",
        ]
    ]
    st.session_state.form_results["final_evaluation"]["score"] = pairwise_win_rate(
        prediction_rankings, y0_rankings
    )

    return True


def render_eval_second_page(
    *, final_prediction: str, y0: Optional[str], db, people_number: int
):
    st.write("### Evaluate a specific plan")
    render_travel_plan_streamlit(final_prediction, db, people_number)

    st.divider()

    form_elements = [
        FormElement(
            input_type="stars",
            label="Rate the overall quality of the plan.",
        ),
        FormElement(
            input_type="radio",
            label='How much do you agree with this statement? "I would rather follow this plan as is than continue my search with the assistant for 10 more minutes."',
            options=["-"] + INSTRUMENT_LIKERT,
        ),
        FormElement(
            input_type="text_area",
            label="If you were to continue your search with the assistant for 10 more minutes, what would you want it to change about the plan?",
            height=120,
        ),
    ]

    with st.form(key="travel_planner_custom_eval_form"):
        feedback: dict = {}
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
                    if (
                        not feedback.get(label)
                        or feedback.get(label) == ""
                        or feedback.get(label) == "-"
                    ):
                        st.error("Please fill in all required fields.")
                        return False, None
            return True, feedback

    return False, None


@st.fragment
def _render_carousel(
    predicted: List[Dict[str, Any]],
    y0: List[Dict[str, Any]],
    name: str,
    md_fn,
    db,
    filter_fn: Callable[[Dict[str, Any]], bool] = None,
    show_k: int = None,
):
    """
    Args:
        predicted: list of predicted options
        y0: list of y0 options
        name: name of the thing being ranked. Used for saving to session state.
        md_fn: function to render the option
        db: database
        filter_fn: function to filter the options
        show_k: number of options to show

    Adds to session state:
        - ranking: a dict mapping a rank (0-indexed) to a name
        - y0_ranks: a dict mapping name to a rank
        - predicted_ranks: a list of ranks for the predicted options
    """
    predicted = [
        p for p in predicted if p is not None and (filter_fn is None or filter_fn(p))
    ]
    predicted = list({d["name"]: d for d in predicted}.values())

    y0 = [p for p in y0 if p is not None and (filter_fn is None or filter_fn(p))]
    y0 = list({d["name"]: d for d in y0}.values())

    if len(predicted) == 0:
        # set difference is the entire y0, and y0 auto-wins
        dummy_rank = {i: y["name"] for i, y in enumerate(y0)}
        st.session_state.form_results["final_evaluation"][name] = {
            "ranking": dummy_rank,
            "y0_ranks": {v: k for k, v in dummy_rank.items()},
            "predicted_ranks": {},
        }
        st.rerun()

    # find the set difference
    predicted_names = set([p["name"] for p in predicted])
    y0_names = set([p["name"] for p in y0])
    diff_names = (predicted_names - y0_names).union(y0_names - predicted_names)
    if not diff_names:
        # don't render anything
        return

    predicted_options = [p for p in predicted if p["name"] in diff_names]
    y0_options = [p for p in y0 if p["name"] in diff_names]
    if show_k is not None and len(diff_names) > show_k:
        # try to get a roughly balanced set of options
        if len(predicted_options) < show_k / 2:
            options = predicted_options + y0_options[: show_k - len(predicted_options)]
        elif len(y0_options) < show_k / 2:
            options = predicted_options[: show_k - len(y0_options)] + y0_options
        else:
            options = (
                predicted_options[: show_k // 2 + show_k % 2]
                + y0_options[: show_k // 2]
            )
    else:
        options = predicted_options + y0_options

    random.shuffle(options)

    # display the carousel
    from evaluation.app.components import carousel

    def display_fn(i):
        st.markdown(
            md_fn(
                options[i],
            ),
            unsafe_allow_html=True,
        )

    st.markdown("### Review the assistant's recommendations")
    st.markdown(f"The assistant has recommended {len(options)} {name}s for you.")
    carousel([lambda i=i: display_fn(i) for i in range(len(options))], height=300)
    with st.form(key=f"ranking_form_{name}"):
        rank = st.multiselect(
            f"Rank the {name} above from MOST to LEAST preferred.",
            [i for i in range(len(options))],
            default=[],
            format_func=lambda x: f"{name.upper()} {x + 1}: {options[x]['name']}",
        )
        submit = st.form_submit_button("Submit", type="primary")
        if submit:
            if len(rank) != len(options):
                st.error("Please rank all options")
                return
            ranking = {i: options[i]["name"] for i in rank}
            st.session_state.form_results["final_evaluation"][name] = {
                "ranking": ranking,
                "y0_ranks": {v: k for k, v in ranking.items() if v in y0_names},
                "predicted_ranks": {
                    v: k for k, v in ranking.items() if v in predicted_names
                },
            }
            st.rerun()


def render_comparison(
    y1: str,
    y2: str,
    db: TravelDB,
    people_number: int,
    valid1: bool,
    valid2: bool,
    metadata1: Dict[str, Any],
    metadata2: Dict[str, Any],
) -> None:
    # Render in two tabs
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
        render_travel_plan_streamlit(y1, db, people_number)

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
        render_travel_plan_streamlit(y2, db, people_number)


def render_travel_plan_streamlit(
    travel_plan: str, travel_db: TravelDB, people_number: int
) -> None:
    travel_plan = parse_travel_plan(
        travel_plan, include_info=True, db=travel_db, people_number=people_number
    )

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
        transportation = day["transportation"]
        if transportation and transportation["time"] == "morning":
            items.append(f"{transportation['icon']} {transportation['name']}")
        breakfast = day["breakfast"]
        if breakfast:
            if breakfast.get("invalid", False):
                items.append(
                    f":red-background[:material/error: {breakfast['name']} ({breakfast['city']})]"
                )
            else:
                items.append(
                    f":material/restaurant: {breakfast['name']} ({breakfast['city']})"
                )
        for att in day["attraction"]:
            if att["time"] == "morning":
                if att.get("invalid", False):
                    items.append(
                        f":red-background[:material/error: {att['name']} ({att['city']})]"
                    )
                else:
                    items.append(f"{att['icon']} {att['name']} ({att['city']})")
        morning_row.append("<br>".join(items) if items else "*No morning activities*")
    lines.append("| " + " | ".join(morning_row) + " |")

    afternoon_row = ["**Afternoon**"]
    for day in travel_plan:
        items: List[str] = []
        transportation = day["transportation"]
        if transportation and transportation["time"] == "afternoon":
            items.append(f"{transportation['icon']} {transportation['name']}")
        lunch = day["lunch"]
        if lunch:
            if lunch.get("invalid", False):
                items.append(
                    f":red-background[:material/error: {lunch['name']} ({lunch['city']})]"
                )
            else:
                items.append(f":material/restaurant: {lunch['name']} ({lunch['city']})")
        for att in day["attraction"]:
            if att["time"] == "afternoon":
                if att.get("invalid", False):
                    items.append(
                        f":red-background[:material/error: {att['name']} ({att['city']})]"
                    )
                else:
                    items.append(f"{att['icon']} {att['name']} ({att['city']})")
        afternoon_row.append(
            "<br>".join(items) if items else "*No afternoon activities*"
        )
    lines.append("| " + " | ".join(afternoon_row) + " |")

    evening_row = ["**Evening**"]
    for day in travel_plan:
        items: List[str] = []
        transportation = day["transportation"]
        if transportation and transportation["time"] == "evening":
            items.append(f"{transportation['icon']} {transportation['name']}")
        dinner = day["dinner"]
        if dinner:
            if dinner.get("invalid", False):
                items.append(
                    f":red-background[:material/error: {dinner['name']} ({dinner['city']})]"
                )
            else:
                items.append(
                    f":material/restaurant: {dinner['name']} ({dinner['city']})"
                )
        for att in day["attraction"]:
            if att["time"] == "evening":
                if att.get("invalid", False):
                    items.append(
                        f":red-background[:material/error: {att['name']} ({att['city']})]"
                    )
                else:
                    items.append(f"{att['icon']} {att['name']} ({att['city']})")
        evening_row.append("<br>".join(items) if items else "*No evening activities*")
    lines.append("| " + " | ".join(evening_row) + " |")

    accommodation_row = ["**Accommodation**"]
    for day in travel_plan:
        accommodation = day["accommodation"]
        if accommodation:
            if accommodation.get("invalid", False):
                accommodation_row.append(
                    f":red-background[:material/error: {accommodation['name']} ({accommodation['city']})]"
                )
            else:
                accommodation_row.append(
                    f":material/hotel: {accommodation['name']} ({accommodation['city']})"
                )
        else:
            accommodation_row.append("*No accommodation*")
    lines.append("| " + " | ".join(accommodation_row) + " |")

    daily_costs_row = ["**Estimated Spend**"]
    total_cost = 0
    for day in travel_plan:
        day_cost = day["total_cost"]
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
        day_num = day_data.get("days", "Unknown")

        # Collect all items for this day
        all_items = []

        # Add meals
        for meal_type in ["breakfast", "lunch", "dinner"]:
            meal = day_data[meal_type]
            if not meal:
                continue
            if not meal["invalid"]:
                all_items.append(
                    {
                        "type": "restaurant",
                        "title": f":material/restaurant: {meal_type.capitalize()}: {meal['name']} ({_get_dollar_signs(meal['average_cost'])})",
                        "data": meal,
                        "meal_type": meal_type,
                    }
                )
            else:
                all_items.append(
                    {
                        "type": "restaurant",
                        "title": f":material/error: {meal_type.capitalize()}: Invalid restaurant",
                        "data": meal,
                        "meal_type": meal_type,
                    }
                )

        # Add attractions
        attractions = day_data["attraction"]
        for attraction in attractions:
            if not attraction["invalid"]:
                all_items.append(
                    {
                        "type": "attraction",
                        "title": f"{attraction['icon']} {attraction['name']}",
                        "data": attraction,
                    }
                )
            else:
                all_items.append(
                    {
                        "type": "attraction",
                        "title": f":material/error: {attraction['name']}",
                        "data": attraction,
                    }
                )

        # Add accommodation
        accommodation = day_data["accommodation"]
        if accommodation:
            if not accommodation["invalid"]:
                all_items.append(
                    {
                        "type": "accommodation",
                        "title": f":material/hotel: Accommodation: {accommodation['name']}",
                        "data": accommodation,
                    }
                )
            else:
                all_items.append(
                    {
                        "type": "accommodation",
                        "title": ":material/error: Accommodation: Invalid accommodation",
                        "data": accommodation,
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

    st.markdown("---")


def render_travel_mentions(travel_names: List[str], db: TravelDB) -> None:
    """
    Render a section showing mentioned travel items (restaurants, attractions, accommodations) with their details.
    Uses a grid of buttons that launch dialogs, similar to workout planning exercises and meal planning recipes.
    """
    if not travel_names:
        return

    st.markdown(
        "Click on an item to view its details, including location, ratings, prices, and other information."
    )

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
                if travel_item["type"] == "attraction":
                    icon = ":material/attractions:"
                elif travel_item["type"] == "accommodation":
                    icon = ":material/hotel:"

                if travel_item is not None:
                    # Create dialog for valid travel item
                    @st.dialog(f"{icon} {travel_name}", width="large")
                    def _show_travel_dialog(
                        travel_item: Dict[str, Any], travel_name: str
                    ) -> None:
                        if travel_item["type"] == "restaurant":
                            _render_restaurant_details(travel_item)
                        elif travel_item["type"] == "attraction":
                            _render_attraction_details(travel_item)
                        elif travel_item["type"] == "accommodation":
                            _render_accommodation_details(travel_item)

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


# ===== Helpers (ported from data.py to avoid circular import) =====


def _is_round_trip(travel_plan: List[Dict[str, Any]]) -> bool:
    if not travel_plan:
        return False, "No travel plan data available"

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


def _restaurant_to_dialog_content(
    restaurant: Dict[str, Any], meal_type: str = None
) -> str:
    """Convert restaurant data to dialog content (similar to collapsible but without details tags)."""
    if meal_type is None:
        meal_type = ""
    else:
        meal_type = meal_type.capitalize() + ": "
    if restaurant.get("invalid"):
        return f"<b>:material/restaurant: {meal_type}Invalid restaurant</b><br><br>:material/error: '{restaurant['name']}' is an invalid restaurant that is not in the database."

    lines: List[str] = []
    lines.append(
        f"<b>:material/restaurant: {meal_type}{restaurant['name']} ({_get_dollar_signs(restaurant['average_cost'])})</b>"
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
    # if attributes:
    #     lines.append("\n".join(attributes))
    #     lines.append("")

    return "\n".join(lines)


def _attraction_to_dialog_content(attraction: Dict[str, Any]) -> str:
    """Convert attraction data to dialog content (similar to collapsible but without details tags)."""
    if attraction.get("invalid"):
        return f"<b>:material/attractions: Invalid attraction</b><br><br>:material/error: '{attraction['name']}' is an invalid attraction that is not in the database."

    lines: List[str] = []
    basic_info: List[str] = []
    if attraction.get("attraction_type"):
        types_val = attraction["attraction_type"]
        types_str = (
            ", ".join([_fmt(t).capitalize() for t in types_val])
            if isinstance(types_val, list)
            else _fmt(types_val)
        )
        basic_info.append(types_str)
    if attraction.get("activity_level"):
        basic_info.append(f"**Activity Level:** {attraction['activity_level']}")

    lines.append(f"<b>{attraction['icon']} {attraction['name']}</b>")
    lines.append("")

    if basic_info:
        lines.append(" | ".join(basic_info))
        lines.append("")
    if attraction.get("address"):
        lines.append("**Address:**")
        lines.append(f"{attraction['address']}")
        lines.append("")
    if attraction.get("description"):
        lines.append(f"{attraction['description']}")
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


def _transportation_to_dialog_content(transportation: Dict[str, Any]) -> str:
    """Convert transportation data to dialog content (similar to collapsible but without details tags)."""
    if transportation.get("invalid"):
        return f"<b>:material/error: Invalid transportation</b><br><br>:material/error: '{transportation['name']}' is an invalid transportation that is not in the database."

    lines: List[str] = []
    lines.append(f"<b>{transportation['icon']} {transportation['name']}</b>")
    lines.append("")
    if transportation.get("time"):
        lines.append(f"**Time:** {transportation['time']}")
        lines.append("")
    if transportation.get("city1"):
        lines.append(f"**From:** {transportation['city1']}")
        lines.append("")
    if transportation.get("city2"):
        lines.append(f"**To:** {transportation['city2']}")
        lines.append("")
    if transportation.get("type"):
        lines.append(f"**Type:** {transportation['type']}")
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


def _render_restaurant_details(restaurant_info: Dict[str, Any]) -> None:
    """Render restaurant details in a dialog."""
    if restaurant_info.get("name"):
        st.markdown(f"**Name:** {restaurant_info['name']}")
    if restaurant_info.get("city"):
        st.markdown(f"**City:** {restaurant_info['city']}")
    if restaurant_info.get("state"):
        st.markdown(f"**State:** {restaurant_info['state']}")
    if restaurant_info.get("rating"):
        st.markdown(f"**Rating:** {restaurant_info['rating']}/5")
    if restaurant_info.get("price_range"):
        st.markdown(f"**Price Range:** {restaurant_info['price_range']}")
    if restaurant_info.get("cuisine"):
        st.markdown(f"**Cuisine:** {restaurant_info['cuisine']}")
    if restaurant_info.get("address"):
        st.markdown(f"**Address:** {restaurant_info['address']}")
    if restaurant_info.get("phone"):
        st.markdown(f"**Phone:** {restaurant_info['phone']}")


def _render_attraction_details(attraction_info: Dict[str, Any]) -> None:
    """Render attraction details in a dialog."""
    if attraction_info.get("name"):
        st.markdown(f"**Name:** {attraction_info['name']}")
    if attraction_info.get("city"):
        st.markdown(f"**City:** {attraction_info['city']}")
    if attraction_info.get("state"):
        st.markdown(f"**State:** {attraction_info['state']}")
    if attraction_info.get("rating"):
        st.markdown(f"**Rating:** {attraction_info['rating']}/5")
    if attraction_info.get("type"):
        st.markdown(f"**Type:** {attraction_info['type']}")
    if attraction_info.get("address"):
        st.markdown(f"**Address:** {attraction_info['address']}")
    if attraction_info.get("description"):
        st.markdown("---")
        st.markdown(attraction_info["description"])


def _render_accommodation_details(accommodation_info: Dict[str, Any]) -> None:
    """Render accommodation details in a dialog."""
    if accommodation_info.get("name"):
        st.markdown(f"**Name:** {accommodation_info['name']}")
    if accommodation_info.get("city"):
        st.markdown(f"**City:** {accommodation_info['city']}")
    if accommodation_info.get("state"):
        st.markdown(f"**State:** {accommodation_info['state']}")
    if accommodation_info.get("rating"):
        st.markdown(f"**Rating:** {accommodation_info['rating']}/5")
    if accommodation_info.get("price_range"):
        st.markdown(f"**Price Range:** {accommodation_info['price_range']}")
    if accommodation_info.get("address"):
        st.markdown(f"**Address:** {accommodation_info['address']}")
    if accommodation_info.get("phone"):
        st.markdown(f"**Phone:** {accommodation_info['phone']}")
    if accommodation_info.get("amenities"):
        st.markdown(f"**Amenities:** {accommodation_info['amenities']}")
