from datasets import load_from_disk
from typing import List, Tuple, Dict, Any, Optional
import os
import sys
import json
from langchain_core.tools import tool
import random

from data.dataset import (
    SpecificationCollection,
    FixedSpecification,
    CustomSpecification,
)
from data.travel_planner.db import TravelDB
from data.actions import Action, get_jupyter_actions

from utils.misc import get_recursive, parse_json, add_section, replace_tags_with_link
from utils.streamlit_types import FormElement, form_element_to_streamlit
import streamlit as st
import data.travel_planner.streamlit_render as renderer

# Add TravelPlanner to path
travel_planner_path = os.path.join(os.path.dirname(__file__), "reward_utils")
if travel_planner_path not in sys.path:
    sys.path.append(travel_planner_path)

from evaluation.commonsense_constraint import evaluation as commonsense_eval
from evaluation.hard_constraint import evaluation as hard_eval
from evaluation.preferences import evaluation as preferences_eval, compute_linear_reward


FORMATTING_INSTRUCTIONS = """All items in the itinerary must exactly match names from the database. Format the response as a JSON array of dictionaries, where each dictionary represents one day's itinerary. Each dictionary must include the following fields:

Required Fields:
   - days: Day number (integer)
   - current_city: Current city or travel route
       * For single-city days: Just the city name (e.g., "Peoria")
       * For travel days: Use format "from A to B" (e.g., "from Dallas to Peoria")
       * Must match destination in transportation field when traveling
   - transportation: Travel details or "-" if no travel
       * For flights: Include "Flight Number: [number], from [city] to [city], Departure Time: [time], Arrival Time: [time]"
       * For other transport: Describe route and mode (e.g., "Self-driving, from Miami(Florida) to Punta Gorda(Florida), duration: 2 hours 42 mins, distance: 292 km, cost: 14")
   - breakfast: Restaurant name and city, or "-" if skipping
      * Format: "Restaurant Name, City" (e.g., "Tandoor Ka Zaika, Peoria")
   - lunch: "Restaurant Name, City", or "-" if skipping
   - dinner: "Restaurant Name, City", or "-" if skipping
   - attraction: List of attractions, or "-" if skipping
      * Format each attraction as "Name, City" (e.g., "Peoria Historical Society, Peoria")
      * Separate multiple attractions with semicolons (e.g., "Peoria Historical Society, Peoria;Glen Oak Park, Peoria;")
   - accommodation: "Hotel/lodging name, City", or "-" if skipping
      * Format: "Hotel/lodging name, City" (e.g., "Bushwick Music Mansion, Peoria")

Remove any "$" symbols from costs.

To render a description of a single restaurant, attraction, or accommodation (instead of a full travel plan) to the user, you can mention its name and wrap it in <travel></travel>, e.g.: '<travel>Restaurant Name, City</travel>' or '<travel>Attraction Name, City</travel>' or '<travel>Hotel Name, City</travel>'. Do not put <travel></travel> tags inside the JSON of a full travel plan.
"""

PREFERENCE_KEYS_TO_TEXT = {
    "liked_tags": "Liked restaurant tags",
    "disliked_tags": "Disliked restaurant tags",
    "liked_attraction_types": "Liked attraction types",
    "disliked_attraction_types": "Disliked attraction types",
    "preferred_activity_level": "Preferred activity level for attractions",
    "liked_room_types": "Liked room types",
    "disliked_room_types": "Disliked room types",
    "min_num_ratings_accommodations": "Minimum number of ratings for accommodations",
    "min_rating_restaurants": "Minimum rating for restaurants",
    "min_num_ratings_restaurants": "Minimum number of ratings for restaurants",
    "specific_liked_attractions": "Specific liked attractions",
    "specific_disliked_attractions": "Specific disliked attractions",
    "specific_liked_restaurants": "Specific liked restaurants",
    "specific_disliked_restaurants": "Specific disliked restaurants",
    "specific_liked_accommodations": "Specific liked accommodations",
    "specific_disliked_accommodations": "Specific disliked accommodations",
    "restaurant_repeats": "Ideally, no restaurant should be repeated across meals",
    "attraction_repeats": "Ideally, no attraction should be repeated across days",
    "min_attractions_per_single_city_day": "Minimum number of attractions per single-city day",
    "max_attractions_per_single_city_day": "Maximum number of attractions per single-city day",
    "min_attractions_per_travel_day": "Minimum number of attractions per travel day",
    "max_attractions_per_travel_day": "Maximum number of attractions per travel day",
    "restaurant_attributes": "Desired restaurant attributes",
}

CONSTRAINT_KEYS_TO_TEXT = {
    "days": "Travel duration: {v} days.",
    "date": "Travel dates: {v}.",
    "people_number": "Party size: {v} people.",
    "budget": "Hard budget: ${v}.",
    "local_constraint.house rule": "House rules: {v}.",
    "local_constraint.room_type": "Must stay at this type of accomodation: {v}.",
    "local_constraint.transportation": "Must factor in these transportation preferences: {v}.",
    "local_constraint.min_attractions_per_travel_day": "Must visit at least {v} attractions per travel day.",
    "local_constraint.max_attractions_per_travel_day": "Must visit at most {v} attractions per travel day.",
    "local_constraint.min_attractions_per_single_city_day": "Must visit at least {v} attractions per single-city day.",
    "local_constraint.max_attractions_per_single_city_day": "Must visit at most {v} attractions per single-city day.",
}

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))

FILE_DESCRIPTIONS = [
    {
        "filename": "attractions.csv",
        "description": "Table of attractions across cities. Filter by the 'city' column to get attractions for specific cities.",
        "columns": {
            "city": "City of the attraction",
            "name": "Name of the attraction",
            "address": "Address of the attraction",
            "attraction_type": "Type of attraction, one of ['outdoor_park', 'history', 'art', 'amusement_park_or_zoo', 'museum', 'food_and_shopping']",
            "activity_level": "Level of activity required for the attraction, one of ['low', 'medium', 'high']",
            "description": "Description of the attraction from the attraction website",
        },
    },
    {
        "filename": "accommodations.csv",
        "description": "Table of accommodations across cities. Filter by the 'city' column to get accommodations for specific cities.",
        "columns": {
            "city": "City of the accommodation",
            "name": "Name of the accommodation",
            "room_type": "Type of room, one of ['Shared room', 'Entire home/apt', 'Private room']",
            "price": "Price of the accommodation",
            "minimum_nights": "If this accommodation is booked, it must be booked for at least this number of nights",
            "num_reviews": "Number of reviews of the accommodation",
            "house_rules": "House rules of the accommodation",
            "maximum_occupancy": "Maximum number of people that can stay in the accommodation",
        },
    },
    {
        "filename": "restaurants.csv",
        "description": "Table of restaurants across cities. Filter by the 'city' column to get restaurants for specific cities.",
        "columns": {
            "city": "City of the restaurant",
            "name": "Name of the restaurant",
            "tags": "Yelp tags for the restaurant",
            "average_cost": "Average cost of the restaurant",
            "rating": "Rating of the restaurant",
            "review_count": "Number of reviews of the restaurant",
            "attributes": "Attributes of the restaurant",
            "has_takeout": "Whether the restaurant has takeout",
            "has_delivery": "Whether the restaurant has delivery",
            "has_reservations": "Whether the restaurant has reservations",
            "noise_level": "Noise level of the restaurant",
            "has_outdoor_seating": "Whether the restaurant has outdoor seating",
            "has_wifi": "Whether the restaurant has wifi",
            "attire": "Attire of the restaurant",
            "parking": "Parking of the restaurant",
            "good_for_groups": "Whether the restaurant is good for groups",
            "good_for_kids": "Whether the restaurant is good for kids",
            "accepts_credit_card": "Whether the restaurant accepts credit card",
            "wheelchair_accessible": "Whether the restaurant is wheelchair accessible",
            "has_table_service": "Whether the restaurant has table service",
            "has_bike_parking": "Whether the restaurant has bike parking",
        },
    },
    {
        "filename": "flights.csv",
        "description": "Table of flights. Filter by 'departure_city' or 'arrival_city' columns to get flights for specific cities.",
        "columns": {
            "flight_number": "Flight number",
            "price": "Price of the flight",
            "departure_time": "Departure time of the flight",
            "arrival_time": "Arrival time of the flight",
            "actual_elapsed_time": "Actual elapsed time of the flight",
            "flight_date": "Date of the flight",
            "departure_city": "Departure city of the flight",
            "arrival_city": "Arrival city of the flight",
            "distance": "Distance of the flight",
        },
    },
    {
        "filename": "city_state.csv",
        "description": "Table of cities and states.",
        "columns": {
            "city": "City name",
            "state": "State of the city",
        },
    },
]

FIXED_INSTRUCTIONS = """
### What you need to prompt the assistant to do
In this task, **your goal is to get the assistant to write the perfect travel itinerary for a client.** A travel itinerary lays out the plan for each day of a trip, including where to eat for each meal, what attractions to visit, where to stay, and how to travel between cities.

The itinerary must satisfy the client's constraints and match their travel preferences. When the chat session starts, some information about the client will appear on the left side of the screen.

### The tricky part
**Some of the client's preferences may not be written out.** For example, they may not have specified what kinds of things they like to do.

To maximize your score, you will have to try different itineraries and ask the client to evaluate them. The client's score will be between 0 and 100. If the itinerary is illogical (e.g., is not a round trip) or violates a user constraint, then the score will be -infinity.
"""


COMMONSENSE_INSTRUCTIONS = """A valid travel plan:
* ONLY uses flights, driving, AirBnB, restaurants, and attractions from the assistant's database. Using other options is not allowed.
* Is a round trip, starting and ending at the same city.
* Specifies the plan for each day of the trip. A plan that says, "You decide" is not valid; all details must be ironed out.

There are two kinds of days in a travel itinerary:
1. **Travel days**: these are days that involve traveling between cities. Transportation must be specified for these days.
2. **Single-city days**: these are days that involve staying in a single city. All meals and attractions must be scheduled for these days.

Other notes:
* You can visit multiple attractions per day.
* You can only eat at one restaurant per meal.
* There must be an accommodation listed every day, except the last day, when you will be returning home.
* Accommodations have rules: you must follow those rules when booking.
"""


def render_fixed_task_explanation():
    """Render the fixed task explanation for travel planning."""
    st.markdown(FIXED_INSTRUCTIONS)
    st.markdown(COMMONSENSE_INSTRUCTIONS)


class TravelPlannerDataset(SpecificationCollection):
    """
    The TravelPlanner benchmark evaluates how well LMs can generate
    travel itineraries which obey some constraints.

    Paper: https://arxiv.org/abs/2402.01622
    Dataset: https://huggingface.co/datasets/osunlp/TravelPlanner

    Each travel planning case study is treated as a separate "task" to solve.
    - The constraints of the task represent the specification
    - The dataset consists of a single (x), representing the set of available
        itinerary options in the `reference_information` column, as well as the
        number of travel days requested by the customer.

    Dev set split:
    - original train set = dev set
    - original validation set = test set
    """

    @property
    def dataset_name(self) -> str:
        return "travel_planner"

    def assets_file_id(self) -> str:
        return None

    @property
    def dataset_pretty_name(self) -> str:
        return "Travel Planning"

    @property
    def dataset_description(self) -> str:
        return "Work with the assistant to **plan a vacation for yourself.**"

    @property
    def default_docker_images(self) -> List[Dict[str, str]]:
        return [
            {
                "image_name": "jupyter_docker_image",
                "dockerfile_path": "utils/jupyter_docker_image/Dockerfile",
                "build_context": "utils/jupyter_docker_image",
                "description": "Docker image for Jupyter notebook",
            },
            {
                "image_name": "travel_planner",
                "dockerfile_path": "data/travel_planner/reward_utils/Dockerfile",
                "build_context": "data/travel_planner",
                "description": "Docker image for Travel Planner code evaluation",
            },
        ]

    def _create_user_expertise_form(self) -> List[FormElement]:
        """Create user expertise form for travel planning."""
        return [
            FormElement(
                input_type="radio",
                label="How familiar are you with planning vacations?",
                options=[
                    "I have never planned a vacation before",
                    "I have planned 1-2 vacations before",
                    "I have planned 3-5 vacations before",
                    "I have planned 6-10 vacations before",
                    "I have planned more than 10 vacations before",
                ],
                required=True,
            )
        ]

    def _create_user_evaluation_form(self) -> List[FormElement]:
        """Create the user evaluation form for travel planning."""
        return [
            FormElement(
                input_type="radio",
                label="Compare the **city choices** in travel plans A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare how complete the **transportation plans** are in travel plans A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare the **restaurant choices** in travel plans A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare the **attraction choices (type, activity level, age appropriateness)** in travel plans A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare **the pacing / busyness** of travel plans A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare **the accommodation choices** of travel plans A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare **the overall price** of travel plans A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Which travel plan are you more likely to adopt: A or B?",
                options=["A", "neutral", "B"],
                required=True,
            ),
        ]

    def __init__(
        self,
        dev: bool = False,
        docker_image: str = None,
        fixed_indexes: Optional[List[int]] = None,
        custom_indexes: Optional[List[int]] = None,
        persist_docker_container: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(dev=dev, **kwargs)

        # Load all the problems
        split = "train" if dev else "validation"
        self._rows = load_from_disk(f"{DATASET_ROOT}/assets/personalized_{split}")
        self._rows = self._rows.select(
            range(1, len(self._rows))
        )  # first row is used as a demo in render_task_explanation
        self.fixed_length = len(self._rows)
        # only pick "simple" prompts for custom: <= 3 days
        self._custom_rows = self._rows.filter(lambda x: x["days"] <= 3  )
        self.custom_length = len(
            self._custom_rows
        )  # Each fixed spec has a corresponding custom spec
        self._docker_image = docker_image
        self._travel_db = TravelDB()
        self._persist_docker_container = persist_docker_container

        # All subclasses must have these attributes set
        self._finish_init()

        if fixed_indexes is not None:
            self.load_fixed_specs(indexes=fixed_indexes)
        if custom_indexes is not None:
            self.load_custom_specs(indexes=custom_indexes)

    def _load_fixed_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, FixedSpecification]:
        if indexes is None:
            return {}

        # Load requested specs
        specs = {}
        for ix in indexes:
            task = self._rows[ix]

            task["local_constraint"] = json.loads(task["local_constraint"])
            task["driving_info"] = json.loads(task["driving_info"])
            task["annotated_plan"] = (
                [d for d in json.loads(task["annotated_plan"]) if len(d) > 0]
                if task["annotated_plan"] is not None
                else None
            )
            task["preferences"] = json.loads(task["preferences"])
            task["preference_weights"] = json.loads(task["preference_weights"])

            # Extract constraints
            constraints = []
            for key in CONSTRAINT_KEYS_TO_TEXT.keys():
                if key == "days":
                    continue
                v = get_recursive(task, key)
                if v is not None:
                    if isinstance(v, list):
                        v = ", ".join(v)
                    constraints.append(CONSTRAINT_KEYS_TO_TEXT[key].format(**{"v": v}))

            # Build a list of (key, value, weight) for sorting
            preferences_with_weights = []
            for k, v in task["preferences"].items():
                if v is None:
                    continue
                weight = task["preference_weights"].get(k, 0)
                preferences_with_weights.append((k, v, weight))
            # Sort by weight descending
            preferences_with_weights.sort(key=lambda x: x[2], reverse=True)

            preference_str = ""
            for k, v, _ in preferences_with_weights:
                if isinstance(v, (list, dict)) and len(v) == 0:
                    continue
                preference_str += f"- {PREFERENCE_KEYS_TO_TEXT[k]}: {v}.\n"

            # Create theta (explicit knowledge) - includes basic task description and hard constraints
            signature = (
                f"The task is to produce a day-by-day travel itinerary to {task['dest']} using only the options found in the database. A good itinerary satisfies all of our constraints, and satisfies as many preferences as possible.\n\nTo help you, the sandbox environment provides a Jupyter notebook and CSV files containing the available options. All items in the itinerary must correspond to entries from the given CSV files, or from the get_driving_options tool.\n\n"
                + add_section(
                    "Constraints (must be satisfied)",
                    f"- The trip should visit {task['visiting_city_number']} cities (i.e. {task['dest']}) and last for {task['days']} days.\n"
                    + f"- For transportation, the trip should start on the first day from our origin city of {task['org']} and end back at {task['org']} on the last day.\n"
                    + "\n- ".join(constraints),
                )
            )
            theta = add_section(
                "Preferences (most important to least important)",
                preference_str.replace("\n", "\n<chunk>\n"),
            )

            if self._persist_docker_container and self._docker_image is not None:
                from llm_sandbox import SandboxSession
                session = SandboxSession(image=self._docker_image)
                session.open()
                container_id = session.container.id
            else:
                container_id = None

            # Create Jupyter actions with all available CSV files
            filename, actions = get_jupyter_actions(
                docker_image=self._docker_image,
                docker_container_id=container_id,
                ls_output=FILE_DESCRIPTIONS,
                root_dir=os.path.join(DATASET_ROOT, "assets"),
            )

            ystar = task["annotated_plan"]
            spec = FixedSpecification(
                dataset_name=self.dataset_name,
                index=f"fixed_{ix}",
                full_specification=theta,
                initial_specification=signature,
                validity_fn=validity_fn,
                validity_kwargs={
                    "query_data": task,
                },
                validity_fn_tool_name="check_travel_plan_validity",
                validity_fn_tool_description="Check if the travel plan is valid and within budget",
                reward_fn=reward_fn,
                reward_kwargs={
                    "query_data": task,
                    "weights": task["preference_weights"],
                },
                reward_fn_tool_name="score_travel_plan",
                reward_fn_tool_description="Score the travel plan based on preferences",
                ystar=ystar,
                # metric_name=None,  # Not provided
                # baseline_scores=None,  # Not provided
                render_task_explanation=render_fixed_task_explanation,
                actions=actions + get_driving_actions(task["driving_info"]),
                fmt_instructions=FORMATTING_INSTRUCTIONS,
                render_msg_fn=output_to_streamlit,
                render_msg_kwargs=["db", "people_number"],
                name=f"travel_planner_{task['level']}_{task['org']}_{task['dest']}",
                state_files=[filename],
                files_to_clean=[filename],
                container_ids=[container_id],
                user_expertise_form=self._create_user_expertise_form(),
            )
            specs[ix] = spec
        return specs

    def _create_user_specification_form_final(self) -> List[FormElement]:
        """Create the user specification form for travel planning."""
        return [
            FormElement(
                input_type="number_input",
                label="How many people are you traveling with? (0 for solo travel)?",
                default=0,
                step=1,
                min_value=0,
                max_value=10,
                required=True,
            ),
            FormElement(
                input_type="toggle",
                label="Does your travel party include kids?",
                default=False,
                required=True,
            ),
        ]

    def _load_custom_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, CustomSpecification]:
        """Load custom specifications for travel planning."""
        if indexes is None:
            return {}

        specs = {}
        for ix in indexes:
            # Parse task data
            fixed_task = self._custom_rows[ix]
            task = {
                "org": fixed_task["org"],
                "dest": fixed_task["dest"],
                "days": fixed_task["days"],
                "visiting_city_number": fixed_task["visiting_city_number"],
                "date": fixed_task["date"],
                "people_number": fixed_task["people_number"],
                "cities": fixed_task["cities"],
                "local_constraint": {},  # drop local constraints from the fixed task
                "driving_info": json.loads(fixed_task["driving_info"]),
                "budget": fixed_task["budget"],
                # drop preferences and preference weights from the fixed task
            }

            # Create Docker container
            if self._persist_docker_container and self._docker_image is not None:
                session = SandboxSession(image=self._docker_image)
                session.open()
                container_id = session.container.id
            else:
                container_id = None

            # Create Jupyter actions
            filename, actions = get_jupyter_actions(
                docker_image=self._docker_image,
                docker_container_id=container_id,
                ls_output=FILE_DESCRIPTIONS,
                root_dir=os.path.join(DATASET_ROOT, "assets"),
            )

            # Create custom specification
            spec = CustomSpecification(
                initial_specification=f"Plan a trip from {task['org']} to {task['dest']} over {task['days']} days, with a budget of ${task['budget']}",
                user_specification_form_initial=[],
                user_specification_form_final=self._create_user_specification_form_final(),
                user_specification_callback=user_specification_callback,
                user_specification_callback_kwargs=[
                    "_validity_kwargs",
                    "initial_specification",
                ],
                validity_fn=validity_fn,
                validity_kwargs={
                    "query_data": task,
                },
                people_number=1,
                validity_fn_tool_name="check_travel_plan_validity",
                validity_fn_tool_description="Check if the travel plan is valid and within budget",
                y0=fixed_task["annotated_plan"],
                render_task_explanation=self._render_custom_task_explanation,
                actions=actions + get_driving_actions(task["driving_info"]),
                fmt_instructions=FORMATTING_INSTRUCTIONS,
                render_msg_fn=output_to_streamlit,
                render_comparison_fn=output_to_streamlit_comparison,
                render_msg_kwargs=["db", "people_number"],
                name=f"custom_travel_planner_{ix}",
                state_files=[filename],
                files_to_clean=[filename],
                container_ids=[container_id],
                user_expertise_form=self._create_user_expertise_form(),
                db=self._travel_db,
                render_evaluation_fn=lambda **kwargs: renderer.render_eval_travel(
                    **kwargs, db=self._travel_db, people_number=1
                ),
                dataset_name=self.dataset_name,
                index=f"custom_{ix}",
            )
            specs[ix] = spec
        return specs

    def _render_custom_task_explanation(self):
        """Render the custom task explanation for travel planning."""

        st.markdown("### What you need to prompt the assistant to do")
        st.markdown(
            "In this task, **your goal is to get the assistant to plan an ideal trip for yourself.** A travel itinerary lays out the plan for each day of a trip, including where to eat for each meal, what attractions to visit, where to stay, and how to travel between cities."
        )

        with st.container(border=True):
            plan = [
                {
                    "days": 1,
                    "current_city": "from St. Petersburg to Rockford",
                    "transportation": "Flight Number: F3573659, from St. Petersburg to Rockford, Departure Time: 15:40, Arrival Time: 17:04",
                    "breakfast": "-",
                    "attraction": "-",
                    "lunch": "-",
                    "dinner": "Bazille, Rockford",
                    "accommodation": "Pure luxury one bdrm + sofa bed on Central Park, Rockford",
                },
                {
                    "days": 2,
                    "current_city": "Rockford",
                    "transportation": "-",
                    "breakfast": "McAlister's Deli, Rockford",
                    "attraction": "Burpee Museum of Natural History, Rockford;Midway Village Museum, Rockford;Discovery Center Museum, Rockford;",
                    "lunch": "Poke Express, Rockford",
                    "dinner": "Al-Sham Palace, Rockford",
                    "accommodation": "Pure luxury one bdrm + sofa bed on Central Park, Rockford",
                },
                {
                    "days": 3,
                    "current_city": "from Rockford to St. Petersburg",
                    "transportation": "Flight Number: F3573120, from Rockford to St. Petersburg, Departure Time: 19:00, Arrival Time: 22:43",
                    "breakfast": "Grain and Berry Cafe - Westchase, Rockford",
                    "attraction": "Klehm Arboretum & Botanic Garden, Rockford;Sinnissippi Park, Rockford;",
                    "lunch": "Sally O' Neal's Pizza Hotline, Rockford",
                    "dinner": "Zelda's Cafe & Deli, Rockford",
                    "accommodation": "-",
                },
            ]
            st.info(
                "*Example:* Trip from St. Petersburg to Rockford over 3 days, with a budget of $1700"
            )
            st.markdown("### 🗓️ Itinerary at a glance")
            st.markdown("This is a quick overview of your daily schedule.")
            st.markdown(renderer._render_round_trip_check(plan))
            st.markdown(
                renderer._render_travel_summary_table(
                    plan,
                    self._travel_db,
                    1,
                    header=True,
                    footer=True,
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            "Think about who (if anyone) you would travel with and what your constraints / preferences are. The assistant should personalize the travel plan to your party, picking restaurants, accommodations, and attractions you like."
        )
        st.markdown("### Making sure your travel plan is valid")
        st.markdown(
            "To successfully complete this task, your travel plan must *be valid.* A valid plan must:"
        )
        st.markdown(
            "* ONLY use transportation, accommodations, restaurants, and attractions from the assistant's database. Using other options is not allowed."
        )

        with st.container(border=True):
            modified_plan = [
                {
                    "days": 1,
                    "current_city": "from St. Petersburg to Rockford",
                    "transportation": "Flight Number: F3573659, from St. Petersburg to Rockford, Departure Time: 15:40, Arrival Time: 17:04",
                    "breakfast": "-",
                    "attraction": "-",
                    "lunch": "-",
                    "dinner": "Made-up restaurant, Rockford",
                    "accommodation": "Pure luxury one bdrm + sofa bed on Central Park, Rockford",
                },
                {
                    "days": 2,
                    "current_city": "Rockford",
                    "transportation": "-",
                    "breakfast": "McAlister's Deli, Rockford",
                    "attraction": "Burpee Museum of Natural History, Rockford;Midway Village Museum, Rockford;Discovery Center Museum, Rockford;",
                    "lunch": "Poke Express, Rockford",
                    "dinner": "Al-Sham Palace, Rockford",
                    "accommodation": "Pure luxury one bdrm + sofa bed on Central Park, Rockford",
                },
                {
                    "days": 3,
                    "current_city": "from Rockford to St. Petersburg",
                    "transportation": "Flight Number: F3573120, from Rockford to St. Petersburg, Departure Time: 19:00, Arrival Time: 22:43",
                    "breakfast": "Grain and Berry Cafe - Westchase, Rockford",
                    "attraction": "Klehm Arboretum & Botanic Garden, Rockford;Sinnissippi Park, Rockford;",
                    "lunch": "Sally O' Neal's Pizza Hotline, Rockford",
                    "dinner": "Zelda's Cafe & Deli, Rockford",
                    "accommodation": "-",
                },
            ]

            st.info(
                f":red[:material/close: *Example:* This is an invalid plan because it uses a made-up restaurant, designated by the :material/error: icon]"
            )
            st.markdown("### 🗓️ Itinerary at a glance")
            st.markdown("This is a quick overview of your daily schedule.")
            st.markdown(renderer._render_round_trip_check(modified_plan))
            st.markdown(
                renderer._render_travel_summary_table(
                    modified_plan,
                    self._travel_db,
                    1,
                    header=True,
                    footer=True,
                ),
                unsafe_allow_html=True,
            )

        st.markdown("* Is a round trip, starting and ending at the same city.")

        with st.container(border=True):
            modified_plan = [
                {
                    "days": 1,
                    "current_city": "from St. Petersburg to Rockford",
                    "transportation": "Flight Number: F3573659, from St. Petersburg to Rockford, Departure Time: 15:40, Arrival Time: 17:04",
                    "breakfast": "-",
                    "attraction": "-",
                    "lunch": "-",
                    "dinner": "Bazille, Rockford",
                    "accommodation": "Pure luxury one bdrm + sofa bed on Central Park, Rockford",
                },
                {
                    "days": 2,
                    "current_city": "Rockford",
                    "transportation": "-",
                    "breakfast": "McAlister's Deli, Rockford",
                    "attraction": "Burpee Museum of Natural History, Rockford;Midway Village Museum, Rockford;Discovery Center Museum, Rockford;",
                    "lunch": "Poke Express, Rockford",
                    "dinner": "Al-Sham Palace, Rockford",
                    "accommodation": "Pure luxury one bdrm + sofa bed on Central Park, Rockford",
                },
                {
                    "days": 3,
                    "current_city": "Rockford",
                    "transportation": "-",
                    "breakfast": "Grain and Berry Cafe - Westchase, Rockford",
                    "attraction": "Klehm Arboretum & Botanic Garden, Rockford;Sinnissippi Park, Rockford;",
                    "lunch": "Sally O' Neal's Pizza Hotline, Rockford",
                    "dinner": "Zelda's Cafe & Deli, Rockford",
                    "accommodation": "-",
                },
            ]
            st.info(
                f":red[:material/close: *Example:* This is an invalid plan because it is not a round trip: the trip does not return to the origin city on the last day (see the 'Cities' row).]"
            )

            st.markdown("### 🗓️ Itinerary at a glance")
            st.markdown("This is a quick overview of your daily schedule.")
            st.markdown(renderer._render_round_trip_check(modified_plan))
            st.markdown(
                renderer._render_travel_summary_table(
                    modified_plan,
                    self._travel_db,
                    1,
                    header=True,
                    footer=True,
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            "* Specifies the plan for each day of the trip. A plan that says, 'You decide' is not valid; all details must be ironed out."
        )
        st.markdown(
            "* There must be an accommodation listed every day, except the last day, when you will be returning home."
        )

        with st.container(border=True):
            modified_plan = [
                {
                    "days": 1,
                    "current_city": "from St. Petersburg to Rockford",
                    "transportation": "Flight Number: F3573659, from St. Petersburg to Rockford, Departure Time: 15:40, Arrival Time: 17:04",
                    "breakfast": "-",
                    "attraction": "-",
                    "lunch": "-",
                    "dinner": "Bazille, Rockford",
                    "accommodation": "Pure luxury one bdrm + sofa bed on Central Park, Rockford",
                },
                {
                    "days": 2,
                    "current_city": "Rockford",
                    "transportation": "-",
                    "breakfast": "McAlister's Deli, Rockford",
                    "attraction": "Burpee Museum of Natural History, Rockford;Midway Village Museum, Rockford;Discovery Center Museum, Rockford;",
                    "lunch": "Poke Express, Rockford",
                    "dinner": "Al-Sham Palace, Rockford",
                    "accommodation": "-",
                },
                {
                    "days": 3,
                    "current_city": "from Rockford to St. Petersburg",
                    "transportation": "Flight Number: F3573120, from Rockford to St. Petersburg, Departure Time: 19:00, Arrival Time: 22:43",
                    "breakfast": "Grain and Berry Cafe - Westchase, Rockford",
                    "attraction": "Klehm Arboretum & Botanic Garden, Rockford;Sinnissippi Park, Rockford;",
                    "lunch": "Sally O' Neal's Pizza Hotline, Rockford",
                    "dinner": "Zelda's Cafe & Deli, Rockford",
                    "accommodation": "-",
                },
            ]
            st.info(
                f":red[:material/close: *Example:* This is an invalid plan because it does not specify an accommodation for the second day.]"
            )
            st.markdown("### 🗓️ Itinerary at a glance")
            st.markdown("This is a quick overview of your daily schedule.")
            st.markdown(renderer._render_round_trip_check(modified_plan))
            st.markdown(
                renderer._render_travel_summary_table(
                    modified_plan,
                    self._travel_db,
                    1,
                    header=True,
                    footer=True,
                ),
                unsafe_allow_html=True,
            )

        st.markdown("Other notes:")
        st.markdown("* You can visit multiple attractions per day.")
        st.markdown("* You can only eat at one restaurant per meal.")
        st.markdown(
            "* Accommodations have rules: you must follow those rules when booking."
        )


def get_driving_actions(refs: Dict[str, str]) -> List[Action]:
    """
    Get driving actions for the driving info.
    """

    @tool(parse_docstring=True)
    def get_driving_options() -> str:
        """
        View taxi and self-driving options for the trip.
        If nothing is returned, you must use flights instead.
        """
        return refs

    return [
        Action(
            fn=get_driving_options,
            is_public=True,
            is_human=False,
            name="Get driving options",
        )
    ]


def user_specification_callback(
    form_results: dict[str, Any], callback_kwargs: dict
) -> dict:
    """
    Process user form and generate hard constraints.
    """
    people_number = form_results.get(
        "How many people are you likely to travel with? (0 for solo travel)?"
    )
    kids = form_results.get("Does your travel party include kids?", False)
    if people_number is None:
        return {}

    try:
        people_number = int(people_number) + 1
    except (ValueError, TypeError):
        people_number = 1
    validity_kwargs = callback_kwargs.get("validity_kwargs", {})
    query_data = validity_kwargs.get("query_data", {})
    query_data["people_number"] = people_number
    validity_kwargs["query_data"] = query_data
    new_specification = callback_kwargs.get("initial_specification", "")
    if people_number is not None:
        new_specification += f" | Party size: {people_number}"
    if kids:
        new_specification += " | Kid friendly"
    return {
        "validity_kwargs": validity_kwargs,
        "people_number": people_number,
        "current_specification": new_specification,
    }


def validity_fn(
    yhat: str, query_data: dict, raise_errors: bool = False
) -> Tuple[bool, dict]:
    """
    Check if the travel plan is valid by checking commonsense constraints and budget.
    """
    yhat_parsed = parse_json(yhat)
    if yhat_parsed is None:
        if raise_errors:
            raise Exception("Could not parse a travel plan from the message.")
        return False, {"error": "Could not parse JSON"}

    # Check commonsense constraints
    commonsense_info = commonsense_eval(query_data, yhat_parsed)
    violated_constraints = []
    if commonsense_info:
        for constraint, (passed, message) in commonsense_info.items():
            if passed is not None and not passed:
                if message is not None:
                    violated_constraints.append(message)
                else:
                    violated_constraints.append(constraint)

    # Check hard constraints
    try:
        hard_info = hard_eval(query_data, yhat_parsed)
        if hard_info:
            for constraint, (passed, message) in hard_info.items():
                if passed is not None and not passed:
                    if message is not None:
                        violated_constraints.append(message)
                    else:
                        violated_constraints.append(constraint)
    except Exception as e:
        if raise_errors:
            raise Exception(f"Error when evaluating hard constraints: {e}")
        violated_constraints.append(f"Error calculating cost: {e}")

    is_valid = len(violated_constraints) == 0
    metadata = {
        "violated_constraints": violated_constraints,
        "commonsense_info": commonsense_info,
        "hard_info": hard_info if "hard_info" in locals() else None,
    }

    if raise_errors and not is_valid:
        # Randomly pick one violated constraint to report
        if violated_constraints:
            raise Exception(
                f"The travel plan violated the following constraint: {random.choice(violated_constraints)}"
            )
        else:
            raise Exception("The travel plan violated some constraints.")

    return is_valid, metadata


def reward_fn(
    yhat: str,
    query_data: dict,
    weights: Dict[str, float],
    raise_errors: bool = False,
) -> Tuple[float, dict]:
    """
    Evaluate a single travel plan's preference score.

    Args:
        yhat: The predicted plan
        query_data: The original row from the dataset
        weights: Dict[str, float] vector of weights for preference scoring
        raise_errors: Whether to raise errors on invalid input

    Returns:
        Tuple of (score, metadata) where score is the preference score
    """
    yhat_parsed = parse_json(yhat)
    if yhat_parsed is None:
        if raise_errors:
            raise Exception("Could not parse a travel plan from the message.")
        return float("-inf"), {"error": "Could not parse JSON"}

    # Evaluate preferences
    try:
        preferences_info = preferences_eval(query_data, yhat_parsed)
        assert preferences_info is not None, "preferences_info is None"
        score, min_unconstrained_score, max_unconstrained_score = compute_linear_reward(
            weights, preferences_info
        )
        score = (score - min_unconstrained_score) / (
            max_unconstrained_score - min_unconstrained_score
        )
        score *= 100

        metadata = {
            "preference_details": preferences_info,
            "parsed_plan": yhat_parsed,
        }

        return score, metadata

    except Exception as e:
        if raise_errors:
            raise Exception(f"Error when evaluating preferences: {e}")
        return float("-inf"), {"error": str(e)}


def output_to_streamlit_comparison(
    y1: str,
    y2: str,
    db: TravelDB,
    people_number: int,
    validity_fn=None,
    validity_kwargs=None,
) -> None:
    parsed1 = parse_json(y1)
    parsed2 = parse_json(y2)
    a_valid = a_metadata = b_valid = b_metadata = None
    if validity_fn and validity_kwargs:
        a_valid, a_metadata = validity_fn(
            y1, **(validity_kwargs or {}), raise_errors=False
        )
        b_valid, b_metadata = validity_fn(
            y2, **(validity_kwargs or {}), raise_errors=False
        )
    renderer.output_to_streamlit_comparison(
        parsed1, parsed2, db, people_number, a_valid, b_valid, a_metadata, b_metadata
    )


def output_to_streamlit(msg: str, db: TravelDB, people_number: int) -> None:
    from utils.misc import parse_for_answer_tags

    # Parse travel plan JSON
    js, start_end = parse_json(msg, return_start_end=True)

    # Parse travel mentions
    mentioned_travel = parse_for_answer_tags(
        msg, keyword="travel", return_all=True, return_none_if_not_found=True
    )
    if mentioned_travel is not None:
        # Don't split on commas since travel items are in "Name, City" format
        mentioned_travel = [
            travel.strip() for travel in mentioned_travel if travel.strip()
        ]
        mentioned_travel = list(set(mentioned_travel))

    if js is None:
        # No travel plan, just render the message with travel mentions
        st.markdown(
            replace_tags_with_link(msg, "travel", "#options-mentioned-in-message"),
            unsafe_allow_html=True,
        )
        if mentioned_travel:
            with st.expander("Travel items mentioned in message", expanded=False):
                renderer.render_travel_mentions(mentioned_travel, db)
        return

    if start_end[0] > 0:
        st.markdown(
            replace_tags_with_link(
                msg[: start_end[0]], "travel", "#options-mentioned-in-message"
            ),
            unsafe_allow_html=True,
        )

    renderer.render_travel_plan_streamlit(js, db, people_number)

    if start_end[1] < len(msg):
        st.markdown(
            replace_tags_with_link(
                msg[start_end[1] :], "travel", "#options-mentioned-in-message"
            ),
            unsafe_allow_html=True,
        )

    # Render travel mentions if any
    if mentioned_travel:
        st.markdown("---")
        with st.expander("Travel items mentioned in message", expanded=False):
            renderer.render_travel_mentions(mentioned_travel, db)
