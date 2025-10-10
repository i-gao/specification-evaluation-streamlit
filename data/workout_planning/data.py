from datasets import load_dataset
import numpy as np
from collections import defaultdict
import tqdm
from typing import List, Tuple, Optional, Dict, Callable, Any
import dill
import sys
import os
import json
import sys
import streamlit as st
from langchain_core.tools import tool
import re

from llm_sandbox import SandboxSession

from data.dataset import (
    SpecificationCollection,
    FixedSpecification,
    CustomSpecification,
)
from data.actions import Action, get_jupyter_actions
import data.workout_planning.streamlit_render as renderer
from utils.streamlit_types import FormElement, DisplayElement, form_element_to_streamlit

from utils.misc import (
    get_recursive,
    parse_json,
    subset_data,
    add_section,
    replace_tags_with_link,
)
from data.workout_planning.db import (
    ExerciseDB,
    DAYS_OF_THE_WEEK,
    TIMES_OF_DAY,
)
from data.reward import linear_reward, Constraint

DEV_FRAC = 0.1
DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))

FMT_INSTRUCTIONS = (
    """You MUST use ONLY exercises in the database, and you MUST structure your plan as a JSON string with the following structure:
{
    "sunday": {
        "early morning (6-9am)": [
            {
                "exercise_name": str (required), 
                "time_or_reps": Optional[Literal["time", "reps"]],
                "num_sets": Optional[int],
                "rest_time": Optional[int],
                "time_per_set": Optional[int],
                "num_reps_per_set": Optional[int],
            },
            ...
        ]
        ...
    },
    ...
}
""".strip()
    + f"\nThe outer keys are the days of the week (choose from {DAYS_OF_THE_WEEK}), and the inner keys are the times of day (choose from {TIMES_OF_DAY})."
    + """Each slot is either null or a list of dictionaries. The dictionaries have the following fields:
    - `exercise_name` is the name of the exercise, copied exactly from the database. All exercises must come from the database. This field is required.
    - `time_or_reps` is the type of set (time-based or rep-based), copied exactly from the database.
    - `num_sets` is the number of sets of the exercise, copied exactly from the database.
    - `rest_time` is the rest time between sets of the exercise, copied exactly from the database.
    - `time_per_set` is the time per set of the exercise, copied exactly from the database.
    - `num_reps_per_set` is the number of reps per set of the exercise, copied exactly from the database.
Only exercises and variations from the database can be used.

To render a description of a single exercise (instead of a full workout plan) to the user, you can mention its exercise_name and wrap it in <exercise></exercise>, e.g.: '<exercise>Bodyweight Glute Bridge</exercise>'. Do not put <exercise></exercise> tags inside the JSON of a full workout plan.
""".strip()
)

CUSTOM_INSTRUCTIONS = """
### What you need to prompt the assistant to do
In this task, **your goal is to get the assistant to write you a perfect workout plan that you can actually follow this coming week.** A workout plan is a week-long calendar that specifies when to work out and what to do for each workout.

The plan should work with your schedule, ability level, equipment access, goals, and preferences. The assistant actually needs to specify when you will work out, and how long to do each exercise in the workout. 

Make sure you review the exercises and watch any provided videos to understand their difficulty level.

If you don't have any experience with exercise, the assistant should help you figure out what to do.
"""

FIXED_INSTRUCTIONS = """
### What you need to prompt the assistant to do
In this task, you are acting as a personal trainer. **Your goal is to get the assistant to write the perfect workout plan for a client.** A workout plan is a week-long calendar that specifies when to work out and what to do for each workout.

The plan must work with the client's ability level, gym access, goals, and preferences. THe timing of workouts must also fit their schedule. When the chat session starts, some information about the client will appear on the left side of the screen.

### The tricky part
Some of the client's details may be missing. For example, they may not have specified what kinds of workouts they like. 

To maximize your score, you will have to try different workout plans and ask the client to evaluate them. The client's score will be between 0 and 100. If the workout plan is invalid, dangerous for the client to follow, or does not fit their schedule, then the score will be -infinity.
"""

COMMONSENSE_INSTRUCTIONS = """
A valid workout plan:
* ONLY uses exercises from the assistant's database. Using other exercises is not allowed.
* Specifies the plan for each day of the week. A plan that says, "You decide" is not valid; all details must be ironed out.
* Specifies the time of day to work out for each day.
* Specifies the exercises to do for each workout.
"""


def render_fixed_task_explanation():
    """Render the fixed task explanation for workout planning."""
    st.markdown(FIXED_INSTRUCTIONS)
    st.markdown(COMMONSENSE_INSTRUCTIONS)


def render_custom_task_explanation():
    """Render the custom task explanation for workout planning."""
    st.markdown(CUSTOM_INSTRUCTIONS)
    st.markdown(COMMONSENSE_INSTRUCTIONS)


class WorkoutPlanningDataset(SpecificationCollection):
    """
    The WorkoutPlanning benchmark evaluates how well LMs can generate personalized
    workout plans which obey some constraints.

    We wrote this dataset.
    Original exercises: https://strengthtoovercome.com/functional-fitness-exercise-database
    Profiles: programmatically generated with GPT-4.1

    Each profile is treated as a separate "task" to solve. A profile consists of information
    about a person, including their fitness goals, preferences, and constraints.
    - The dataset consists of a single (x), representing the set of available exercises.

    Dev set split:
    - 10% of cases in dev set
    - Disjoint cases in test set
    """

    @property
    def dataset_name(self) -> str:
        return "workout_planning"

    @property
    def assets_file_id(self) -> str:
        return None

    @property
    def dataset_pretty_name(self) -> str:
        return "Workout Planning"

    @property
    def dataset_description(self) -> str:
        return "Work with the assistant to **write a personalized workout plan for the next week.**"

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
                "image_name": "workout_planning",
                "dockerfile_path": "data/workout_planning/reward_utils/Dockerfile",
                "build_context": "data/workout_planning",
                "description": "Docker image for Workout Planning code evaluation",
            },
        ]

    def _create_user_expertise_form(self) -> List[FormElement]:
        """
        Create user expertise form elements for workout planning domain knowledge.
        """
        return [
            FormElement(
                input_type="radio",
                label="How often do you currently work out?",
                options=[
                    "Never",
                    "A few times a month",
                    "1-3 times a week",
                    "4+ times a week",
                ],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="How familiar are you with strength training?.",
                options=[
                    "I am not familiar with the terms 'set' or 'rep'.",
                    "I know the terms 'set' and 'rep' but could not design a safe and effective strength training routine for my goals",
                    "I can design a safe and effective strength training routine on my own",
                ],
                required=True,
            ),
        ]

    def _create_user_specification_form_initial(self) -> List[FormElement]:
        """
        Create initial form elements for basic workout planning requirements.
        """
        return [
            FormElement(
                input_type="radio",
                label="Do you have access to a full gym?",
                options=["Yes", "No"],
                default="No",
                required=True,
                help="A full gym includes equipment like dumbbells, barbells, resistance bands, treadmills, etc.",
            )
        ]

    def _create_user_specification_form_final(self) -> List[FormElement]:
        """
        Create final form elements for detailed workout planning requirements.
        """
        return [
            FormElement(
                input_type="radio",
                label="What is the maximum number of workouts you want to do per week?",
                options=["1", "2", "3", "4", "5", "6", "7"],
                default="7",
                required=True,
                help="This helps us design workouts that fit your schedule",
            ),
            FormElement(
                input_type="radio",
                label="What is the maximum time you want to spend on each workout?",
                options=["30min", "60min", "90min", "120min"],
                default="60min",
                required=True,
                help="This helps us design workouts that fit your schedule",
            ),
        ]

    def _create_user_evaluation_form(self) -> List[FormElement]:
        """Create the user evaluation form for workout planning."""
        return [
            FormElement(
                input_type="radio",
                label="Compare the **difficulty levels** of plans A and B. Which one do you prefer?",
                options=["A", "neutral", "B"],
                default="0",
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare how well plans A and B align with your **fitness goals**. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare how well plans A and B align with your **schedule**. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare how well plans A and B align with your **equipment availability**. Which one do you prefer?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Compare how well plans A and B accommodate your **injury / movement constraints**. Which one do you prefer?",
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

        # Load all the profiles
        profiles = dill.load(open(f"{DATASET_ROOT}/assets/full-profiles.pkl", "rb"))
        profiles = subset_data(profiles, DEV_FRAC, 1.0, dev)
        self._profiles = profiles
        self.fixed_length = len(self._profiles)

        # Only 1 custom spec for workout planning
        self.custom_length = 1

        self._docker_image = docker_image
        self._persist_docker_container = persist_docker_container

        # Load the exercises database to get column information
        self._exercise_db = ExerciseDB()
        # Import extractors and build lookup
        import data.workout_planning.extractors as extractors_mod

        self._extractor_lookup = {
            name: func
            for name, func in extractors_mod.__dict__.items()
            if callable(func)
        }
        # Create description JSON for the exercises CSV
        self._desc_json = {
            "filename": "exercises_with_variations.csv",
            "description": "Database of exercises (e.g. 'Stability Ball Dead Bug') and variations (e.g. '3 sets of 20 seconds with 40 seconds rest') to use for creating workouts.",
            "columns": self._exercise_db._list_columns("exercises"),
        }

        # Load y0 values for different initial form combinations
        js = json.load(open(f"{DATASET_ROOT}/assets/y0_mapping.json"))
        self._y0_mapping = js

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
            profile = self._profiles[ix]

            # constraints & theta
            constraints = [
                Constraint.from_dict(c, extractor_lookup=self._extractor_lookup)
                for c in profile["constraints"]
            ]

            theta = "Below is a summary of the client's hard and soft constraints."
            theta += "\n\n" + add_section(
                "Hard constraints (must be satisfied)", profile["hard_description"]
            )
            theta += "\n\n" + add_section(
                "Soft constraints (most important to least important)",
                profile["soft_description"].replace("\n", "\n<chunk>\n"),
            )
            theta += "\n<chunk>\nNote that barring rest day constraints, more workouts per week will be more effective."

            if self._persist_docker_container and self._docker_image is not None:
                session = SandboxSession(image=self._docker_image)
                session.open()
                container_id = session.container.id
            else:
                container_id = None

            filename, actions = get_jupyter_actions(
                docker_image=self._docker_image,
                docker_container_id=container_id,
                ls_output=self._desc_json,
                root_dir=os.path.join(DATASET_ROOT, "assets"),
            )

            signature = (
                "Design a workout plan for the following client: " + profile["text"]
            )
            # signature += "\n\nBased on this information, here are the client's hard constraints:\n" + profile["hard_description"]

            # Split constraints into hard and soft
            hard_constraints = [c for c in constraints if c.is_hard]
            soft_constraints = [c for c in constraints if not c.is_hard]

            # specification
            spec = FixedSpecification(
                dataset_name=self.dataset_name,
                index=f"fixed_{ix}",
                full_specification=theta,
                initial_specification=signature,
                validity_fn=validity_fn,
                validity_kwargs={
                    "hard_constraints": hard_constraints,
                    "exercise_db": self._exercise_db,
                },
                # validity_fn_tool_name=None,  # Not provided
                # validity_fn_tool_description=None,  # Not provided
                reward_fn=reward_fn,
                reward_kwargs={
                    "soft_constraints": soft_constraints,
                    "weights": profile["weights"],
                    "exercise_db": self._exercise_db,
                },
                # reward_fn_tool_name=None,  # Not provided
                # reward_fn_tool_description=None,  # Not provided
                ystar=None,  # No ystar for workout planning
                # metric_name=None,  # Not provided
                # baseline_scores=None,  # Not provided
                render_task_explanation=render_fixed_task_explanation,
                actions=actions,
                fmt_instructions=FMT_INSTRUCTIONS,
                render_msg_fn=output_to_streamlit,
                render_msg_kwargs=["db"],
                db=self._exercise_db,
                name=f"workout_planning_{ix}",
                state_files=[filename],
                files_to_clean=[filename],
                container_ids=[container_id],
                user_expertise_form=self._create_user_expertise_form(),
            )
            specs[ix] = spec
        return specs

    def _load_custom_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, CustomSpecification]:
        """
        Create custom workout planning specifications.
        """
        if indexes is None:
            return {}

        specs = {}
        for ix in indexes:
            if self._persist_docker_container and self._docker_image is not None:
                session = SandboxSession(image=self._docker_image)
                session.open()
                container_id = session.container.id
            else:
                container_id = None

            # Get Jupyter actions for the environment
            filename, actions = get_jupyter_actions(
                docker_image=self._docker_image,
                docker_container_id=container_id,
                ls_output=self._desc_json,
                root_dir=os.path.join(DATASET_ROOT, "assets"),
            )

            # Start with basic constraints that will be updated by the callback
            initial_constraints = [
                Constraint.create_boolean_penalize_false_constraint(
                    description="Workout plan must have at least one workout per week",
                    extractor="min_workouts_satisfied",
                    extractor_kwargs={"min_workouts": 1},
                    is_hard=True,
                ),
                Constraint.create_boolean_penalize_false_constraint(
                    description="All exercises must be valid",
                    extractor="all_valid",
                    is_hard=True,
                ),
            ]
            initial_constraints = [
                Constraint.from_dict(c, extractor_lookup=self._extractor_lookup)
                for c in initial_constraints
            ]

            spec = CustomSpecification(
                dataset_name=self.dataset_name,
                index=f"custom_{ix}",
                initial_specification="Design a workout plan for your next week.",
                user_specification_form_initial=self._create_user_specification_form_initial(),
                user_specification_form_final=self._create_user_specification_form_final(),
                user_specification_callback=user_specification_callback,
                user_specification_callback_kwargs=[
                    "_validity_kwargs",
                    "_y0_mapping",
                    "_extractor_lookup",
                    "initial_specification",
                ],
                validity_fn=validity_fn,
                validity_kwargs={
                    "hard_constraints": initial_constraints,
                    "exercise_db": self._exercise_db,
                },
                validity_fn_tool_name="check_workout_plan_validity",
                validity_fn_tool_description="Check if the workout plan is valid and follows constraints",
                render_task_explanation=self._render_custom_task_explanation,
                actions=actions,
                fmt_instructions=FMT_INSTRUCTIONS,
                render_msg_fn=output_to_streamlit,
                render_msg_kwargs=["db"],
                db=self._exercise_db,
                render_comparison_fn=output_to_streamlit_comparison,
                name=f"custom_workout_planning_{ix}",
                state_files=[filename],
                files_to_clean=[filename],
                container_ids=[container_id],
                user_expertise_form=self._create_user_expertise_form(),
                _y0_mapping=self._y0_mapping,
                _extractor_lookup=self._extractor_lookup,
                render_evaluation_fn=lambda **kwargs: renderer.render_eval_workout(
                    **kwargs, db=self._exercise_db
                ),
            )
            specs[ix] = spec
        return specs

    def _render_custom_task_explanation(self):
        """Render the custom task explanation for workout planning."""

        st.markdown("### What you need to prompt the assistant to do")
        st.markdown(
            "In this task, **your goal is to get the assistant to write you a perfect workout plan that you can actually follow this coming week.** A workout plan is a week-long calendar that specifies when to work out and what to do for each workout. A workout is a list of exercises that you will do in one session."
        )
        st.markdown(
            "The workout plan should be personalized to your fitness goals, equipment availability, injuries, and experience level."
        )

        with st.container(border=True):
            # Example workout plan with valid exercises
            example_plan = {
                "monday": {
                    "early morning (6-9am)": [
                        {
                            "exercise_name": "Stability Ball Dead Bug",
                            "time_or_reps": "time",
                            "num_sets": 3,
                            "rest_time": 40,
                            "time_per_set": 20,
                        },
                        {
                            "exercise_name": "Superband Pull Apart",
                            "time_or_reps": "time",
                            "num_sets": 2,
                            "rest_time": 60,
                            "time_per_set": 60,
                        },
                        {
                            "exercise_name": "Slider Reverse Lunge",
                            "time_or_reps": "rep",
                            "num_sets": 3,
                            "rest_time": 60,
                            "time_per_set": 30,
                            "num_reps_per_set": 10,
                        },
                    ]
                },
                "wednesday": {
                    "early morning (6-9am)": [
                        {
                            "exercise_name": "Stability Ball Dead Bug",
                            "time_or_reps": "time",
                            "num_sets": 3,
                            "rest_time": 40,
                            "time_per_set": 20,
                        },
                        {
                            "exercise_name": "Superband Pull Apart",
                            "time_or_reps": "time",
                            "num_sets": 2,
                            "rest_time": 60,
                            "time_per_set": 60,
                        },
                        {
                            "exercise_name": "Slider Reverse Lunge",
                            "time_or_reps": "rep",
                            "num_sets": 3,
                            "rest_time": 60,
                            "time_per_set": 30,
                            "num_reps_per_set": 10,
                        },
                    ]
                },
            }
            st.info(
                "*Example:* A 2-day workout plan with exercises for Monday and Wednesday mornings"
            )
            # Parse the example plan first
            parsed_plan = parse_workout_plan(
                json.dumps(example_plan), self._exercise_db, leave_invalid=True
            )
            st.markdown(
                "\n".join(renderer._render_calendar_table(parsed_plan)),
                unsafe_allow_html=True,
            )

        st.markdown(
            "Think about your fitness goals, schedule, equipment access, and experience level. The assistant should personalize the workout plan to your needs, picking exercises that match your ability and available equipment."
        )
        st.markdown("### Making sure your workout plan is valid")
        st.markdown(
            "To successfully complete this task, your workout plan must *be valid.* A valid plan must ONLY use exercises from the assistant's database. Using other exercises is not allowed."
        )

        with st.container(border=True):
            # Example with invalid exercise
            invalid_plan = {
                "monday": {
                    "early morning (6-9am)": [
                        {
                            "exercise_name": "Stability Ball Dead Bug",
                            "time_or_reps": "time",
                            "num_sets": 3,
                            "rest_time": 40,
                            "time_per_set": 20,
                        },
                        {
                            "exercise_name": "Made-up Exercise",
                            "time_or_reps": "rep",
                            "num_sets": 3,
                            "rest_time": 60,
                            "num_reps_per_set": 10,
                        },
                    ]
                }
            }
            st.info(
                ":red[:material/close: *Example:* This is an invalid plan because it includes a made-up exercise, designated by the :material/error: icon]"
            )
            # Parse the invalid plan first
            parsed_invalid_plan = parse_workout_plan(
                json.dumps(invalid_plan), self._exercise_db, leave_invalid=True
            )
            st.markdown(
                "\n".join(renderer._render_calendar_table(parsed_invalid_plan)),
                unsafe_allow_html=True,
            )

        st.markdown("Other notes:")
        st.markdown(
            "* You will be able to view details of the recommended exercises. You can watch exercise demonstration videos to understand proper form and difficulty level."
        )
        with st.container(border=True):
            st.info("Here is an example exercise")
            st.markdown(
                renderer._render_exercise_details(
                    0, parsed_plan["monday"]["early morning (6-9am)"][0]
                ),
                unsafe_allow_html=True,
            )


def user_specification_callback(
    form_results: dict[str, Any], callback_kwargs: dict
) -> dict:
    """
    Process form results and return updates for the specification.
    This callback handles both initial and final form results.
    """
    # Get validity_kwargs from callback_kwargs
    validity_kwargs = callback_kwargs.get("_validity_kwargs", {})
    constraints = [
        Constraint.create_boolean_penalize_false_constraint(
            description="Workout plan must have at least one workout per week",
            extractor="min_workouts_satisfied",
            extractor_kwargs={"min_workouts": 1},
            is_hard=True,
        ),
        Constraint.create_boolean_penalize_false_constraint(
            description="All exercises must be valid",
            extractor="all_valid",
            is_hard=True,
        ),
    ]

    # Add constraints based on form responses
    max_workout_time = form_results.get(
        "What is the maximum time you want to spend on each workout?", None
    )
    max_workouts_per_week = form_results.get(
        "What is the maximum number of workouts you want to do per week?", None
    )
    gym_access = form_results.get("Do you have access to a full gym?", None)

    # Get y0 from callback_kwargs using gym access and max workout time
    if max_workout_time is not None and max_workouts_per_week is not None:
        y0_mapping = callback_kwargs.get("_y0_mapping", {})
        y0 = y0_mapping.get(str(max_workout_time).replace("min", ""), {}).get(
            str(max_workouts_per_week), None
        )
    else:
        y0 = None

    if max_workout_time is not None:
        # Add workout duration constraints based on max_workout_time
        for day in DAYS_OF_THE_WEEK:
            for time in TIMES_OF_DAY:
                constraints.append(
                    Constraint.create_boolean_penalize_false_constraint(
                        description=f"Workout at {day.capitalize()} {time} must be under {max_workout_time}",
                        extractor="workout_duration_under",
                        extractor_kwargs={
                            "day": day,
                            "time": time,
                            "max_duration_minutes": int(
                                max_workout_time.replace("min", "")
                            ),
                        },
                        is_hard=True,
                    )
                )

    if max_workouts_per_week is not None:
        # Add workout count constraints based on max_workouts_per_week
        constraints.append(
            Constraint.create_boolean_penalize_false_constraint(
                description=f"Workout plan must have at most {max_workouts_per_week} workouts per week",
                extractor="min_rest_days_satisfied",
                extractor_kwargs={"min_rest_days": 7 - int(max_workouts_per_week)},
                is_hard=True,
            )
        )

    # Update validity_kwargs
    constraints = [
        Constraint.from_dict(
            c, extractor_lookup=callback_kwargs.get("_extractor_lookup", {})
        )
        for c in constraints
    ]
    validity_kwargs["hard_constraints"] = constraints

    # Get new specification from callback_kwargs
    new_specification = callback_kwargs.get("initial_specification") or ""
    if max_workouts_per_week is not None:
        new_specification += f" | Workouts per week: {max_workouts_per_week}"
    if max_workout_time is not None:
        new_specification += f" | Time per workout: {max_workout_time}"
    if gym_access is not None:
        new_specification += (
            f" | Equipment access: {'full gym' if gym_access == 'Yes' else 'limited'}"
        )

    # Return updates for the specification object
    return {
        "validity_kwargs": validity_kwargs,
        "y0": y0,
        "current_specification": new_specification,
        "_render_evaluation_kwargs": {
            "y0": y0,
        },
    }


def validity_fn(
    yhat: str,
    hard_constraints: List[Constraint],
    exercise_db: ExerciseDB,
    raise_errors: bool = False,
) -> Tuple[bool, dict]:
    """
    Evaluate a single workout plan against its constraints and return detailed violation information.
    """
    workout_plan = parse_workout_plan(
        yhat, exercise_db, raise_errors=raise_errors, leave_invalid=True
    )
    if workout_plan is None:
        if raise_errors:
            raise Exception("Could not parse a workout plan from the message.")
        return False, {"parsed_plan": None}

    try:
        is_valid, score, min_unconstrained_score, max_unconstrained_score, metadata = (
            linear_reward(
                workout_plan,
                constraints=hard_constraints,
                weights=None,
                enforce_hard=True,
                raise_errors=raise_errors,
            )
        )
    except Exception as e:
        if raise_errors:
            raise Exception(e)
        return False, {"parsed_plan": workout_plan, "error": str(e)}

    return is_valid, metadata


def reward_fn(
    yhat: str,
    soft_constraints: List[Constraint],
    weights: np.ndarray,
    exercise_db: ExerciseDB,
    raise_errors: bool = False,
) -> Tuple[bool, float, dict]:
    """
    Evaluate a single workout plan against its soft constraints and return detailed violation information.

    Args:
        yhat: The predicted workout plan
        soft_constraints: The soft constraints of the task
        weights: The weights of the soft constraints
        exercise_db: The database of exercises
    """
    # convert yhat to a workout plan
    workout_plan = parse_workout_plan(yhat, exercise_db, raise_errors=raise_errors)
    if workout_plan is None:
        if raise_errors:
            raise Exception("Could not parse a workout plan from the message.")
        return False, float("-inf"), {}

    try:
        is_valid, score, min_unconstrained_score, max_unconstrained_score, metadata = (
            linear_reward(
                workout_plan,
                constraints=soft_constraints,
                weights=weights,
                enforce_hard=False,
                raise_errors=raise_errors,
            )
        )
    except Exception as e:
        if raise_errors:
            raise Exception(f"The workout plan is invalid: {str(e)}")
        return False, float("-inf"), {"error": str(e)}

    # rescale from real numbers to [0, 1]
    score = (score - min_unconstrained_score) / (
        max_unconstrained_score - min_unconstrained_score
    )

    return (
        is_valid,
        score * 100,  # rescale from [0, 1] to [0, 100]
        metadata,
    )


def parse_workout_plan(
    yhat: str,
    exercise_db: ExerciseDB,
    raise_errors: bool = False,
    leave_invalid: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Parse a workout plan from a JSON string.
    Assumes a dictionary with the following structure:
    {
        "sunday": {
            "early morning (6-9am)": [
                {"exercise_name": str, "variation": str},
                ...
            ],
        }
    }
    and returns a dictionary with the following structure:
    {
        "sunday": {
            "early morning (6-9am)": [
                Exercise (pd.series, row from exercise_db),
                ...
            ],
        }
    }
    """
    workout_plan = parse_json(yhat)
    if workout_plan is None:
        return None

    # do some automatic correction for case, missing fields, etc.
    try:
        # lower case all the keys
        workout_plan = {
            k.lower(): (
                {ki.lower(): vi for ki, vi in v.items()} if type(v) == dict else {}
            )
            for k, v in workout_plan.items()
        }
    except Exception as e:
        print(f"Error parsing workout plan: {workout_plan}: {e}")
        return None

    corrected_workout_plan = {}
    for day in DAYS_OF_THE_WEEK:
        for time_of_day in TIMES_OF_DAY:
            # first, make sure to add the fields to the new dict
            if day not in corrected_workout_plan:
                corrected_workout_plan[day] = {}
            if time_of_day not in corrected_workout_plan[day]:
                corrected_workout_plan[day][time_of_day] = []

            # then, copy over the fields from the old dict
            if (
                day not in workout_plan
                or time_of_day not in workout_plan[day]
                or workout_plan[day][time_of_day] is None
                or len(workout_plan[day][time_of_day]) == 0
            ):
                corrected_workout_plan[day][time_of_day] = None
            else:
                new_list = []
                for d in workout_plan[day][time_of_day]:
                    exercise_name = d.pop("exercise_name")
                    exercise = exercise_db.get_exercise_by_variation(
                        name=exercise_name, **d
                    )
                    if exercise is None and raise_errors:
                        raise Exception(
                            f"Exercise not found in database: {exercise_name}. For this task, plans are only valid if all exercises are from the database."
                        )
                    if exercise is None and not leave_invalid:
                        continue
                    if exercise is None and leave_invalid:
                        exercise = {
                            "exercise_name": exercise_name,
                            "invalid": True,
                        }
                    new_list.append(exercise)
                corrected_workout_plan[day][time_of_day] = new_list

    return corrected_workout_plan


def output_to_streamlit(msg: str, db: ExerciseDB) -> str:
    """
    Convert a message containing a solution attempt to streamlit.
    """
    from utils.misc import parse_for_answer_tags

    # Parse workout plan JSON
    js, start_end = parse_json(msg, return_start_end=True)

    # Parse exercise mentions
    mentioned_exercises = parse_for_answer_tags(
        msg, keyword="exercise", return_all=True, return_none_if_not_found=True
    )
    if mentioned_exercises is not None:
        mentioned_exercises = [
            exercise.strip()
            for mentions in mentioned_exercises
            for exercise in mentions.split(",")
            if exercise.strip()
        ]
        mentioned_exercises = list(set(mentioned_exercises))

    if js is None:
        # No workout plan, just render the message with exercise mentions
        st.markdown(
            replace_tags_with_link(msg, "exercise", "#mentioned-exercises"),
            unsafe_allow_html=True,
        )
        if mentioned_exercises:
            with st.expander("Exercises mentioned in message", expanded=False):
                renderer.render_exercise_mentions(mentioned_exercises, db)
        return

    # Render message before workout plan
    st.markdown(
        msg[: start_end[0]]
        .replace("<exercise>", "<a href='#mentioned-exercises'>")
        .replace("</exercise>", "</a>")
    )

    # Render workout plan
    renderer.render_workout_plan_streamlit(
        parse_workout_plan(msg, db, leave_invalid=True)
    )

    # Render message after workout plan
    st.markdown(
        replace_tags_with_link(msg[start_end[1] :], "exercise", "#mentioned-exercises"),
        unsafe_allow_html=True,
    )

    # Render exercise mentions if any
    if mentioned_exercises:
        st.markdown("---")
        with st.expander("Exercises mentioned in message", expanded=False):
            renderer.render_exercise_mentions(mentioned_exercises, db)


def output_to_streamlit_comparison(
    y1: str, y2: str, db: ExerciseDB, validity_fn=None, validity_kwargs=None
) -> None:
    """
    Convert a message containing a solution attempt to streamlit.
    """

    parsed1 = parse_workout_plan(y1, db, leave_invalid=True)
    parsed2 = parse_workout_plan(y2, db, leave_invalid=True)

    a_valid = a_metadata = b_valid = b_metadata = None
    if validity_fn is not None and validity_kwargs is not None:
        a_valid, a_metadata = validity_fn(
            y1, **(validity_kwargs or {}), raise_errors=False
        )
        b_valid, b_metadata = validity_fn(
            y2, **(validity_kwargs or {}), raise_errors=False
        )

    renderer.output_to_streamlit_comparison(
        parsed1,
        parsed2,
        db,
        a_valid,
        b_valid,
        a_metadata,
        b_metadata,
    )
