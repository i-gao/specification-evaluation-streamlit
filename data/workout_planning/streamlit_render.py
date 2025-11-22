from typing import List, Dict, Any, Optional, Callable
import streamlit as st
import uuid
import re
import random
from data.workout_planning.db import DAYS_OF_THE_WEEK, TIMES_OF_DAY
from data.workout_planning.parser import parse_workout_plan
from evaluation.qualitative_eval import COMPARISON_LIKERT
from data.reward import likert_to_win_rate, pairwise_win_rate
import inflect
import math
import pandas as pd

# Session state key prefixes used by this render module that should be cleared between rounds
RENDER_SESSION_STATE_KEY_PREFIXES = [
    "workout_options_order_",
    "workout_comparison_",
]


def render_eval(
    *,
    final_prediction: str,
    y0: Optional[str],
    db,
    num_items_per_comparison: int = 5,
    **kwargs,
):
    """
    Render evaluation UI: first rank workouts, then show A vs B comparison and collect Likert-scale preferences.
    Returns (completed, feedback_dict_or_none).
    """
    ranking_done = rank_workouts(
        final_prediction=final_prediction,
        y0=y0,
        db=db,
        num_items_per_comparison=num_items_per_comparison,
    )
    if not ranking_done:
        return False, None

    comparison_done = render_comparison(final_prediction=final_prediction, y0=y0, db=db)
    if not comparison_done:
        return False, None

    # Compute the final score
    prediction_rankings = [
        list(
            st.session_state.form_results["final_evaluation"]["workout"][
                "predicted_ranks"
            ].values()
        )
    ]
    y0_rankings = [
        list(
            st.session_state.form_results["final_evaluation"]["workout"][
                "y0_ranks"
            ].values()
        )
    ]
    p_workout_wins = pairwise_win_rate(prediction_rankings, y0_rankings)
    other_wins, total = likert_to_win_rate(
        [
            st.session_state.form_results["final_evaluation"]["goals_preference"],
            st.session_state.form_results["final_evaluation"]["schedule_preference"],
            st.session_state.form_results["final_evaluation"]["equipment_preference"],
        ],
        return_total=True,
    )
    p_wins = (p_workout_wins + other_wins) / (total + 1)
    st.session_state.form_results["final_evaluation"]["score"] = p_wins

    return True, None


def rank_workouts(
    *, final_prediction: str, y0: Optional[str], db, num_items_per_comparison: int = 5
):
    """
    Rank workouts from both plans using a carousel.
    Returns True when ranking is complete.
    """
    predicted = parse_workout_plan(final_prediction, db, leave_invalid=True)
    parsed_y0 = (
        parse_workout_plan(y0, db, leave_invalid=True) if y0 is not None else None
    )

    done = "workout" in st.session_state.form_results["final_evaluation"]
    if not done:
        # Extract all workouts from both plans
        # A workout is a (day, time_of_day, exercises) tuple
        predicted_workouts = []
        for day in DAYS_OF_THE_WEEK:
            if day not in predicted:
                continue
            for time_of_day in TIMES_OF_DAY:
                if (
                    time_of_day not in predicted[day]
                    or predicted[day][time_of_day] is None
                ):
                    continue
                exercises = predicted[day][time_of_day]
                if exercises:  # Only include non-empty workouts
                    predicted_workouts.append(
                        {
                            "day": day,
                            "time_of_day": time_of_day,
                            "exercises": exercises,
                            "plan": "predicted",
                        }
                    )

        y0_workouts = []
        if parsed_y0 is not None:
            for day in DAYS_OF_THE_WEEK:
                if day not in parsed_y0:
                    continue
                for time_of_day in TIMES_OF_DAY:
                    if (
                        time_of_day not in parsed_y0[day]
                        or parsed_y0[day][time_of_day] is None
                    ):
                        continue
                    exercises = parsed_y0[day][time_of_day]
                    if exercises:  # Only include non-empty workouts
                        y0_workouts.append(
                            {
                                "day": day,
                                "time_of_day": time_of_day,
                                "exercises": exercises,
                                "plan": "y0",
                            }
                        )

        _render_carousel(
            predicted_workouts,
            y0_workouts,
            name="workout",
            num_items_per_comparison=num_items_per_comparison,
        )
        return False

    return True


@st.fragment
def _render_carousel(
    predicted: List[Dict[str, Any]],
    y0: List[Dict[str, Any]],
    name: str = "workout",
    md_fn: Callable = None,
    filter_fn: Callable[[Dict[str, Any]], bool] = None,
    num_items_per_comparison: int = 5,
):
    """
    Args:
        predicted: list of predicted workouts (dicts with day, time_of_day, exercises, plan)
        y0: list of y0 workouts (dicts with day, time_of_day, exercises, plan)
        name: name of the thing being ranked. Used for saving to session state.
        md_fn: function to render the workout (optional, defaults to workout rendering)
        filter_fn: function to filter the workouts
        num_items_per_comparison: number of workouts to show

    Adds to session state:
        - ranking: a dict mapping a rank (0-indexed) to a workout identifier
        - y0_ranks: a dict mapping workout identifier to a rank
        - predicted_ranks: a dict mapping workout identifier to a rank
    """

    # Helper function to create workout identifier
    def get_workout_id(workout):
        # Create identifier based on workout content (exercises)
        exercise_ids = []
        for ex in workout["exercises"]:
            if not ex.get("invalid", False):
                exercise_ids.append(
                    f"{ex['exercise_name']} - {ex.get('variation_name', 'default')}"
                )
        # Sort to make identifier stable regardless of exercise order
        exercise_ids.sort()
        return f"{workout['day']} {workout['time_of_day']}: {', '.join(exercise_ids)}"

    predicted = [
        p for p in predicted if p is not None and (filter_fn is None or filter_fn(p))
    ]
    y0 = [p for p in y0 if p is not None and (filter_fn is None or filter_fn(p))]

    if len(predicted) == 0:
        # set difference is the entire y0, and y0 auto-wins
        dummy_rank = {i: get_workout_id(y) for i, y in enumerate(y0)}
        st.session_state.form_results["final_evaluation"][name] = {
            "ranking": dummy_rank,
            "y0_ranks": {v: k for k, v in dummy_rank.items()},
            "predicted_ranks": {},
        }
        st.rerun()

    # find the set difference based on workout content
    predicted_ids = set([get_workout_id(p) for p in predicted])
    y0_ids = set([get_workout_id(y) for y in y0])
    diff_ids = (predicted_ids - y0_ids).union(y0_ids - predicted_ids)
    if not diff_ids:
        # don't render anything
        return

    predicted_options = [p for p in predicted if get_workout_id(p) in diff_ids]
    y0_options = [p for p in y0 if get_workout_id(p) in diff_ids]
    if (
        num_items_per_comparison is not None
        and len(diff_ids) > num_items_per_comparison
    ):
        # try to get a roughly balanced set of options
        if len(predicted_options) < num_items_per_comparison / 2:
            options = (
                predicted_options
                + y0_options[: num_items_per_comparison - len(predicted_options)]
            )
        elif len(y0_options) < num_items_per_comparison / 2:
            options = (
                predicted_options[: num_items_per_comparison - len(y0_options)]
                + y0_options
            )
        else:
            options = (
                predicted_options[
                    : num_items_per_comparison // 2 + num_items_per_comparison % 2
                ]
                + y0_options[: num_items_per_comparison // 2]
            )
    else:
        options = predicted_options + y0_options

    # Stabilize the options order across reruns within this fragment
    state_key = f"workout_options_order_{name}"
    if state_key not in st.session_state:
        # Store stable order as indices into the current options list
        order = list(range(len(options)))
        random.shuffle(order)
        st.session_state[state_key] = order
    order = st.session_state[state_key]
    # Reorder options according to stored order, truncating/expanding safely
    if len(order) != len(options):
        order = list(range(len(options)))
        st.session_state[state_key] = order
    options = [options[i] for i in order]

    # display the carousel
    from evaluation.app.components import carousel

    def display_fn(i):
        workout = options[i]

        # Generate and display workout name
        workout_name = _generate_workout_name(workout["exercises"])
        st.markdown(f"### {workout_name}")

        # Render workout summary table
        _render_workout_summary_table(workout["exercises"])
        st.markdown("#### Exercises in this workout")

        # Render all exercises in the workout
        for j, exercise in enumerate(workout["exercises"]):
            exercise_name = exercise.get("exercise_name", "Unknown")
            variation_name = exercise.get("variation_name", "default")
            with st.expander(f"{j + 1}. {exercise_name} - {variation_name}"):
                st.markdown(
                    _render_exercise_details(j, exercise), unsafe_allow_html=True
                )

    st.markdown("### Review the assistant's recommendations")
    st.markdown(f"The assistant has recommended {len(options)} {name}s for you.")
    # Increased height for larger workout cards
    carousel([lambda i=i: display_fn(i) for i in range(len(options))], height=700)

    with st.form(key=f"ranking_form_{name}"):
        rank = st.multiselect(
            f"Rank the {name}s above from MOST to LEAST preferred.",
            [i for i in range(len(options))],
            default=[],
            format_func=lambda x: f"Workout {x + 1}: {_generate_workout_name(options[x]['exercises'])}",
        )
        submit = st.form_submit_button("Submit", type="primary")
        if submit:
            if len(rank) != len(options):
                st.error("Please rank all options")
                return
            ranking = {i: get_workout_id(options[i]) for i in rank}
            st.session_state.form_results["final_evaluation"][name] = {
                "ranking": ranking,
                "y0_ranks": {v: k for k, v in ranking.items() if v in y0_ids},
                "predicted_ranks": {
                    v: k for k, v in ranking.items() if v in predicted_ids
                },
            }
            st.rerun()


def _generate_workout_name(exercises: List[Dict[str, Any]]) -> str:
    """Generate a descriptive name for the workout based on duration, target muscle groups, and exercise classification."""
    duration = _calculate_workout_duration(exercises)
    valid_exercises = [e for e in exercises if not e.get("invalid", False)]

    ABBREVIATIONS = {
        "Abdominal": "Ab",
        "Quadricep": "Quad",
        "Postural": "Posture-focused",
        "Unsorted*": "",
        "Animal Flow": "",
        "Grinds": "",
    }
    p = inflect.engine()

    # Collect unique target muscle groups
    target_muscles = set()
    for exercise in valid_exercises:
        if exercise.get("target_muscle_group"):
            singular_group = p.singular_noun(exercise["target_muscle_group"])
            pretty_group = ABBREVIATIONS.get(singular_group, singular_group)
            target_muscles.add(pretty_group)

    # Collect unique primary exercise classifications
    classifications = set()
    for exercise in valid_exercises:
        if exercise.get("primary_exercise_classification"):
            classifications.add(
                ABBREVIATIONS.get(
                    exercise["primary_exercise_classification"],
                    exercise["primary_exercise_classification"],
                )
            )

    # Format target areas
    if not target_muscles:
        target_str = "total body"
    elif len(target_muscles) == 1:
        target_str = list(target_muscles)[0]
    else:
        # Sort for consistency and join with " & "
        try:
            sorted_muscles = sorted([m.lower() for m in target_muscles if isinstance(m, str)])
            target_str = " & ".join(sorted_muscles)
        except:
            target_str = "total body"

    # Format classification
    if classifications:
        classification_str = " & ".join(sorted(classifications)).lower()
        return f"{duration} minute {target_str}-focused {classification_str} workout"
    else:
        return f"{duration} minute {target_str}-focused workout"


def _render_workout_summary_table(exercises: List[Dict[str, Any]]) -> None:
    """Render a summary table with key facts about the workout."""
    # Calculate totals
    total_time = _calculate_workout_duration(exercises)
    valid_exercises = [e for e in exercises if not e.get("invalid", False)]
    exercise_count = len(valid_exercises)

    # Collect unique target muscle groups
    target_muscles = set()
    for exercise in valid_exercises:
        if exercise.get("target_muscle_group"):
            target_muscles.add(exercise["target_muscle_group"])

    # Collect unique equipment
    equipment = set()
    for exercise in valid_exercises:
        if exercise.get("primary_equipment"):
            equipment.add(exercise["primary_equipment"])

    # Find max difficulty
    DIFFICULTY_ORDER = [
        "Novice",
        "Beginner",
        "Intermediate",
        "Advanced",
        "Expert",
        "Master",
        "Grand Master",
        "Legendary",
    ]
    difficulty_levels = []
    for exercise in valid_exercises:
        if exercise.get("difficulty_level"):
            difficulty_levels.append(exercise["difficulty_level"])

    max_difficulty = "N/A"
    if difficulty_levels:
        # Find the maximum difficulty by comparing indices in the ordered list
        max_difficulty = max(
            difficulty_levels,
            key=lambda d: DIFFICULTY_ORDER.index(d) if d in DIFFICULTY_ORDER else -1,
        )

    # Render as markdown table
    target_muscles_str = ", ".join(sorted(target_muscles)) if target_muscles else "N/A"
    equipment_str = ", ".join(sorted(equipment)) if equipment else "N/A"

    table_data = pd.DataFrame.from_dict(
        {
            "Total Time": total_time,
            "Number of Exercises": exercise_count,
            "Max Difficulty": max_difficulty,
            "Target Muscle Groups": target_muscles_str,
            "Equipment Needed": equipment_str,
        },
        orient="index",
        columns=["Value"],
    )
    st.table(table_data)


def _exercise_details(exercise: Dict[str, Any]) -> str:
    """Render exercise details as markdown with better spacing."""
    lines = []

    # Basic information - each on its own line
    if exercise.get("difficulty_level"):
        lines.append(f"**Difficulty:** {exercise['difficulty_level']}")
    if exercise.get("target_muscle_group"):
        lines.append(f"**Target:** {exercise['target_muscle_group']}")
    if exercise.get("prime_mover_muscle"):
        lines.append(f"**Prime mover:** {exercise['prime_mover_muscle']}")
    if exercise.get("primary_exercise_classification"):
        lines.append(f"**Type:** {exercise['primary_exercise_classification']}")
    if exercise.get("primary_equipment"):
        lines.append(f"**Equipment:** {exercise['primary_equipment']}")

    # Add spacing before exercise details
    if lines and (
        exercise.get("num_sets")
        or exercise.get("time_per_set")
        or exercise.get("num_reps_per_set")
    ):
        lines.append("")

    # Exercise details - each on its own line
    if exercise.get("num_sets"):
        lines.append(f"**Sets:** {exercise['num_sets']}")
    if exercise.get("time_per_set"):
        lines.append(f"**Time per set:** {exercise['time_per_set']}s")
    if exercise.get("num_reps_per_set"):
        lines.append(f"**Reps:** {exercise['num_reps_per_set']}")

    # YouTube link (if available)
    youtube_url = exercise.get("URL")
    if youtube_url and youtube_url != "nan":
        if lines:
            lines.append("")
        lines.append(f"Youtube video: [[link]]({youtube_url})")

    return "\n".join(lines)


def render_comparison(*, final_prediction: str, y0: Optional[str], db):
    """
    Render comparison view and collect Likert-scale preferences.
    Returns True when comparison is complete.
    """
    parsed_pred = parse_workout_plan(final_prediction, db, leave_invalid=True)
    parsed_y0 = (
        parse_workout_plan(y0, db, leave_invalid=True) if y0 is not None else None
    )

    # if both are invalid, just return True
    if parsed_pred is None and parsed_y0 is None:
        return True

    # Randomize which plan goes on which side (stable across reruns)
    side_key = "workout_comparison_side_assignment"
    if side_key not in st.session_state:
        # Randomly assign: True means pred on left (Plan A), False means y0 on left (Plan A)
        st.session_state[side_key] = random.choice([True, False])
    pred_on_left = st.session_state[side_key]

    # Assign plans to left/right based on randomization
    plan_a = parsed_pred if pred_on_left else parsed_y0
    plan_b = parsed_y0 if pred_on_left else parsed_pred

    with st.container(border=True):
        st.markdown("## Compare these workout plans")
        output_to_streamlit_comparison(
            plan_a,
            plan_b,
            db,
            valid1=None,
            valid2=None,
            metadata1=None,
            metadata2=None,
        )

    st.divider()

    # Likert questionnaire
    with st.form(key="workout_planning_comparison_form"):
        goals_preference = st.radio(
            "Compare how well plans A and B align with your fitness goals.",
            options=["-"] + COMPARISON_LIKERT,
        )
        schedule_preference = st.radio(
            "Compare how well plans A and B fit your schedule.",
            options=["-"] + COMPARISON_LIKERT,
        )
        equipment_preference = st.radio(
            "Compare how well plans A and B match your equipment availability.",
            options=["-"] + COMPARISON_LIKERT,
        )

        submit = st.form_submit_button("Submit", type="primary")
        if submit:
            if any(
                v is None or v == "-"
                for v in [
                    goals_preference,
                    schedule_preference,
                    equipment_preference,
                ]
            ):
                st.error("Please fill out all fields")
                return False

            st.session_state.form_results["final_evaluation"].update(
                {
                    "goals_preference": goals_preference,
                    "schedule_preference": schedule_preference,
                    "equipment_preference": equipment_preference,
                }
            )
            return True

    return False


def render_workout_plan_streamlit(plan: Any) -> None:
    """
    Render a parsed workout plan using Streamlit components.
    """
    unique_id = str(uuid.uuid4())[:8]
    _render_workout_plan_streamlit(plan, unique_id)


def output_to_streamlit_comparison(
    parsed1: Any,
    parsed2: Any,
    db: Any,
    valid1: bool,
    valid2: bool,
    metadata1: Any,
    metadata2: Any,
) -> None:
    """
    Render a comparison of two parsed workout plans using Streamlit.
    """
    unique_id = str(uuid.uuid4())[:8]
    a_valid, a_metadata = valid1, metadata1
    b_valid, b_metadata = valid2, metadata2

    tab1, tab2 = st.tabs(["Plan A", "Plan B"])

    with tab1:
        if a_valid is not None:
            if a_valid:
                st.markdown(":small[:green[:material/check: Plan A is valid]]")
            else:
                st.markdown(":small[:red[:material/close: Plan A invalid]]\n\n")
                constraints_md = "\n\n".join(
                    [
                        f":small[:red[- {constraint}]]"
                        for constraint in (a_metadata or {}).get(
                            "violated_constraints", []
                        )
                        if constraint is not None
                    ]
                )
                if constraints_md:
                    st.markdown(constraints_md)
        _render_workout_plan_streamlit(parsed1, unique_id, show_expanders=False)

    with tab2:
        if b_valid is not None:
            if b_valid:
                st.markdown(":small[:green[:material/check: Plan B is valid]]")
            else:
                st.markdown(":small[:red[:material/close: Plan B invalid]]\n\n")
                constraints_md = "\n\n".join(
                    [
                        f":small[:red[- {constraint}]]"
                        for constraint in (b_metadata or {}).get(
                            "violated_constraints", []
                        )
                        if constraint is not None
                    ]
                )
                if constraints_md:
                    st.markdown(constraints_md)
        _render_workout_plan_streamlit(parsed2, unique_id, show_expanders=False)


def _render_workout_plan_streamlit(
    plan: Any, unique_id: str, show_expanders: bool = True
) -> None:
    if not plan:
        st.markdown("*No workout plan data available*")
        return

    # Calendar overview
    st.markdown("\n".join(_render_calendar_table(plan)), unsafe_allow_html=True)

    # Show plan summary: equipment, max difficulty, target muscle groups, and classifications
    all_equipment = set()
    all_target_muscles = set()
    all_classifications = set()
    all_difficulty_levels = []
    DIFFICULTY_ORDER = [
        "Novice",
        "Beginner",
        "Intermediate",
        "Advanced",
        "Expert",
        "Master",
        "Grand Master",
        "Legendary",
    ]

    for day in DAYS_OF_THE_WEEK:
        if day not in plan:
            continue
        for time_of_day in TIMES_OF_DAY:
            if time_of_day not in plan[day] or plan[day][time_of_day] is None:
                continue
            exercises = plan[day][time_of_day]
            if not exercises:
                continue
            for exercise in exercises:
                if exercise.get("invalid", False):
                    continue
                if exercise.get("primary_equipment"):
                    all_equipment.add(exercise["primary_equipment"])
                if exercise.get("target_muscle_group"):
                    all_target_muscles.add(exercise["target_muscle_group"])
                if exercise.get("primary_exercise_classification"):
                    all_classifications.add(exercise["primary_exercise_classification"])
                if exercise.get("difficulty_level"):
                    all_difficulty_levels.append(exercise["difficulty_level"])

    # Calculate max difficulty
    max_difficulty = "N/A"
    if all_difficulty_levels:
        max_difficulty = max(
            all_difficulty_levels,
            key=lambda d: DIFFICULTY_ORDER.index(d) if d in DIFFICULTY_ORDER else -1,
        )

    # Display summary information
    summary_items = []
    if all_equipment:
        equipment_str = ", ".join(sorted(all_equipment))
        summary_items.append(f"**Equipment Required:** {equipment_str}")
    if max_difficulty != "N/A":
        summary_items.append(f"**Max Difficulty:** {max_difficulty}")
    if all_target_muscles:
        target_muscles_str = ", ".join(sorted(all_target_muscles))
        summary_items.append(f"**Target Muscle Groups:** {target_muscles_str}")
    if all_classifications:
        classifications_str = ", ".join(sorted(all_classifications))
        summary_items.append(f"**Exercise Classifications:** {classifications_str}")

    if summary_items:
        st.markdown("\n\n".join(summary_items))

    if not show_expanders:
        return

    # Detailed workouts: one expander per workout slot
    st.markdown("### Workout Details")
    slots = _get_workout_slots(plan)
    if not slots:
        st.markdown("*No workouts planned*")
    else:
        for i, (day, time_of_day) in enumerate(slots):
            title = f"💪 {day.capitalize()} {time_of_day} | {_generate_workout_name(plan[day][time_of_day])}"
            with st.expander(title, expanded=False):
                _render_workout_summary_table(plan[day][time_of_day])
                st.markdown(
                    _render_workout_details(i, day, time_of_day, plan),
                    unsafe_allow_html=True,
                )


# ===== Markdown rendering helpers (moved from data.py) =====


def workout_plan_to_markdown(workout_plan):
    if not workout_plan:
        return "*No workout plan data available*"
    markdown_lines = []
    markdown_lines += _render_calendar_table(workout_plan)
    workout_slots = _get_workout_slots(workout_plan)
    if workout_slots:
        markdown_lines.append("### 💪 Workout Details")
        workout_lines = []
        for i, (day, time_of_day) in enumerate(workout_slots):
            workout_lines.append(
                _render_workout_details(i, day, time_of_day, workout_plan) + "\n\n"
            )
        markdown_lines.append("\n<hr>\n".join(workout_lines))
    else:
        markdown_lines.append("*No workouts planned*")
    return "\n".join(markdown_lines)


def _render_calendar_table(workout_plan):
    markdown_lines = []
    markdown_lines.append("### 📆 Workout Calendar")
    header_row = (
        "| | "
        + " | ".join([f"**{day[:3].upper()}**" for day in DAYS_OF_THE_WEEK])
        + " |"
    )
    separator_row = "|" + "|".join(["------"] * (len(DAYS_OF_THE_WEEK) + 1)) + "|"
    markdown_lines.append(header_row)
    markdown_lines.append(separator_row)

    for time_of_day in TIMES_OF_DAY:
        row_cells = [f"**{re.search(r'\((.+)\)', time_of_day).group(1)}**"]
        for day in DAYS_OF_THE_WEEK:
            if day not in workout_plan:
                row_cells.append("")
                continue
            day_plan = workout_plan[day]
            if time_of_day not in day_plan or day_plan[time_of_day] is None:
                row_cells.append("")
                continue
            exercises = day_plan[time_of_day]
            if not exercises:
                row_cells.append("")
                continue
            num_invalid_exercises = len(
                [e for e in exercises if e.get("invalid", False)]
            )
            workout_name = _generate_workout_name(exercises)
            cell_content = (
                f"{workout_name} (:red-background[:material/error: {num_invalid_exercises} invalid exercise{'s' if num_invalid_exercises != 1 else ''}])"
                if num_invalid_exercises > 0
                else workout_name
            )
            row_cells.append(cell_content)
        row = "| " + " | ".join(row_cells) + " |"
        markdown_lines.append(row)

    total_time_row = [""]

    for day in DAYS_OF_THE_WEEK:
        if day not in workout_plan:
            total_time_row.append("")
            continue
        day_plan = workout_plan[day]
        day_total_time = 0
        day_has_workout = False
        for time_of_day in TIMES_OF_DAY:
            if time_of_day in day_plan and day_plan[time_of_day]:
                exercises = day_plan[time_of_day]
                if exercises:
                    day_total_time += _calculate_workout_duration(exercises)
                    day_has_workout = True
        if day_has_workout:
            total_time_row.append(f"**Daily total: {day_total_time:.0f} min**")
        else:
            total_time_row.append("")
    total_row = "| " + " | ".join(total_time_row) + " |"
    markdown_lines.append(total_row)
    return markdown_lines


def _render_workout_details(i, day, time_of_day, workout_plan):
    lines = []
    lines.append("")
    exercises = workout_plan[day][time_of_day]

    for j, exercise in enumerate(exercises):
        lines.append(_render_exercise_details(j, exercise))
        lines.append("")
    return "\n".join(lines)


def _render_exercise_details(i, exercise):
    if exercise.get("invalid", False):
        return "\n".join(
            [
                f"<b>{i + 1}. {exercise['exercise_name']}</b>",
                "",
                ":red-background[:material/error: The assistant listed an invalid variation that is not in the database.]",
            ]
        )
    lines = [
        f"<b>{i + 1}. {exercise['exercise_name']} -- {exercise['variation_name']}</b>"
    ]
    lines.append("")
    basic_info = []
    if exercise.get("difficulty_level"):
        basic_info.append(f"**Difficulty:** {exercise['difficulty_level']}")
    if exercise.get("target_muscle_group"):
        basic_info.append(f"**Target:** {exercise['target_muscle_group']}")
    if exercise.get("prime_mover_muscle"):
        basic_info.append(f"**Prime mover:** {exercise['prime_mover_muscle']}")
    if exercise.get("secondary_muscle"):
        basic_info.append(f"**Secondary:** {exercise['secondary_muscle']}")
    if exercise.get("tertiary_muscle"):
        basic_info.append(f"**Tertiary:** {exercise['tertiary_muscle']}")
    if exercise.get("primary_exercise_classification"):
        basic_info.append(f"**Type:** {exercise['primary_exercise_classification']}")

    exercise_details = []
    if exercise.get("num_sets"):
        exercise_details.append(f"**Sets:** {exercise['num_sets']}")
    time_or_reps = exercise.get("time_or_reps", "")
    if time_or_reps == "time" and exercise.get("time_per_set"):
        exercise_details.append(f"**Time per set:** {exercise['time_per_set']}s")
    elif exercise.get("num_reps_per_set"):
        exercise_details.append(f"**Reps:** {exercise['num_reps_per_set']}")
    if exercise.get("total_time_seconds"):
        total_minutes = exercise.get("total_time_seconds", 0) / 60
        exercise_details.append(f"**Total time:** {total_minutes:.0f} min")

    youtube_url = exercise.get("URL")
    if youtube_url and youtube_url != "nan":
        lines.append("")
        lines.append(_add_video_link(youtube_url, exercise["exercise_name"], width=300))
        lines.append("")
        lines.append(f"Youtube video demonstration: [[link]]({youtube_url})")
        lines.append("")

    block = []
    if exercise.get("primary_equipment"):
        block.append("**Equipment:** " + exercise.get("primary_equipment", ""))
    if basic_info:
        block.append(" | ".join(basic_info))
    if exercise_details:
        block.append(" | ".join(exercise_details))
    lines.append("\n\n".join(block))
    return "\n".join(lines)


def _calculate_workout_duration(exercises):
    total_seconds = 0
    for exercise in exercises:
        if exercise.get("invalid", False):
            continue
        total_seconds += exercise.get("total_time_seconds")
    return math.ceil(total_seconds / 60)


def _add_video_link(url, exercise_name, width=150):
    return f'<a href="{url}">{_get_exercise_image(url, exercise_name, width)}</a>'


def _get_exercise_image(url, exercise_name, width=150):
    if url is None:
        return ""
    video_id = url.split("/")[-1]
    return f'<img src="https://img.youtube.com/vi/{video_id}/0.jpg" alt="{exercise_name}" style="width: {width}px; height: auto;">'


def _get_workout_days_list(workout_plan):
    workout_days = []

    for day in DAYS_OF_THE_WEEK:
        if day not in workout_plan:
            continue
        day_plan = workout_plan[day]
        for time_of_day in TIMES_OF_DAY:
            if time_of_day not in day_plan or day_plan[time_of_day] is None:
                continue
            exercises = day_plan[time_of_day]
            if not exercises:
                continue
            exercise_count = len([e for e in exercises if not e.get("invalid", False)])
            total_duration = _calculate_workout_duration(exercises)
            workout_days.append(
                f"{day.capitalize()} {time_of_day}: ({total_duration:.0f}min, {exercise_count} exercises)"
            )
    return workout_days


def _get_workout_slots(workout_plan):
    workout_slots = []

    for day in DAYS_OF_THE_WEEK:
        if day not in workout_plan:
            continue
        day_plan = workout_plan[day]
        for time_of_day in TIMES_OF_DAY:
            if time_of_day not in day_plan or day_plan[time_of_day] is None:
                continue
            exercises = day_plan[time_of_day]
            if not exercises:
                continue
            workout_slots.append((day, time_of_day))
    return workout_slots


def render_exercise_mentions(exercise_names: List[str], db: Any) -> None:
    """
    Render a section showing mentioned exercises with their details.
    Uses a grid of buttons that launch dialogs, similar to meal planning recipes.
    Shows all variations of each exercise in a single dialog.
    """
    if not exercise_names:
        return

    st.markdown('<div id="mentioned-exercises"></div>', unsafe_allow_html=True)
    st.markdown(
        "Click on an exercise to view all its variations, including difficulty level, target muscles, equipment needed, and demonstration videos."
    )

    # Get unique exercises (first occurrence only as requested)
    seen_exercises = set()
    unique_exercises = []
    for exercise_name in exercise_names:
        if exercise_name not in seen_exercises:
            seen_exercises.add(exercise_name)
            unique_exercises.append(exercise_name)

    if not unique_exercises:
        return

    # Create a grid of buttons (3 columns)
    cols = st.columns(3)
    for i, exercise_name in enumerate(unique_exercises):
        with cols[i % 3]:
            try:
                # Get all exercises with this name
                all_exercises = db.get_all_exercises_by_name(exercise_name)
                if all_exercises:
                    # Create dialog for all exercise variations
                    @st.dialog(
                        f"{exercise_name} - {len(all_exercises)} Variations",
                        width="large",
                    )
                    def _show_exercise_dialog(
                        exercises: List[dict], exercise_name: str
                    ) -> None:
                        tabs = st.tabs(
                            [f"{exercise['variation_name']}" for exercise in exercises]
                        )
                        for i, exercise in enumerate(exercises):
                            with tabs[i]:
                                st.markdown(
                                    _render_exercise_details(i, exercise),
                                    unsafe_allow_html=True,
                                )

                    st.button(
                        f":material/exercise: {exercise_name} ({len(all_exercises)} variations)",
                        on_click=_show_exercise_dialog,
                        args=(all_exercises, exercise_name),
                        key=f"exercise_{exercise_name}_{uuid.uuid4().hex[:8]}",
                        use_container_width=True,
                    )
                else:
                    # Exercise not found in database - show disabled button
                    st.button(
                        f":material/error: {exercise_name} (not found)",
                        disabled=True,
                        key=f"exercise_invalid_{exercise_name}_{uuid.uuid4().hex[:8]}",
                        use_container_width=True,
                    )
            except Exception:
                # Error finding exercise - show disabled button
                st.button(
                    f":material/error: {exercise_name} (error)",
                    disabled=True,
                    key=f"exercise_error_{exercise_name}_{uuid.uuid4().hex[:8]}",
                    use_container_width=True,
                )
