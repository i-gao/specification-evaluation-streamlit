from typing import List, Dict, Any, Optional, Callable
import streamlit as st
import uuid
import re
import random
from data.workout_planning.db import DAYS_OF_THE_WEEK, TIMES_OF_DAY
from data.workout_planning.parser import parse_workout_plan
from evaluation.qualitative_eval import COMPARISON_LIKERT
from data.reward import likert_to_win_rate, pairwise_win_rate


def render_eval(*, final_prediction: str, y0: Optional[str], db, num_items_per_comparison: int = 5, **kwargs):
    """
    Render evaluation UI: first rank exercises, then show A vs B comparison and collect Likert-scale preferences.
    Returns (completed, feedback_dict_or_none).
    """
    ranking_done = rank_exercises(final_prediction=final_prediction, y0=y0, db=db, num_items_per_comparison=num_items_per_comparison)
    if not ranking_done:
        return False, None

    comparison_done = render_comparison(final_prediction=final_prediction, y0=y0, db=db)
    if not comparison_done:
        return False, None

    # Compute the final score
    prediction_rankings = [
        list(
            st.session_state.form_results["final_evaluation"]["exercise"][
                "predicted_ranks"
            ].values()
        )
    ]
    y0_rankings = [
        list(
            st.session_state.form_results["final_evaluation"]["exercise"][
                "y0_ranks"
            ].values()
        )
    ]
    p_exercise_wins = pairwise_win_rate(prediction_rankings, y0_rankings)
    other_wins, total = likert_to_win_rate(
        [
            st.session_state.form_results["final_evaluation"]["goals_preference"],
            st.session_state.form_results["final_evaluation"]["schedule_preference"],
            st.session_state.form_results["final_evaluation"]["equipment_preference"],
            st.session_state.form_results["final_evaluation"]["injury_preference"],
            st.session_state.form_results["final_evaluation"]["difficulty_preference"],
        ],
        return_total=True,
    )
    p_wins = (p_exercise_wins + other_wins) / (total + 1)
    st.session_state.form_results["final_evaluation"]["score"] = p_wins

    return True, None


def rank_exercises(*, final_prediction: str, y0: Optional[str], db, num_items_per_comparison: int = 5):
    """
    Rank exercises from both plans using a carousel.
    Returns True when ranking is complete.
    """
    predicted = parse_workout_plan(final_prediction, db, leave_invalid=True)
    parsed_y0 = (
        parse_workout_plan(y0, db, leave_invalid=True) if y0 is not None else None
    )

    done = "exercise" in st.session_state.form_results["final_evaluation"]
    if not done:
        # Extract all exercises from both plans
        predicted_exercises = []
        for day in DAYS_OF_THE_WEEK:
            if day not in predicted:
                continue
            for time_of_day in TIMES_OF_DAY:
                if time_of_day not in predicted[day] or predicted[day][time_of_day] is None:
                    continue
                for exercise in predicted[day][time_of_day]:
                    if not exercise.get("invalid", False):
                        predicted_exercises.append(exercise)

        y0_exercises = []
        if parsed_y0 is not None:
            for day in DAYS_OF_THE_WEEK:
                if day not in parsed_y0:
                    continue
                for time_of_day in TIMES_OF_DAY:
                    if time_of_day not in parsed_y0[day] or parsed_y0[day][time_of_day] is None:
                        continue
                    for exercise in parsed_y0[day][time_of_day]:
                        if not exercise.get("invalid", False):
                            y0_exercises.append(exercise)

        _render_carousel(
            predicted_exercises,
            y0_exercises,
            num_items_per_comparison=num_items_per_comparison,
        )
        return False

    return True


@st.fragment
def _render_carousel(
    predicted: List[Dict[str, Any]],
    y0: List[Dict[str, Any]],
    name: str = "exercise",
    md_fn: Callable = None,
    filter_fn: Callable[[Dict[str, Any]], bool] = None,
    num_items_per_comparison: int = 5,
):
    """
    Args:
        predicted: list of predicted exercises
        y0: list of y0 exercises
        name: name of the thing being ranked. Used for saving to session state.
        md_fn: function to render the exercise
        filter_fn: function to filter the exercises
        num_items_per_comparison: number of exercises to show

    Adds to session state:
        - ranking: a dict mapping a rank (0-indexed) to an exercise identifier
        - y0_ranks: a dict mapping exercise identifier to a rank
        - predicted_ranks: a dict mapping exercise identifier to a rank
    """
    if md_fn is None:
        def md_fn(exercise):
            return _exercise_details(exercise)

    predicted = [
        p for p in predicted if p is not None and (filter_fn is None or filter_fn(p))
    ]
    # Create unique identifiers for exercises (exercise_name + variation_name)
    predicted = list({
        f"{d['exercise_name']} - {d.get('variation_name', 'default')}": d 
        for d in predicted
    }.values())

    y0 = [p for p in y0 if p is not None and (filter_fn is None or filter_fn(p))]
    y0 = list({
        f"{d['exercise_name']} - {d.get('variation_name', 'default')}": d 
        for d in y0
    }.values())

    if len(predicted) == 0:
        # set difference is the entire y0, and y0 auto-wins
        dummy_rank = {i: f"{y['exercise_name']} - {y.get('variation_name', 'default')}" for i, y in enumerate(y0)}
        st.session_state.form_results["final_evaluation"][name] = {
            "ranking": dummy_rank,
            "y0_ranks": {v: k for k, v in dummy_rank.items()},
            "predicted_ranks": {},
        }
        st.rerun()

    # find the set difference
    predicted_names = set([f"{p['exercise_name']} - {p.get('variation_name', 'default')}" for p in predicted])
    y0_names = set([f"{p['exercise_name']} - {p.get('variation_name', 'default')}" for p in y0])
    diff_names = (predicted_names - y0_names).union(y0_names - predicted_names)
    if not diff_names:
        # don't render anything
        return

    predicted_options = [p for p in predicted if f"{p['exercise_name']} - {p.get('variation_name', 'default')}" in diff_names]
    y0_options = [p for p in y0 if f"{p['exercise_name']} - {p.get('variation_name', 'default')}" in diff_names]
    if num_items_per_comparison is not None and len(diff_names) > num_items_per_comparison:
        # try to get a roughly balanced set of options
        if len(predicted_options) < num_items_per_comparison / 2:
            options = predicted_options + y0_options[: num_items_per_comparison - len(predicted_options)]
        elif len(y0_options) < num_items_per_comparison / 2:
            options = predicted_options[: num_items_per_comparison - len(y0_options)] + y0_options
        else:
            options = (
                predicted_options[: num_items_per_comparison // 2 + num_items_per_comparison % 2]
                + y0_options[: num_items_per_comparison // 2]
            )
    else:
        options = predicted_options + y0_options

    # Stabilize the options order across reruns within this fragment
    state_key = f"options_order_{name}"
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
        exercise = options[i]
        st.markdown(f"### {exercise['exercise_name']}")
        if exercise.get('variation_name'):
            st.markdown(f"**Variation:** {exercise['variation_name']}")
        # Display exercise image if available (YouTube thumbnail)
        youtube_url = exercise.get("URL")
        if youtube_url and youtube_url != "nan":
            video_id = youtube_url.split("/")[-1]
            image_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
            st.image(image_url, width=400)
        st.markdown(
            md_fn(exercise),
            unsafe_allow_html=True,
        )

    st.markdown("### Review the assistant's recommendations")
    st.markdown(f"The assistant has recommended {len(options)} {name}s for you.")
    carousel([lambda i=i: display_fn(i) for i in range(len(options))], height=550)

    with st.form(key=f"ranking_form_{name}"):
        rank = st.multiselect(
            f"Rank the {name} above from MOST to LEAST preferred.",
            [i for i in range(len(options))],
            default=[],
            format_func=lambda x: f"{name.upper()} {x + 1}: {options[x]['exercise_name']} - {options[x].get('variation_name', 'default')}",
        )
        submit = st.form_submit_button("Submit", type="primary")
        if submit:
            if len(rank) != len(options):
                st.error("Please rank all options")
                return
            ranking = {i: f"{options[i]['exercise_name']} - {options[i].get('variation_name', 'default')}" for i in rank}
            st.session_state.form_results["final_evaluation"][name] = {
                "ranking": ranking,
                "y0_ranks": {v: k for k, v in ranking.items() if v in y0_names},
                "predicted_ranks": {
                    v: k for k, v in ranking.items() if v in predicted_names
                },
            }
            st.rerun()


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
    if lines and (exercise.get("num_sets") or exercise.get("time_per_set") or exercise.get("num_reps_per_set")):
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

    with st.container(border=True):
        st.markdown("## Compare these workout plans")
        output_to_streamlit_comparison(
            parsed_pred,
            parsed_y0,
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
        injury_preference = st.radio(
            "Compare how well plans A and B accommodate your injury/mobility constraints.",
            options=["-"] + COMPARISON_LIKERT,
        )
        difficulty_preference = st.radio(
            "Compare the difficulty levels of exercises in plans A and B. Which do you prefer?",
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
                    injury_preference,
                    difficulty_preference,
                ]
            ):
                st.error("Please fill out all fields")
                return False

            st.session_state.form_results["final_evaluation"].update(
                {
                    "goals_preference": goals_preference,
                    "schedule_preference": schedule_preference,
                    "equipment_preference": equipment_preference,
                    "injury_preference": injury_preference,
                    "difficulty_preference": difficulty_preference,
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
        _render_workout_plan_streamlit(parsed1, unique_id)

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
        _render_workout_plan_streamlit(parsed2, unique_id)


def _render_workout_plan_streamlit(plan: Any, unique_id: str) -> None:
    if not plan:
        st.markdown("*No workout plan data available*")
        return

    # Calendar overview
    st.markdown("\n".join(_render_calendar_table(plan)), unsafe_allow_html=True)

    with st.container(horizontal=False, gap="small"):
        # Detailed workouts: one expander per workout slot
        slots = _get_workout_slots(plan)
        if not slots:
            st.markdown("*No workouts planned*")
        else:
            for i, (day, time_of_day) in enumerate(slots):
                title = f"💪 {day.capitalize()} {time_of_day} workout ({_calculate_workout_duration(plan[day][time_of_day])} min)"
                with st.expander(title, expanded=False):
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
            exercise_count = len(exercises)
            num_invalid_exercises = len(
                [e for e in exercises if e.get("invalid", False)]
            )
            total_duration = _calculate_workout_duration(exercises)
            cell_content = (
                (
                    f"{total_duration:.0f} min workout (:red-background[:material/error: {exercise_count} exercises, {num_invalid_exercises} invalid])"
                )
                if num_invalid_exercises > 0
                else f"{total_duration:.0f} min workout ({exercise_count} exercises)"
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
                ":red-background[:material/error: This is an invalid exercise that is not in the database.]",
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
    return total_seconds // 60


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
