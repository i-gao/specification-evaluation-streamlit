from typing import List, Dict, Any, Optional
import streamlit as st
import uuid
import re
from data.workout_planning.db import DAYS_OF_THE_WEEK, TIMES_OF_DAY
from data.workout_planning.parser import parse_workout_plan
from evaluation.qualitative_eval import COMPARISON_LIKERT
from data.reward import likert_to_win_rate


def render_eval(*, final_prediction: str, y0: Optional[str], db):
    """
    Render evaluation UI: show A vs B comparison and collect Likert-scale preferences.
    Returns (completed, feedback_dict_or_none).
    """
    # Comparison view
    parsed_pred = parse_workout_plan(final_prediction, db, leave_invalid=True)
    parsed_y0 = (
        parse_workout_plan(y0, db, leave_invalid=True) if y0 is not None else None
    )

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
            responses = [
                difficulty_preference,
                goals_preference,
                schedule_preference,
                equipment_preference,
                injury_preference,
            ]
            if any(v is None or v == "-" for v in responses):
                st.error("Please fill out all fields")
                return False, None

            st.session_state.form_results["final_evaluation"].update(
                {
                    "difficulty_preference": difficulty_preference,
                    "goals_preference": goals_preference,
                    "schedule_preference": schedule_preference,
                    "equipment_preference": equipment_preference,
                    "injury_preference": injury_preference,
                    "score": likert_to_win_rate(responses),
                }
            )
            return True, None

    return False, None

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
