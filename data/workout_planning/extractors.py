# Extractor functions for workout planning constraints
# Each function takes a workout_plan as input and returns a tuple (value, detailed_message)
# where value is the value needed for the constraint and detailed_message describes the result

DAYS_OF_THE_WEEK = [
    "sunday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
]
TIMES_OF_DAY = [
    "early morning (6-9am)",
    "late morning (9-11am)",
    "midday (11am-1pm)",
    "afternoon (2-5pm)",
    "evening (5-10pm)",
]

"""
Assume a workout plan is a dictionary with the following structure:
{
    "sunday": {
        "early morning (6-9am)": [
            Exercise (pd.Series),
            ...
        ],
        ...
    },
    ...
}
"""

def all_valid(workout_plan):
    """
    Check if all exercises are valid.
    """
    for day in DAYS_OF_THE_WEEK:
        for time in TIMES_OF_DAY:
            if workout_plan[day][time] is not None:
                for exercise in workout_plan[day][time]:
                    if exercise.get("invalid", False):
                        return False, f"Exercise {exercise['exercise_name']} is invalid"
    return True, "All exercises are valid"


def exercise_attributes(workout_plan, day, time, column):
    """
    Extract exercise attributes for a specific day and time.

    Args:
        workout_plan: The workout plan dictionary
        day: Day of the week
        time: Time of day
        column: The exercise attribute column to extract

    Returns:
        (List, str): (List of attribute values for exercises at the specified day/time, or None if no exercises, detailed_message)
    """
    if workout_plan[day][time] is None:
        return None, f"No exercises scheduled for {day.capitalize()} {time}"

    attributes = [exercise[column] for exercise in workout_plan[day][time]]
    exercise_names = [exercise["exercise_name"] for exercise in workout_plan[day][time]]
    message = f"Exercises at {day.capitalize()} {time}: {', '.join(exercise_names)} with {column} values: {attributes}"
    return attributes, message


def exercise_has_attribute(workout_plan, day, time, column, values):
    """
    Check if any exercise at the specified day/time has the given attribute value.

    Args:
        workout_plan: The workout plan dictionary
        day: Day of the week
        time: Time of day
        column: The exercise attribute column to check
        values: List of acceptable values

    Returns:
        (bool, str): (True if any exercise has an attribute in the values list, False otherwise, None if no exercises, detailed_message)
    """
    if workout_plan[day][time] is None:
        return None, f"No exercises scheduled for {day.capitalize()} {time}"

    matching_exercises = []
    for exercise in workout_plan[day][time]:
        if exercise[column] in values:
            matching_exercises.append(exercise["exercise_name"])

    if matching_exercises:
        return (
            True,
            f"Found exercises with {column} in {values} at {day.capitalize()} {time}: {', '.join(matching_exercises)}",
        )
    else:
        exercise_names = [
            exercise["exercise_name"] for exercise in workout_plan[day][time]
        ]
        return (
            False,
            f"No exercises at {day.capitalize()} {time} have {column} in {values}. Available exercises: {', '.join(exercise_names)}",
        )


def exercise_avoids_attribute(workout_plan, day, time, column, values):
    """
    Check if any exercise at the specified day/time has an attribute in the avoided values.

    Args:
        workout_plan: The workout plan dictionary
        day: Day of the week
        time: Time of day
        column: The exercise attribute column to check
        values: List of values to avoid

    Returns:
        (bool, str): (True if any exercise has an attribute in the values list, False otherwise, None if no exercises, detailed_message)
    """
    if workout_plan[day][time] is None:
        return None, f"No exercises scheduled for {day.capitalize()} {time}"

    violating_exercises = []
    for exercise in workout_plan[day][time]:
        if exercise[column] in values:
            violating_exercises.append(
                f"{exercise['exercise_name']} ({exercise[column]})"
            )

    if violating_exercises:
        return (
            True,
            f"Found exercises with forbidden {column} values at {day.capitalize()} {time}: {', '.join(violating_exercises)}",
        )
    else:
        exercise_names = [
            exercise["exercise_name"] for exercise in workout_plan[day][time]
        ]
        return (
            False,
            f"All exercises at {day.capitalize()} {time} avoid forbidden {column} values {values}. Exercises: {', '.join(exercise_names)}",
        )


def workout_slot_empty(workout_plan, day, time):
    """
    Check if a workout slot is empty (no exercises scheduled).

    Args:
        workout_plan: The workout plan dictionary
        day: Day of the week
        time: Time of day

    Returns:
        (bool, str): (True if the slot is empty, False otherwise, detailed_message)
    """
    if workout_plan[day][time] is None:
        return True, f"No workout scheduled for {day.capitalize()} {time}"
    else:
        exercise_names = [
            exercise["exercise_name"] for exercise in workout_plan[day][time]
        ]
        return (
            False,
            f"Workout scheduled for {day.capitalize()} {time}: {', '.join(exercise_names)}",
        )


def workout_duration_under(workout_plan, day, time, max_duration_minutes):
    """
    Return False if the total workout duration for a day/time is not under the limit.

    Args:
        workout_plan: The workout plan dictionary
        day: Day of the week
        time: Time of day
        max_duration_minutes: Maximum duration in minutes

    Returns:
        (bool, str): (True if total duration <= max_duration_minutes, False otherwise, detailed_message)
    """
    if workout_plan[day][time] is None:
        return True, f"No workout scheduled for {day.capitalize()} {time}"

    total_time = sum(
        exercise["total_time_seconds"] / 60.0 for exercise in workout_plan[day][time]
    )
    exercise_details = [
        f"{exercise['exercise_name']} ({exercise['total_time_seconds']/60.0:.1f} min)"
        for exercise in workout_plan[day][time]
    ]

    if total_time <= max_duration_minutes:
        return (
            True,
            f"Workout duration at {day.capitalize()} {time}: {total_time:.1f} minutes ({', '.join(exercise_details)}) within limit of {max_duration_minutes} minutes",
        )
    else:
        return (
            False,
            f"Workout duration at {day.capitalize()} {time}: {total_time:.1f} minutes ({', '.join(exercise_details)}) exceeds limit of {max_duration_minutes} minutes",
        )


def workout_duration_over(workout_plan, day, time, min_workout_duration):
    """
    Return True if the total workout duration for a day/time is over the minimum.
    """
    if workout_plan[day][time] is None:
        return False, f"No workout scheduled for {day.capitalize()} {time}"

    total_time = sum(
        exercise["total_time_seconds"] / 60.0 for exercise in workout_plan[day][time]
    )
    exercise_details = [
        f"{exercise['exercise_name']} ({exercise['total_time_seconds']/60.0:.1f} min)"
        for exercise in workout_plan[day][time]
    ]

    if total_time >= min_workout_duration:
        return (
            True,
            f"Workout duration at {day.capitalize()} {time}: {total_time:.1f} minutes ({', '.join(exercise_details)}) exceeds limit of {min_workout_duration} minutes",
        )
    else:
        return (
            False,
            f"Workout duration at {day.capitalize()} {time}: {total_time:.1f} minutes ({', '.join(exercise_details)}) exceeds limit of {min_workout_duration} minutes",
        )


def min_workouts_satisfied(workout_plan, min_workouts):
    """
    Check if the minimum number of workouts is satisfied.

    Args:
        workout_plan: The workout plan dictionary
        min_workouts: Minimum number of workouts required

    Returns:
        (bool, str): (True if total workouts >= min_workouts, False otherwise, detailed_message)
    """
    total = sum(
        1 if workout_plan[day][time] is not None else 0
        for day in DAYS_OF_THE_WEEK
        for time in TIMES_OF_DAY
    )

    scheduled_workouts = []
    for day in DAYS_OF_THE_WEEK:
        for time in TIMES_OF_DAY:
            if workout_plan[day][time] is not None:
                exercise_names = [
                    exercise["exercise_name"] for exercise in workout_plan[day][time]
                ]
                scheduled_workouts.append(
                    f"{day.capitalize()} {time}: {', '.join(exercise_names)}"
                )

    if total >= min_workouts:
        return (
            True,
            f"Total workouts: {total} (minimum {min_workouts} required). Scheduled workouts: {'; '.join(scheduled_workouts)}",
        )
    else:
        return (
            False,
            f"Total workouts: {total} (minimum {min_workouts} required). Scheduled workouts: {'; '.join(scheduled_workouts)}",
        )


def max_workouts_per_day_satisfied(workout_plan, day, max_workouts):
    """
    Check if the maximum number of workouts per day is satisfied.

    Args:
        workout_plan: The workout plan dictionary
        day: Day of the week
        max_workouts: Maximum number of workouts allowed per day

    Returns:
        (bool, str): (True if workouts per day <= max_workouts, False otherwise, detailed_message)
    """
    workouts = sum(1 for time in TIMES_OF_DAY if workout_plan[day][time] is not None)

    scheduled_times = []
    for time in TIMES_OF_DAY:
        if workout_plan[day][time] is not None:
            exercise_names = [
                exercise["exercise_name"] for exercise in workout_plan[day][time]
            ]
            scheduled_times.append(f"{time}: {', '.join(exercise_names)}")

    if workouts <= max_workouts:
        return (
            True,
            f"Workouts on {day.capitalize()}: {workouts} (maximum {max_workouts} allowed). Scheduled: {'; '.join(scheduled_times)}",
        )
    else:
        return (
            False,
            f"Workouts on {day.capitalize()}: {workouts} (maximum {max_workouts} allowed). Scheduled: {'; '.join(scheduled_times)}",
        )


def min_rest_days_satisfied(workout_plan, min_rest_days=1):
    """
    Check if the minimum number of rest days is satisfied.

    Args:
        workout_plan: The workout plan dictionary
        min_rest_days: Minimum number of rest days required (default: 1)

    Returns:
        (bool, str): (True if rest days >= min_rest_days, False otherwise, detailed_message)
    """
    rest_days = []
    workout_days = []

    for day in DAYS_OF_THE_WEEK:
        if all(workout_plan[day][time] is None for time in TIMES_OF_DAY):
            rest_days.append(day.capitalize())
        else:
            workout_days.append(day.capitalize())

    num_rest_days = len(rest_days)

    if num_rest_days >= min_rest_days:
        return (
            True,
            f"Rest days: {num_rest_days} (minimum {min_rest_days} required). Rest days: {', '.join(rest_days)}. Workout days: {', '.join(workout_days)}",
        )
    else:
        return (
            False,
            f"Rest days: {num_rest_days} (minimum {min_rest_days} required). Rest days: {', '.join(rest_days)}. Workout days: {', '.join(workout_days)}",
        )


def max_rest_days_satisfied(workout_plan, max_rest_days=1):
    """
    Check if the maximum number of rest days is satisfied.
    """
    rest_days = []
    workout_days = []

    for day in DAYS_OF_THE_WEEK:
        if all(workout_plan[day][time] is None for time in TIMES_OF_DAY):
            rest_days.append(day.capitalize())
        else:
            workout_days.append(day.capitalize())

    num_rest_days = len(rest_days)

    if num_rest_days <= max_rest_days:
        return (
            True,
            f"Rest days: {num_rest_days} (maximum {max_rest_days} allowed). Rest days: {', '.join(rest_days)}. Workout days: {', '.join(workout_days)}",
        )
    else:
        return (
            False,
            f"Rest days: {num_rest_days} (maximum {max_rest_days} allowed). Rest days: {', '.join(rest_days)}. Workout days: {', '.join(workout_days)}",
        )


def min_rest_days_between_workouts(workout_plan, min_rest_days=1):
    """
    Check if the minimum number of full rest days between workouts is satisfied.
    """
    last_workout_day = None
    for day in DAYS_OF_THE_WEEK:
        if any(workout_plan[day][time] is not None for time in TIMES_OF_DAY):
            if last_workout_day is not None:
                rest_days = (
                    DAYS_OF_THE_WEEK.index(day)
                    - DAYS_OF_THE_WEEK.index(last_workout_day)
                    - 1
                )
                if rest_days < min_rest_days:
                    return (
                        False,
                        f"Rest days between workouts: {rest_days} (minimum {min_rest_days} required). Last workout day: {last_workout_day}. Current day: {day}",
                    )
            last_workout_day = day
    return True, f"Min rest days between workouts: {min_rest_days} satisfied"


def workout_variety_score(workout_plan):
    """
    Calculate variety score based on unique workout combinations.

    Args:
        workout_plan: The workout plan dictionary

    Returns:
        (int, str): (Difference between total workouts and unique workouts (0 = all unique, higher = more repeats), detailed_message)
    """
    workout_sets = set()
    workout_count = 0
    workout_details = []

    for day in DAYS_OF_THE_WEEK:
        for time in TIMES_OF_DAY:
            if workout_plan[day][time] is not None:
                workout_count += 1
                exercise_names = [
                    exercise["exercise_name"] for exercise in workout_plan[day][time]
                ]
                workout_set = tuple(sorted(exercise_names))
                workout_sets.add(workout_set)
                workout_details.append(
                    f"{day.capitalize()} {time}: {', '.join(exercise_names)}"
                )

    variety_score = workout_count - len(workout_sets)

    message = f"Total workouts: {workout_count}, Unique combinations: {len(workout_sets)}, Variety score: {variety_score}"
    if workout_details:
        message += f". Workouts: {'; '.join(workout_details)}"

    return variety_score, message


def day_has_workout(workout_plan, day):
    """
    Check if a specific day has any workouts scheduled.

    Args:
        workout_plan: The workout plan dictionary
        day: Day of the week

    Returns:
        (bool, str): (True if the day has at least one workout, False otherwise, detailed_message)
    """
    workouts = []
    for time in TIMES_OF_DAY:
        if workout_plan[day][time] is not None:
            exercise_names = [
                exercise["exercise_name"] for exercise in workout_plan[day][time]
            ]
            workouts.append(f"{time}: {', '.join(exercise_names)}")

    if workouts:
        return True, f"{day.capitalize()} has workouts: {'; '.join(workouts)}"
    else:
        return False, f"{day.capitalize()} has no workouts scheduled"


def time_has_workout(workout_plan, time):
    """
    Check if a specific time has any workouts scheduled across the week.

    Args:
        workout_plan: The workout plan dictionary
        time: Time of day

    Returns:
        (bool, str): (True if the time has at least one workout across the week, False otherwise, detailed_message)
    """
    workouts = []
    for day in DAYS_OF_THE_WEEK:
        if workout_plan[day][time] is not None:
            exercise_names = [
                exercise["exercise_name"] for exercise in workout_plan[day][time]
            ]
            workouts.append(f"{day.capitalize()}: {', '.join(exercise_names)}")

    if workouts:
        return True, f"Workouts scheduled at {time}: {'; '.join(workouts)}"
    else:
        return False, f"No workouts scheduled at {time}"


def day_time_has_workout(workout_plan, day, time):
    """
    Check if a specific day and time has workouts scheduled.

    Args:
        workout_plan: The workout plan dictionary
        day: Day of the week
        time: Time of day

    Returns:
        (bool, str): (True if the day/time has workouts, False otherwise, detailed_message)
    """
    if workout_plan[day][time] is None:
        return False, f"No workout scheduled for {day.capitalize()} {time}"
    else:
        exercise_names = [
            exercise["exercise_name"] for exercise in workout_plan[day][time]
        ]
        return (
            True,
            f"Workout scheduled for {day.capitalize()} {time}: {', '.join(exercise_names)}",
        )


def day_is_rest_day(workout_plan, day):
    """
    Check if a specific day is a rest day (no workouts).

    Args:
        workout_plan: The workout plan dictionary
        day: Day of the week

    Returns:
        (bool, str): (True if the day has no workouts, False otherwise, detailed_message)
    """
    workouts = []
    for time in TIMES_OF_DAY:
        if workout_plan[day][time] is not None:
            exercise_names = [
                exercise["exercise_name"] for exercise in workout_plan[day][time]
            ]
            workouts.append(f"{time}: {', '.join(exercise_names)}")

    if workouts:
        return (
            False,
            f"{day.capitalize()} is not a rest day. Workouts: {'; '.join(workouts)}",
        )
    else:
        return True, f"{day.capitalize()} is a rest day (no workouts scheduled)"


def time_is_empty(workout_plan, time):
    """
    Check if a specific time is empty across all days.

    Args:
        workout_plan: The workout plan dictionary
        time: Time of day

    Returns:
        (bool, str): (True if the time has no workouts across all days, False otherwise, detailed_message)
    """
    workouts = []
    for day in DAYS_OF_THE_WEEK:
        if workout_plan[day][time] is not None:
            exercise_names = [
                exercise["exercise_name"] for exercise in workout_plan[day][time]
            ]
            workouts.append(f"{day.capitalize()}: {', '.join(exercise_names)}")

    if workouts:
        return False, f"Workouts scheduled at {time}: {'; '.join(workouts)}"
    else:
        return True, f"No workouts scheduled at {time} across all days"
