from typing import Dict, Any

from utils.misc import parse_json
from data.workout_planning.db import (
    ExerciseDB,
    DAYS_OF_THE_WEEK,
    TIMES_OF_DAY,
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


