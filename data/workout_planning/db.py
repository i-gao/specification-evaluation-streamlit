import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from functools import cached_property
from typing import List, Optional, Literal, Union, Dict   
import pandas as pd
import sys
import os
import warnings

from data.database import Database


# Global constants
DAYS_OF_THE_WEEK = [
    "sunday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
]
TIMES_OF_DAY: List[str] = [
    "early morning (6-9am)",
    "late morning (9-11am)",
    "midday (11am-1pm)",
    "afternoon (2-5pm)",
    "evening (5-10pm)",
]


DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))


class ExerciseDB(Database):
    """
    A database of exercises.
    """

    def __init__(
        self, file_path: str = f"{DATASET_ROOT}/assets/exercises_with_variations.csv"
    ):
        df = pd.read_csv(file_path)
        df['URL'] = df['URL'].fillna("").astype(str)
        df = df.fillna("")

        super().__init__(
            {
                "exercises": (
                    "Database of functional fitness exercises to create workouts from",
                    df,
                    {
                        "exercise_name": "Exercise name",
                        "variation_name": "Variation name",
                        "difficulty_level": "Difficulty level of the exercise",
                        "target_muscle_group": "Target muscle group of the exercise",
                        "prime_mover_muscle": "Prime mover muscle of the exercise",
                        "secondary_muscle": "Secondary muscle of the exercise",
                        "tertiary_muscle": "Tertiary muscle of the exercise",
                        "primary_equipment": "Primary equipment of the exercise",
                        "num_primary_items": "Number of primary items of the exercise",
                        "secondary_equipment": "Secondary equipment of the exercise",
                        "num_secondary_items": "Number of secondary items of the exercise",
                        "posture": "Posture of the exercise",
                        "single_or_double_arm": "Whether the exercise is done with one or two arms",
                        "continuous_or_alternating_arms": "Whether the exercise is done continuously or alternating arms",
                        "grip": "Grip of the exercise",
                        "load_position_ending": "Load position of the exercise",
                        "continuous_or_alternating_legs": "Whether the exercise is done continuously or alternating legs",
                        "foot_elevation": "Foot elevation of the exercise",
                        "combination_exercises": "Whether the exercise is a combination exercise",
                        "movement_pattern_1": "Movement pattern of the exercise",
                        "movement_pattern_2": "Movement pattern of the exercise",
                        "movement_pattern_3": "Movement pattern of the exercise",
                        "plane_of_motion_1": "Plane of motion of the exercise",
                        "plane_of_motion_2": "Plane of motion of the exercise",
                        "plane_of_motion_3": "Plane of motion of the exercise",
                        "body_region": "Body region of the exercise",
                        "force_type": "Force type of the exercise",
                        "mechanics": "Mechanics of the exercise",
                        "laterality": "Laterality of the exercise",
                        "primary_exercise_classification": "Primary exercise classification of the exercise",
                        "time_or_reps": "Whether the variation is a time-based set or reps-based set",
                        "num_sets": "Number of sets of the exercise",
                        "time_per_set": "Time per set of the exercise",
                        "rest_time": "Rest time of the exercise",
                        "total_time_seconds": "Total time of all sets of the exercise in seconds",
                        "num_reps_per_set": "Number of reps per set of the exercise",
                    },
                )
            }
        )
        self.df = self.tables["exercises"]

    def get_exercise_by_variation(
        self,
        name: str,
        time_or_reps: Optional[Literal["time", "reps"]] = None,
        num_sets: Optional[Union[int, float]] = None,
        rest_time: Optional[Union[int, float]] = None,
        time_per_set: Optional[Union[int, float]] = None,
        num_reps_per_set: Optional[Union[int, float]] = None,
    ) -> pd.Series:
        """
        Returns the exercise with the given name and variation details.
        Match as many of the details as given
        """
        conditions = []
        if time_or_reps is not None:
            if time_or_reps == "reps":
                time_or_reps = "rep"
            conditions.append(self.df["time_or_reps"] == time_or_reps)
        if num_sets is not None:
            conditions.append(self.df["num_sets"] == float(num_sets))
        if rest_time is not None:
            conditions.append(self.df["rest_time"] == float(rest_time))
        if time_per_set is not None:
            conditions.append(self.df["time_per_set"] == float(time_per_set))
        if num_reps_per_set is not None:
            conditions.append(self.df["num_reps_per_set"] == float(num_reps_per_set))
        filtered = self.df[self.df["exercise_name"] == str(name)]
        for condition in conditions:
            try:
                # Suppress warnings:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    filtered = filtered[condition]
            except:
                pass
        if len(filtered) == 0:
            return None
        return filtered.iloc[0].to_dict()

    def get_exercise_by_name(self, name: str) -> Optional[Dict]:
        """
        Returns the first exercise with the given name.
        Since multiple exercises have the same name, just show the first one.
        """
        filtered = self.df[self.df["exercise_name"] == str(name)]
        if len(filtered) == 0:
            return None
        return filtered.iloc[0].to_dict()

    def get_all_exercises_by_name(self, name: str) -> List[Dict]:
        """
        Returns all exercises with the given name.
        """
        filtered = self.df[self.df["exercise_name"] == str(name)]
        if len(filtered) == 0:
            return []
        return [row.to_dict() for _, row in filtered.iterrows()]

    @cached_property
    def stats(self):
        print("########### Exercise DB Stats ###########")
        print(f"Number of exercise variations: {len(self.df)}")
        print(f"Distribution of exercises:\n\t{Counter(self.df['exercise_name'])}")
