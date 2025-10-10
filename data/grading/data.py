from typing import List, Tuple, Dict, Literal, Optional
import sys
import re
import json
import os
import numpy as np
import copy
import pandas as pd
import random
from statistics import mode


from data.dataset import SpecificationCollection, FixedSpecification
from utils.misc import parse_code, parse_json, subset_data, add_section, parse_for_answer_tags
from utils.streamlit_types import FormElement, DisplayElement
from data.actions import Action, get_classification_actions, get_annotator_output_action
from data.reward import get_avg_score
from langchain_core.tools import tool

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_RELATED_EXAMPLES = 2
DEV_FRAC = 0.3
MAX_COMMENTS_IN_RUBRIC = 3

LETTER_GRADE_TO_NUMBER = {
    "A++": 100,
    "A+": 95,
    "A": 90,
    "A-": 85,
    "B+": 80,
    "B": 75,
    "B-": 70,
    "C+": 65,
    "C": 60,
    "C-": 55,
    "D+": 50,
    "D": 45,
    "D-": 40,
    "F": 30,
}
RUBRIC_KEY_TO_DF_KEY = {
    "program_correctness": "Correctness",
    "code_elegance": "Code Elegance",
    "documentation": "Documentation",
    "readability": "Readability",
}

FIXED_INSTRUCTIONS = """
In this task, **your goal is to come up with a rubric for grading student programming assignments in Java.**

On the next screen, you will see some information about the assignments. Your job is to come up with a rubric for grading the assignments. The rubric should be a set of fair criteria that graders can use to grade the assignments.

### The tricky part
Coming up with a fair rubric is tricky. You will need to use the labeled examples to understand the pattern of how the assignments should be graded.

### How you will be scored
You will prompt the assistant via a chat interface. When the timer runs out, the session is over.

We will then ask the assistant to generate the final rubric, given all it has learned from your conversation. Then, we will ask the assistant to grade all of the assignments based on those instructions. **You will be scored based on what \% of the assignments are graded correctly.** The minimum score is 0\%, and the maximum is 100\%. The better your rubric, the higher your score.
"""


def render_fixed_task_explanation():
    """Render the fixed task explanation for grading."""
    st.markdown(FIXED_INSTRUCTIONS)


def _agg_grade(submission_rows: pd.DataFrame) -> float:
    return np.mean(
        [LETTER_GRADE_TO_NUMBER[row["grade"]] for _, row in submission_rows.iterrows()]
    )


class GradingDataset(SpecificationCollection):

    @property
    def dataset_name(self) -> str:
        return "grading"

    @property
    def dataset_pretty_name(self) -> str:
        return "Java Assignment Grading"

    @property
    def dataset_description(self) -> str:
        return "Work with the assistant to **write a rubric for grading a student programming assignment** from an introductory Java course."

    @property
    def assets_file_id(self) -> str:
        return "1qvg3qaj4fD12KmDAAfKIiePr9sBLHtzn"

    def _create_user_expertise_form(self) -> List[FormElement]:
        """Create user expertise form for grading."""
        return [
            FormElement(
                input_type="radio",
                label="How familiar are you with object-oriented programming concepts and/or introductory Java?",
                options=[
                    "I have no experience with object-oriented programming or Java",
                    "I have basic knowledge of object-oriented programming concepts",
                    "I have some experience with Java programming",
                    "I am comfortable with Java and object-oriented programming",
                    "I am an expert in Java and object-oriented programming"
                ],
                required=True
            )
        ]

    def __init__(
        self,
        dev: bool = False,
        models: List[str] = ["gpt-5-mini"],
        fixed_indexes: Optional[List[int]] = None,
        allow_multimodal_actions: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(dev=dev, **kwargs)

        # Load the data
        df = (
            pd.read_csv(os.path.join(DATASET_ROOT, "assets", "grades_with_paths.csv"))
            .dropna(subset=["grade"])
            .sort_values(["participant_id", "assignment_number"])
        )
        df["assignment_number"] = df["assignment_number"].astype(int)
        participant_ids = df["participant_id"].unique()
        ixs_to_keep = subset_data(
            list(range(len(participant_ids))),
            dev_frac=DEV_FRAC,
            frac=1.0,
            dev=dev,
        )
        participant_ids = participant_ids[ixs_to_keep]

        self._df = df
        self._participant_ids = participant_ids
        self.fixed_length = len(participant_ids)
        self.custom_length = 0  # No custom specifications as requested
        self._allow_multimodal_actions = allow_multimodal_actions
        self._models = models
        self._assignment_spec = open(
            os.path.join(DATASET_ROOT, "assets", "assignment_spec.txt")
        ).read()
        self._rubric = json.load(
            open(os.path.join(DATASET_ROOT, "assets", "rubric.json"))
        )
        rubric = json.load(open(os.path.join(DATASET_ROOT, "assets", "rubric.json")))

        # All subclasses must have these attributes set
        self._finish_init()

        if fixed_indexes is not None:
            self.load_fixed_specs(indexes=fixed_indexes)

    def _load_fixed_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, FixedSpecification]:
        if indexes is None:
            return {}

        # Create specs
        specs = {}
        for ix in indexes:
            participant_id = self._participant_ids[ix]
            participant_df = self._df[self._df["participant_id"] == participant_id]
            ids = participant_df["assignment_number"].unique().tolist()

            if len(ids) <= MAX_RELATED_EXAMPLES:
                raise ValueError(
                    f"Participant {participant_id} has <= {MAX_RELATED_EXAMPLES} assignments."
                )

            prototypes = {
                id: _agg_grade(
                    participant_df[participant_df["assignment_number"] == id]
                )
                for id in ids[:MAX_RELATED_EXAMPLES]
            }
            test_data = {
                id: _agg_grade(
                    participant_df[participant_df["assignment_number"] == id]
                )
                for id in ids[MAX_RELATED_EXAMPLES:]
            }

            signature = (
                "The task is to produce instructions for a grader who is grading student programming assignments. "
                + f"Grades should range from 30 to 100; you can think of the numeric values as these letter grades: {LETTER_GRADE_TO_NUMBER} "
                + "Note that each student submission is given an id; you can use the provided tools to read the submission. "
            )
            signature += "\n\n" + add_section(
                "Assignment specification",
                self._assignment_spec,
            )
            signature += "\n\n" + add_section(
                "Labeled examples",
                "\n".join(
                    [f"Submission {id}: {grade}" for id, grade in prototypes.items()]
                ),
            )

            # Create theta - the rubric with example comments
            theta = (
                "When grading, assign a score to each of the following 4 criteria, and then weight them equally. The conversion between letter grades and numeric scores is as follows: "
                + str(LETTER_GRADE_TO_NUMBER)
            )

            # Get all unique criteria from the rubric
            all_criteria = sorted(
                set().union(
                    *[criteria_dict.keys() for criteria_dict in self._rubric.values()]
                )
            )
            grade_levels = ["A+ to A++", "A", "B", "C", "D", "F"]

            # Organize by criteria first, then by grade levels
            for i, criterion in enumerate(all_criteria):
                criterion_title = (
                    str(i + 1) + ". " + criterion.replace("_", " ").title()
                )
                theta += f"\n{'=' * 60}\n"
                theta += f"{criterion_title:^60}\n"
                theta += f"{'=' * 60}\n\n"

                for grade_level in grade_levels:
                    if grade_level in self._rubric and criterion in self._rubric[grade_level]:
                        description = self._rubric[grade_level][criterion]
                        theta += f"Score: {grade_level}\n"
                        theta += f"{'─' * 40}\n"
                        theta += f"{description}\n\n"

                        # Add example comments for this criterion and grade level
                        example_comments = participant_df[
                            (participant_df["grade"].isin(grade_level.split(" to ")))
                            & (
                                participant_df["skill"]
                                == RUBRIC_KEY_TO_DF_KEY[criterion]
                            )
                            & (participant_df["comments"].notna())
                        ]["comments"].tolist()[:MAX_COMMENTS_IN_RUBRIC]

                        if example_comments:
                            theta += "💬 Example Comments:\n"
                            for i, comment in enumerate(example_comments):
                                theta += f"   {i+1}. {comment}\n"
                            theta += "\n"
                theta += "\n<chunk>\n"
            theta = add_section("Rubric", theta)

            ystar = signature + "\n\n" + theta
            fmt_instructions = "Return a markdown document of annotation guidelines for this task. Wrap the document in <instructions></instructions> tags."

            test_ys = [test_data[id] for id in test_data.keys()]

            spec = FixedSpecification(
                dataset_name=self.dataset_name,
                index=f"fixed_{ix}",
                full_specification=theta,
                initial_specification=signature,
                validity_fn=validity_fn,
                reward_fn=reward_fn,
                reward_kwargs={
                    "test_dataset": test_data,
                    "models": self._models,
                    "df": participant_df,
                },
                reward_fn_tool_name="score_grading_rubric",
                reward_fn_tool_description="Score the grading rubric based on grading accuracy",
                ystar=ystar,
                # metric_name=None,  # Not provided
                # baseline_scores=None,  # Not provided
                render_task_explanation=render_fixed_task_explanation,
                actions=[
                    a
                    for a in get_classification_actions(
                        prototypes,
                        test_data,
                        override_tool_names={
                            "get_documents_to_annotate": "get_submission_ids",
                            "sample_documents_to_annotate": "sample_submission_ids",
                            "get_correct_annotation": "get_correct_grade",
                        },
                        override_tool_descriptions={
                            "get_submission_ids": "Get the ids of the submissions to grade",
                            "sample_submission_ids": "Sample a batch of the ids of the submissions to grade",
                            "get_correct_grade": "Get the correct numerical grade for a submission",
                        },
                        fuzzy_match_y=lambda y1, y2: np.abs(float(y1) - float(y2)) <= 5,
                        return_as_df=self._allow_multimodal_actions,
                    )
                    if a.fn.name != "check_single_annotation"
                ]
                + get_actions(participant_df)
                + get_annotator_output_action(self._models),
                fmt_instructions=fmt_instructions,
                name=f"grader_{participant_id}",
                user_expertise_form=self._create_user_expertise_form(),
            )
            specs[ix] = spec
        return specs


def load_all_files(id: int, df: pd.DataFrame) -> str:
    path = df[df["assignment_number"] == id]["assignment_path"].unique().tolist()[0]

    def _read_file(file_path: str) -> str:
        try:
            return open(file_path).read()
        except FileNotFoundError:
            return f"File {file_path} not found in submission {id}."

    valid_files = [
        filename
        for filename in os.listdir(os.path.join(DATASET_ROOT, "assets", path))
        if (
            filename.endswith(".java")
            or filename.endswith(".md")
            or filename.endswith(".txt")
        )
    ]

    file_contents = {
        filename: _read_file(os.path.join(DATASET_ROOT, "assets", path, filename))
        for filename in valid_files
    }
    out = ""
    for filename, content in file_contents.items():
        out += add_section(filename, content, style="code") + "\n"
    return out


def can_be_cast_as_float(z: str) -> bool:
    try:
        float(z)
        return True
    except ValueError:
        return False


def validity_fn(yhat: str, raise_errors: bool = False) -> bool:
    valid = "<instructions>" in yhat and "</instructions>" in yhat
    if not valid and raise_errors:
        raise Exception("Could not parse grading rubric from the output. Make sure the output is a valid markdown document of grading rubric. Wrap the document in <instructions></instructions> tags.")
    return valid, {}

def reward_fn(
    yhat: str,
    test_dataset: Dict[str, str],
    models: List[str],
    df: pd.DataFrame,
    raise_errors: bool = False,
) -> Tuple[float, dict]:

    parsed_yhat = parse_for_answer_tags(yhat, keyword="instructions", return_none_if_not_found=True)
    if parsed_yhat is None:
        parsed_yhat = yhat
    
    test_dataset_with_explicit_files = {
        load_all_files(id, df): y for id, y in test_dataset.items()
    }

    scores, outputs = get_avg_score(
        yhat,
        test_dataset_with_explicit_files,
        models,
        score_fn=squared_error,
        validate_fn=can_be_cast_as_float,
        final_answer_desc="numerical grade for the submission",
    )
    mse = np.mean([v for v in scores.values() if v is not None])
    return (
        100 - (mse / 100),  # scale to 0-100
        {"outputs": outputs, "mse": mse},
    )


def squared_error(ystar: str, yhat: float) -> float:
    """
    Check the grade of a given submission.

    Args:
        yhat (str): The predicted grade
        ystar (Dict[int, float]): The true grade
    """
    yhat = float(yhat)
    ystar = float(ystar)
    return np.square(yhat - ystar)


def get_actions(df: pd.DataFrame) -> List[Action]:

    @tool(parse_docstring=True)
    def ls_starter_code() -> str:
        """
        List the files in the starter code.
        """
        return [
            filename
            for filename in os.listdir(
                os.path.join(DATASET_ROOT, "assets", "starter_code")
            )
            if (
                filename.endswith(".java")
                or filename.endswith(".md")
                or filename.endswith(".txt")
            )
        ]

    @tool(parse_docstring=True)
    def read_starter_code_file(filename: str) -> str:
        """
        List the contents of a file in the starter code.

        Args:
            filename (str): The name of the file to read
        """
        try:
            return open(
                os.path.join(DATASET_ROOT, "assets", "starter_code", filename)
            ).read()
        except FileNotFoundError:
            return f"File {filename} not found in starter code."

    @tool(parse_docstring=True)
    def ls_submission_files(id: int) -> str:
        """
        List the files in the submission for a given submission id.

        Args:
            id (int): The id of the submission
        """
        assert id in df["assignment_number"].unique(), f"ID {id} is invalid."
        path = df[df["assignment_number"] == id]["assignment_path"].unique().tolist()
        assert len(path) == 1, f"ID {id} is invalid."
        path = path[0]
        return [
            filename
            for filename in os.listdir(os.path.join(DATASET_ROOT, "assets", path))
            if (
                filename.endswith(".java")
                or filename.endswith(".md")
                or filename.endswith(".txt")
            )
        ]

    @tool(parse_docstring=True)
    def read_submission_file(id: int, filename: str) -> str:
        """
        Read the contents of a file in the submission for a given submission id.

        Args:
            id (int): The id of the submission
            filename (str): The name of the file to read
        """
        assert id in df["assignment_number"].unique(), f"ID {id} is invalid."
        path = df[df["assignment_number"] == id]["assignment_path"].unique().tolist()
        assert len(path) == 1, f"ID {id} is invalid."
        path = path[0]
        try:
            return open(os.path.join(DATASET_ROOT, "assets", path, filename)).read()
        except FileNotFoundError:
            return f"File {filename} not found in submission {id}."

    @tool(parse_docstring=True)
    def comment_on_submission(id: int) -> str:
        """
        Returns a comment that gives feedback on the submission for a given submission id.

        Args:
            id (int): The id of the submission
        """
        assert id in df["assignment_number"].unique(), f"ID {id} is invalid."
        comments = (
            df[df["assignment_number"] == id][["comments", "skill", "grade"]]
            .dropna(subset=["comments"])
            .sample(frac=1)
        )
        out = ""
        for comment in comments["comments"]:
            if comment is None:
                continue
            out += f"{comment} "
        return out

    @tool(parse_docstring=True)
    def check_grade(id: int, grade: float) -> str:
        """
        Check the grade for a given submission id. Describes how close the grade is to the correct grade.

        Args:
            id (int): The id of the submission
            grade (float): The grade to check
        """
        id = int(id)
        assert id in df["assignment_number"].unique(), f"ID {id} is invalid."
        true_grade = _agg_grade(df[df["assignment_number"] == id])
        try:
            grade = float(grade)
        except ValueError:
            return f"Grade {grade} is invalid. It should be a number."

        if np.abs(true_grade - grade) <= 2.5:
            return "This is a good grade."
        elif np.abs(true_grade - grade) <= 10:
            modifier = "by a little"
        else:
            modifier = "by a lot"

        if true_grade > grade:
            return f"This grade is too low {modifier}."
        else:
            return f"This grade is too high {modifier}."

    return [
        Action(fn=check_grade, is_public=True, is_human=False, name="Check grade"),
        Action(fn=ls_starter_code, is_public=True, is_human=False, name="List starter code"),
        Action(fn=read_starter_code_file, is_public=True, is_human=False, name="Read starter code file"),
        Action(fn=ls_submission_files, is_public=True, is_human=False, name="List submission files"),
        Action(fn=read_submission_file, is_public=True, is_human=False, name="Read submission file"),
        Action(fn=comment_on_submission, is_public=False, is_human=False, name="Comment on submission"),
    ]
