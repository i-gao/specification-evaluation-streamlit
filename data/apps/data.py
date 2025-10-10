from typing import List, Tuple, Dict, Optional
import sys
import re
import json
import os
from datasets import load_dataset, load_from_disk
import numpy as np
import copy
import streamlit as st

from llm_sandbox import SandboxSession
from data.dataset import SpecificationCollection, Specification, FixedSpecification
from utils.streamlit_types import FormElement
from utils.misc import parse_code, parse_json, subset_data, add_section
from utils.code_sandbox import run_python_script_with_json_input
from data.actions import Action, get_classification_actions
from langchain_core.tools import tool
from utils.misc import fuzzy_match

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))
PROTOTYPES_FRAC = 0.2
MAX_TEST_CASES = 10

FIXED_INSTRUCTIONS = """
### What you need to prompt the assistant to do
In this task, **your goal is to get the assistant to write code which solves a given problem.** When the chat session starts, you will see an explanation of what the code should do on the left side of the screen.

> For example, if the problem statement is to "sum two numbers", then the assistant should write code which takes in two numbers, adds them, and then outputs the sum.

To evaluate how well the code solves the problem, one can think through whether the code does the right thing on some **test cases.** A "test case" is an input-output pair. When the code is run on the input, it should produce the corresponding output. If it produces something different, then the code is incorrect.

> For example, if the problem statement is to "sum two numbers", then some test cases could be:
> ```
> input: 1, 1
> output: 2
> ```
> ```
> input: 2, 5
> output: 7
> ```

Thinking through test cases can also help you understand the problem better. When the chat session starts, you will see some test cases on the left side of the screen.

*It's okay if you don't know how to code yourself --- the challenge is to prompt the assistant to write the code for you!*

### The tricky part

Correct code needs to behave correctly on *all* test cases, including inputs we might not expect, called "edge cases." These edge cases may not be spelled out in the problem description. You will need to think through what the logical output is for these edge cases, and/or reference the test cases.

> For example, if the problem statement is to "sum two numbers", one edge case could be if the inputs are not numbers: e.g., input: "a", "b". The correct output here could be to print "Unknown", or to print "Not valid." Defining the right behavior for these edge cases is the core challenge of the task.

### How you will be scored
We provide a tool that scores code. The score is between 0\% and 100\%, with higher being better. The score corresponds to the percentage of test cases that the code passes. You can view the inputs of those test cases on the left side of the screen, but we do not initially provide the correct outputs.
"""

COMMONSENSE_DESCRIPTION = """The code must be efficient and terminate in 60 seconds or less. It must also be bug-free and not cause errors on the computer."""


def render_fixed_task_explanation():
    """Render the fixed task explanation for APPS."""
    st.markdown(FIXED_INSTRUCTIONS)
    st.markdown(COMMONSENSE_DESCRIPTION)


class APPSDataset(SpecificationCollection):
    """
    The APPS code dataset pairs introductory / interview / competition code problems (in natural langauge)
    with Python solutions.

    Paper: https://arxiv.org/abs/2105.09938
    Dataset: https://huggingface.co/test_cases/codeparrot/apps

    Each code problem is a specification (R, φ, θ):
    - R(y) validates the code solution y using some test cases.
    - θ is the full problem text, which is fully specified except for the test cases
    - φ is the set of test cases used to evaluate the code solution

    Dev / Test split:
    - original train split is used as dev set
    - original test split is used as test set
    """

    @property
    def dataset_name(self) -> str:
        return "apps"

    @property
    def dataset_pretty_name(self) -> str:
        return "Introductory programming problems"

    @property
    def dataset_description(self) -> str:
        return "Work with the assistant to **solve introductory programming problems.**"

    @property
    def assets_file_id(self) -> str:
        return "1UZ2_tidKeLX8EJ2FPIlxkXPE_GkYJZxy"

    @property
    def default_docker_images(self) -> List[Dict[str, str]]:
        return [
            {
                "image_name": "apps",
                "dockerfile_path": "data/apps/reward_utils/Dockerfile",
                "build_context": "data/apps",
                "description": "Docker image for APPS code evaluation",
            }
        ]

    def _create_user_expertise_form(self) -> List[FormElement]:
        """Create the user expertise form for APPS."""
        return [
            FormElement(
                input_type="radio",
                label="How familiar are you with coding in Python?",
                options=[
                    "Never coded in any language",
                    "Coded in languages other than Python",
                    "Introductory Python experience",
                    "Python experience",
                    "Expert in Python",
                ],
                default="Python experience",
                required=True,
                help="This helps us understand your Python programming experience level",
            )
        ]

    def __init__(
        self,
        dev: bool = False,
        docker_image: str = "apps",
        fixed_indexes: Optional[List[int]] = None,
        persist_docker_container: bool = True,
        allow_multimodal_actions: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(dev=dev, **kwargs)

        # Load all the problems
        self._rows = load_from_disk(
            f"{DATASET_ROOT}/assets/apps_{'train' if dev else 'test'}"
        )
        self.fixed_length = len(self._rows)
        self.custom_length = 0  # No custom specifications for APPS
        self._docker_image = docker_image
        self._persist_docker_container = persist_docker_container
        self._allow_multimodal_actions = allow_multimodal_actions
        if self._docker_image is None:
            try:
                from pyext import RuntimeModule
            except:
                raise ValueError(
                    "Running without docker container. APPS requires the pyext package; run `pip install pyext` to install."
                )

        # All subclasses must have these attributes set
        self._finish_init()

        if fixed_indexes is not None:
            self._load_fixed_specs(fixed_indexes)

    def _load_fixed_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, FixedSpecification]:
        if indexes is None:
            return {}

        # Load requested specs
        specs = {}
        for ix in indexes:
            row = _parse_problem(self._rows[ix])
            row["expect_fn"] = "fn_name" in row["input_output"]
            # Some problems expect the model to write a function with a specific name,
            # while others want the model to read from input()

            fmt_instructions = "\n".join(
                f"*\t{line}"
                for line in [
                    "Solve the problem in Python.",
                    "Do not use '___main___' or 'if __name__ == '__main__'': just write the code.",
                    "Do not import any libraries.",
                    "Only use one code block in your response (```python).",
                    "The code must be efficient and terminate in 60 seconds or less.",
                ]
                + (
                    [
                        "Do not wrap the code in a function, but instead write code that runs when it is executed in the global scope.",
                        "Use input() for inputs.",
                        "Print all outputs to the console.",
                    ]
                    if not row["expect_fn"]
                    else [
                        f"Write a function named {row['input_output']['fn_name']}.",
                        "Return the result of the function call.",
                    ]
                )
            )
            theta = _remove_examples(row["question"])
            theta, input_output = theta.split("-----Input-----")
            input, output = input_output.split("-----Output-----")
            theta = add_section(
                "Full problem description",
                theta.replace("\n\n", "\n<chunk>\n")
                + "\n<chunk>\n"
                + "----- Input format: "
                + input
                + "\n<chunk>\n"
                + "----- Output format: "
                + output,
            )
            test_cases = row["test_cases"]

            observed_test_cases = subset_data(
                test_cases, PROTOTYPES_FRAC, 1.0, True, max_len=MAX_TEST_CASES
            )
            assert len(observed_test_cases) > 0, "No test cases found: "

            signature = (
                "The task is to get the assistant to write code that accomplishes the following task.\n\n"
                + row["high_level_goal"].replace(
                    "The high-level, core goal of this coding problem is:\n\n",
                    "Problem statement (what the code should do): ",
                )
            )
            signature += "\n\n" + add_section(
                "Test cases",
                f"Here are some test cases for the code:\n\n{fmt_test_cases(observed_test_cases)}",
            )

            # persist one docker container
            if self._persist_docker_container and self._docker_image is not None:
                session = SandboxSession(image=self._docker_image)
                session.open()
                container_id = session.container.id
            else:
                container_id = None

            spec = FixedSpecification(
                dataset_name=self.dataset_name,
                index=f"fixed_{ix}",
                full_specification=theta,
                initial_specification=signature,
                validity_fn=validity_fn,
                validity_kwargs={
                    "row": row,
                    "docker_image": self._docker_image,
                    "docker_container_id": container_id,
                },
                # validity_fn_tool_name=None,  # Not provided
                # validity_fn_tool_description=None,  # Not provided
                reward_fn=reward_fn,
                reward_kwargs={
                    "row": row,
                    "docker_image": self._docker_image,
                    "docker_container_id": container_id,
                },
                # reward_fn_tool_name=None,  # Not provided
                # reward_fn_tool_description=None,  # Not provided
                ystar=row["solutions"][0],
                metric_name="test_pass_rate",
                render_task_explanation=render_fixed_task_explanation,
                actions=get_actions(
                    test_cases,
                    row,
                    docker_image=self._docker_image,
                    docker_container_id=container_id,
                )
                + get_classification_actions(
                    observed_test_cases,
                    test_cases,
                    override_tool_names={
                        "get_documents_to_annotate": "get_test_cases",
                        "sample_documents_to_annotate": "sample_test_cases",
                        "get_correct_annotation": "get_expected_test_case_output",
                        "check_single_annotation": "check_test_case_output",
                    },
                    override_tool_descriptions={
                        "get_documents_to_annotate": "View the test cases (inputs only)",
                        "sample_documents_to_annotate": "Sample a batch of test cases (inputs only)",
                        "get_correct_annotation": "Get the expected output for a test case input",
                        "check_single_annotation": "Check if the output for a test case input matches the expected output",
                    },
                    fuzzy_match_x=lambda x, y: fuzzy_match(
                        x,
                        y,
                        ignore_whitespace=True,
                        ignore_punctuation=True,
                        ignore_case=True,
                    ),
                    return_as_df=self._allow_multimodal_actions,
                ),
                fmt_instructions=fmt_instructions,
                name=f"{'train' if self.dev else 'test'}_all_{row['problem_id']}",
                user_expertise_form=self._create_user_expertise_form(),
                test_cases=test_cases,  # Not in base class
            )
            spec._container_ids = [container_id]
            specs[ix] = spec
        return specs


def fmt_test_cases(test_cases: Dict[str, str]) -> str:
    return "\n\n".join(
        [
            f"input: {input.strip()}\noutput: {output.strip()}"
            for input, output in test_cases.items()
        ]
    )


def get_actions(
    test_cases: Dict[str, str], row: dict, docker_image: str, docker_container_id: str
) -> List[Action]:
    # @tool(parse_docstring=True)
    # def get_expected_output(input: str) -> str:
    #     """
    #     Get the expected output for a test case input.
    #     Returns the output of the test case if it is known, otherwise returns "Unknown".

    #     Args:
    #         input (str): The input test case
    #     """
    #     if input in test_cases:
    #         return test_cases[input]

    #     for k, v in test_cases.items():
    #         if input.strip() == k.strip():
    #             return v

    #     return "Unknown"

    @tool(parse_docstring=True)
    def run_code_on_input(code: str, input: str) -> str:
        """
        Get the output of a code solution for a given input.

        Args:
            code (str): The code solution
            input (str): The input test case
        """
        row_copy = copy.deepcopy(row)
        return get_code_output(
            yhat=code,
            input=input,
            row=row_copy,
            docker_image=docker_image,
            docker_container_id=docker_container_id,
        )

    return [
        # Action(fn=get_expected_output, is_public=False),
        Action(
            fn=run_code_on_input,
            is_public=True,
            is_human=True,
            name="Run code on test input",
        ),
    ]


def _fmt_output(lst: List[List[set]]) -> List[str]:
    return [
        "\n".join(" ".join(str(si) for si in list(s)) for s in line) for line in lst
    ]


def validity_fn(
    yhat: str,
    row: dict,
    docker_image: str,
    docker_container_id: str,
    raise_errors: bool = False,
) -> Tuple[bool, dict]:
    if "```" not in yhat:
        yhat = "```python\n" + yhat + "\n```"
    yhat = parse_code(yhat, language="python")
    if yhat is None:
        if raise_errors:
            raise Exception(
                "Could not parse code from the response. Wrap code with ```{language} and ``` at the beginning and end."
            )
        return False, {"error_msg": "Could not parse code from the response."}

    b, s, d = get_pass_rate(
        yhat,
        row,
        docker_image,
        docker_container_id,
        raise_errors,
    )
    return b, d


def reward_fn(
    yhat: str,
    row: dict,
    docker_image: str,
    docker_container_id: str,
    raise_errors: bool = False,
) -> Tuple[float, dict]:
    b, s, d = get_pass_rate(
        yhat,
        row,
        docker_image,
        docker_container_id,
        raise_errors,
    )
    return s, d


def get_pass_rate(
    yhat: str,
    row: dict,
    docker_image: str,
    docker_container_id: str,
    raise_errors: bool = False,
) -> Tuple[bool, float, dict]:
    """
    row is the original row from the dataset.
    -2 is a compile error, -1 is a runtime error, False is a failed test case, True is a passed test case
    """

    # Implement the code & run the test cases
    if "```" not in yhat:
        yhat = "```python\n" + yhat + "\n```"
    yhat = parse_code(yhat, language="python")
    if yhat is None:
        if raise_errors:
            raise Exception(
                "Could not parse code from the response. Wrap code with ```{language} and ``` at the beginning and end."
            )
        return (
            False,
            float("-inf"),
            {"error_msg": "Could not parse code from the response."},
        )
    try:
        if row["expect_fn"]:
            assert (f"def {row['input_output']['fn_name']}(") in yhat, (
                f"Code must be a function that is named {row['input_output']['fn_name']}"
            )
        else:
            assert "input()" in yhat, "Code must use input() for inputs."
    except Exception as e:
        if raise_errors:
            raise e
        return (
            False,
            float("-inf"),
            {"error_msg": str(e)},
        )

    # Run the testing script
    try:
        script_dir = (
            f"{DATASET_ROOT}/reward_utils" if docker_image is None else "/sandbox"
        )
        run_id, result, output_filenames = run_python_script_with_json_input(
            input_dict={"problem": row, "yhat": yhat},
            command=f"python {script_dir}/check_correctness.py" + " {input_filename}",
            docker_image=docker_image,
            docker_container_id=docker_container_id,
            root_dir="" if docker_image is None else "/sandbox",
        )
    except Exception as e:
        if raise_errors:
            raise Exception("The code failed to run.")
        return False, float("-inf"), {"error_msg": e}

    raw_passed = re.search(r">>>>>> UNIT TEST RESULTS <<<<<<\s+(\[.*\])", result).group(
        1
    )
    raw_passed = eval(raw_passed)
    if raw_passed == -2 or all(p == -2 for p in raw_passed):
        if raise_errors:
            raise Exception("The code failed to compile.")
        return (
            False,
            float("-inf"),
            {"error_msg": "The code failed to compile.", "log": result},
        )
    if all(p == -1 for p in raw_passed):
        if raise_errors:
            raise Exception("The code encountered runtime errors on all test cases.")
        return (
            False,
            float("-inf"),
            {
                "error_msg": "The code encountered runtime errors on all test cases.",
                "log": result,
            },
        )

    def _eval(p):
        if p in ["True", True]:
            return True
        elif p in ["False", False]:
            return False
        else:
            return -1

    raw_passed = [_eval(p) for p in raw_passed]
    passed = [p == True for p in raw_passed]

    raw_output = re.search(r">>>>>> UNIT TEST OUTPUT <<<<<<\s+(\[.*\])", result).group(
        1
    )
    try:
        raw_output = eval(raw_output)
        raw_output = _fmt_output(raw_output)
    except:
        raw_output = None

    if raw_output is None or len(raw_output) == 0:
        return (
            False,
            float("-inf"),
            {"error_msg": "Nothing was printed.", "log": result},
        )

    is_valid = all(p != -1 for p in raw_passed)
    return (
        is_valid,
        np.mean(passed) * 100 if is_valid else float("-inf"),
        {
            "passed": raw_passed,
            "test_inputs": row["input_output"]["inputs"],
            "test_outputs": raw_output,
            "test_expected_outputs": row["input_output"]["outputs"],
        },
    )


def get_code_output(
    yhat: str, input: str, row: dict, docker_image: str, docker_container_id: str
) -> str:
    """
    Get the output of a code solution for a given input.
    row is the original row from the dataset.
    """

    # Basic code validation
    if "```" not in yhat:
        yhat = "```python\n" + yhat + "\n```"
    yhat = parse_code(yhat, language="python")
    if yhat is None:
        raise Exception(
            "Could not parse code from the response. Wrap code with ```{language} and ``` at the beginning and end."
        )
    if row["expect_fn"]:
        assert (f"def {row['input_output']['fn_name']}(") in yhat, (
            f"Code must be a function that is named {row['input_output']['fn_name']}"
        )
    else:
        assert "input()" in yhat, "Code must use input() for inputs."

    # Run the code

    row["input_output"]["inputs"] = [input]
    row["input_output"]["outputs"] = [""]

    try:
        script_dir = (
            f"{DATASET_ROOT}/reward_utils" if docker_image is None else "/sandbox"
        )
        run_id, result, output_filenames = run_python_script_with_json_input(
            input_dict={"problem": row, "yhat": yhat},
            command=f"python {script_dir}/check_correctness.py" + " {input_filename}",
            docker_image=docker_image,
            docker_container_id=docker_container_id,
            root_dir="" if docker_image is None else "/sandbox",
        )
    except:
        raise Exception("The code failed to run.")

    # import pdb; pdb.set_trace()

    raw_output = re.search(r">>>>>> UNIT TEST OUTPUT <<<<<<\s+(\[.*\])", result).group(
        1
    )
    try:
        # preserve set order
        raw_output = raw_output.replace("{", "[").replace("}", "]")
        raw_output = eval(raw_output)
        raw_output = _fmt_output(raw_output)
    except:
        raw_output = None

    if raw_output is None or len(raw_output) == 0:
        raise Exception("Nothing was printed.")
    return raw_output[0]


def _remove_examples(x: str) -> str:
    """
    Remove the examples from the problem text.
    Match the first instance of "-" * n + "Example" or "-" * n + "Examples" and take the text before that, where n is unknown.
    """
    match = re.search(r"(-{2,})Example", x)
    if match:
        return x[: match.start()]
    else:
        return x


def _parse_problem(row: dict) -> dict:
    """
    Call parse_json on the input_output and solutions fields.
    """
    try:
        row["input_output"] = json.loads(row["input_output"])
    except Exception as e:
        pass
    try:
        row["solutions"] = json.loads(row["solutions"])
    except Exception as e:
        pass
    try:
        row["test_cases"] = json.loads(row["test_cases"])
    except Exception as e:
        pass
    return row
