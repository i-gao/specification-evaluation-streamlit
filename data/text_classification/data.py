import pandas as pd
from typing import List, Tuple, Dict, Optional
import warnings
import sys
import re
import json
import os
import numpy as np
from collections import defaultdict
from langchain_core.tools import tool
import streamlit as st


from data.dataset import SpecificationCollection, Specification, FixedSpecification
from data.actions import get_classification_actions, get_annotator_output_action
from utils.misc import (
    compute_majority_class_accuracy,
    compute_random_sampling_accuracy,
    subset_data,
    add_section,
    parse_json,
    parse_for_answer_tags,
)
from data.reward import get_avg_score
from utils.streamlit_types import FormElement, DisplayElement
from data.text_classification.tasks import TASKS, get_task

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))

TRAIN_FRAC = 0.1  # Use 10% of data for related examples
MAX_TRAIN_EXAMPLES = 10
MAX_TEST_COVARIATES = 500

DEV_FRAC = 0.4


class TextClassificationDataset(SpecificationCollection):
    """
    Text classification tasks.
    """

    @property
    def dataset_name(self) -> str:
        return "text_classification"

    @property
    def dataset_pretty_name(self) -> str:
        return "Text classification"

    @property
    def dataset_description(self) -> str:
        return "Work with the assistant to **label a dataset of text snippets.**"

    @property
    def assets_file_id(self) -> str:
        return "1C2peMt49kgEdicC-RLiUVekjIicqpk0T"

    def _create_user_expertise_form(self) -> List[FormElement]:
        """Create the user expertise form for text classification."""
        return [
            FormElement(
                input_type="radio",
                label="How familiar are you with data labeling?",
                options=["Beginner", "Intermediate", "Advanced", "Expert"],
                default="Intermediate",
                required=True,
                help="This helps us understand your data labeling experience level",
            )
        ]

    def __init__(
        self,
        dev: bool = False,
        fixed_indexes: Optional[List[int]] = None,
        allow_multimodal_actions: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(dev=dev, **kwargs)

        # Load all the data
        self._data = {}
        self._codebooks = {}
        self._subsets = subset_data(TASKS, DEV_FRAC, 1.0, dev)
        self.fixed_length = len(self._subsets)
        self.custom_length = 0  # No custom specifications for codebook_llm
        self._allow_multimodal_actions = allow_multimodal_actions

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
            subset = self._subsets[ix]
            data, description, label_set, codebook, num_labels_per_x = get_task(subset)
            train_df = subset_data(
                data, TRAIN_FRAC, 1.0, True, max_len=MAX_TRAIN_EXAMPLES
            ).reset_index(drop=True)
            test_df = subset_data(
                data, TRAIN_FRAC, 1.0, False, max_len=MAX_TEST_COVARIATES
            ).reset_index(drop=True)
            test_df["id"] = test_df.index.astype(str)
            test_df = test_df[["id", "input", "label"]]

            train_data = {row["input"]: row["label"] for _, row in train_df.iterrows()}
            test_data = {row["input"]: row["label"] for _, row in test_df.iterrows()}

            # build initial specification
            signature = (
                f"{description}\nThe input is a text snippet. Each input should be labeled with up to {num_labels_per_x} of the following labels:\n"
                + "\n".join([f"* `{label}`" for label in label_set])
            )
            signature_multimodal = [
                DisplayElement(
                    input_type="markdown",
                    value=signature,
                ),
                DisplayElement(
                    input_type="markdown",
                    value="Here are some labeled examples of the task.",
                ),
                DisplayElement(
                    input_type="dataframe",
                    value=train_df,
                    hide_index=True,
                ),
                DisplayElement(
                    input_type="markdown",
                    value="Here is the above data, but in an easily copy-pasteable format.",
                ),
                DisplayElement(
                    input_type="code",
                    value=fmt_examples(train_data),
                ),
                DisplayElement(
                    input_type="markdown",
                    value="Here are the inputs you need to label.",
                ),
                DisplayElement(
                    input_type="dataframe",
                    value=test_df[["id", "input"]],
                    hide_index=True,
                ),
            ]
            signature += "\n\n" + add_section(
                "Labeled examples", fmt_examples(train_data)
            )
            signature += "\n\n" + add_section(
                "Inputs to label", fmt_examples(test_data, show_output=False)
            )

            fmt_instructions = f"The user is asking you to label all {len(test_data)} inputs in the test set. When labeling, you should return a JSON object where the keys are the input ids (0 -- {len(test_df) - 1}), and the values are the corresponding labels (as strings). When fully labeling the dataset, there should be a total of {len(test_data)} entries in the JSON object."
            fmt_instructions += "\n\n" + add_section(
                "Inputs to label", test_df[["id", "input"]].to_json(orient="records")
            )

            spec = FixedSpecification(
                dataset_name=self.dataset_name,
                index=f"fixed_{ix}",
                full_specification=signature + "\n\n" + codebook,
                initial_specification=signature,
                initial_specification_multimodal=signature_multimodal,
                reward_fn=reward_fn,
                reward_kwargs={
                    "test_df": test_df,
                },
                validity_fn=validity_fn,
                validity_kwargs={
                    "test_df": test_df,
                },
                ystar=dict(zip(test_df["id"], test_df["label"])),
                metric_name="accuracy",
                baseline_scores={
                    "majority": compute_majority_class_accuracy(test_df["label"]),
                    "random": compute_random_sampling_accuracy(test_df["label"]),
                    "uniform": compute_random_sampling_accuracy(np.array(label_set)),
                },
                render_task_explanation=self._render_fixed_task_explanation,
                actions=get_classification_actions(
                    train_data,
                    test_data,
                    override_tool_names={
                        "get_correct_annotation": "get_correct_label",
                        "check_single_annotation": "check_single_label",
                    },
                    override_tool_descriptions={
                        "get_correct_annotation": "Get the correct label for a text snippet",
                        "check_single_annotation": "Check if the predicted label for a text snippet matches the correct label",
                    },
                    return_as_df=self._allow_multimodal_actions,
                )[-2:],
                fmt_instructions=fmt_instructions,
                render_msg_fn=render_fn,
                render_msg_kwargs=["test_df"],
                name=subset,
                user_expertise_form=self._create_user_expertise_form(),
                initial_shared_state=[
                    (
                        "Test examples to label",
                        DisplayElement(
                            input_type="dataframe",
                            value=test_df[["id", "input"]],
                            hide_index=True,
                        ),
                    ),
                ],
                test_df=test_df,
            )
            specs[ix] = spec
        return specs

    def _render_fixed_task_explanation(self):
        """Render the fixed task explanation for codebook LLM."""
        FIXED_INSTRUCTIONS = """
Text classification is the task of assigning a label to a text snippet. For example, an annotator might be trained to classify tweets into one of the following labels: "safe", "unsafe."

### What you need to prompt the assistant to do
Your task is to get the assistant to **classify all of the text inputs in a dataset.** On the next page, you will see the details about possible labels for each input, as well as some labeled examples.

You will be scored based on the accuracy of the assistant's classifications. The minimum score is 0\%, and the maximum is 100\%. The better your classifications, the higher your score.

The key to your success will be giving clear instructions to the assistant. 

### Tools
In addition to the labeled examples on the next page, we will provide you a magic tool which gives you the correct label for any given text snippet. To access this tool, see the lower right corner of the screen.
"""
        st.markdown(FIXED_INSTRUCTIONS)


def render_fn(msg: str, test_df: pd.DataFrame):
    js, start_end = parse_json(msg, return_start_end=True)
    if js is None:
        st.markdown(msg)
        return
    df = pd.DataFrame.from_dict(js, orient="index", columns=["predicted_label"])
    df["id"] = df.index.astype(str)
    df = df.merge(test_df[["id", "input"]], on="id", how="left")
    df = df[["id", "input", "predicted_label"]]

    st.markdown(msg[: start_end[0]])
    st.dataframe(df, hide_index=True)
    st.markdown(msg[start_end[1] :])


def validity_fn(
    yhat: str, test_df: pd.DataFrame, raise_errors: bool = False
) -> bool:
    js = parse_json(yhat)
    if js is None:
        if raise_errors:
            raise Exception(
                "Could not parse the output. Make sure it is a valid JSON object."
            )
        return False, {}

    predicted_ids = set(js.keys())
    true_ids = set(test_df["id"].tolist())

    if not predicted_ids == true_ids:
        if raise_errors and len(predicted_ids) != len(true_ids):
            raise Exception(
                f"The predicted labels have {len(predicted_ids)} ids; expected {len(true_ids)} ids"
            )
        elif raise_errors:
            raise Exception(
                "The predicted labels have IDs that do not match the test ids"
            )
        return False, {
            "error": "The predicted labels have IDs that do not match the test ids"
        }

    return True, {}


def reward_fn(
    yhat: str,
    test_df: pd.DataFrame,
    raise_errors: bool = False,
) -> Tuple[float, dict]:
    js = parse_json(yhat)
    if js is None:
        if raise_errors:
            raise Exception(
                "Could not parse the output. Make sure it is a valid JSON object."
            )
        return float("-inf"), {"error": "Could not parse the output"}

    expanded_df = test_df.merge(pd.DataFrame(js.items(), columns=["id", "predicted_label"]), on="id", how="left")
    expanded_df["predicted_label"] = expanded_df["predicted_label"].fillna("UNKNOWN")
    expanded_df.set_index("id", inplace=True)
    score = (expanded_df['label'].str.strip().str.upper() == expanded_df['predicted_label'].str.strip().str.upper()).mean()
    return score, {"outputs": js}


def fmt_examples(examples: Dict[str, str], show_output: bool = True) -> str:
    very_long_covariates = any(
        (len(input) > 1000 or "\n" in input) for input in list(examples.keys())
    )
    fmt_x = lambda i, x: (
        f'{i + 1}. "{x}"\n'
        if not very_long_covariates
        else add_section(f"Covariate {i + 1}", x, style="code")
    )
    return "\n\n".join(
        [
            fmt_x(i, input) + (f"\nlabel: {output.strip()}" if show_output else "")
            for i, (input, output) in enumerate(examples.items())
        ]
    )
