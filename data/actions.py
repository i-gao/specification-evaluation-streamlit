from dataclasses import dataclass
from typing import Callable, Tuple
from langchain_core.tools.structured import StructuredTool
from langchain_core.tools import tool
from typing import Dict, Any, List
from utils.misc import add_section
import re  # noqa: F401
import time  # noqa: F401
import numpy as np
import uuid
from utils.code_sandbox import reset_jupyter_session, run_jupyter_script
import os
import lotus
from lotus.models import LM
import pandas as pd
from utils.model import init_langchain_model


@dataclass
class Action:
    fn: StructuredTool  # specifically, a function wrapped with the @tool decorator
    name: str  # a pretty name for the action, used for display in the UI
    is_public: bool = False  # models can call public, but not private actions
    is_human: bool = False  # we include extra actions for the simulator, but don't show them to the user


############# Common actions #############


def get_classification_actions(
    train_data: Dict[str, Any],
    test_data: Dict[str, Any],
    fuzzy_match_x: Callable[[str, str], bool] = None,
    fuzzy_match_x_error_msg: str = "Could not ascertain the correct annotation for the input. Make sure your input string is exactly the same as the covariate you're interested in.",
    fuzzy_match_y: Callable[[str, str], bool] = None,
    override_tool_names: Dict[str, str] = {},
    override_tool_descriptions: Dict[str, str] = {},
    return_as_df: bool = False,
) -> List[Action]:
    """
    Get the actions for a classification task.

    Args:
        train_data (Dict[str, Any]): The training dataset, where the keys are the inputs and the values are the expected outputs
        test_data (Dict[str, Any]): The test dataset, where the keys are the inputs and the values are the expected outputs
        fuzzy_match_x (Callable[[str, str], bool]): A function that takes two strings and returns True if they are a fuzzy match, False otherwise.
            If None, no fuzzy matching will be done.
            The fuzzy match operation is applied to the covariates.
        fuzzy_match_y (Callable[[str, str], bool]): A function that takes two strings and returns True if they are a fuzzy match, False otherwise.
            If None, no fuzzy matching will be done.
            The fuzzy match operation is applied to the expected outputs.
        fuzzy_match_error_msg (str): The error message to return if the fuzzy match fails.
        override_tool_names (Dict[str, str]): A dictionary of tool names to override.
        override_tool_descriptions (Dict[str, str]): A dictionary of tool descriptions to override.
        return_as_df (bool): Whether to return the action results as a pandas DataFrame. If False, will return a formatted string.
    
    Returns:
        List[Action]: The actions for the classification task
    """
    train_data = {str(k): v for k, v in train_data.items()}
    test_data = {str(k): v for k, v in test_data.items()}

    # Create fmt function for covariates
    very_long_covariates = any(
        (len(str(input)) > 1000 or "\n" in str(input))
        for input in list(train_data.keys()) + list(test_data.keys())
    )
    if not very_long_covariates:
        sep = "\n"

        def fmt_x(i: int, x: str) -> str:
            return f'{i + 1}. "{x}"\n'

    else:
        sep = "\n\n"

        def fmt_x(i: int, x: str) -> str:
            return add_section(
                f"Example {i + 1}",
                x,
                style="code",
            )

    def _get_correct_annotation(input: str) -> str:
        if not isinstance(input, str):
            input = str(input)

        # First try exact match
        if input in train_data:
            return train_data[input]
        if input in test_data:
            return test_data[input]

        # Fall back to fuzzy matching
        if fuzzy_match_x is not None:
            for key in train_data.keys():
                if fuzzy_match_x(input, key):
                    return train_data[key]
            for key in test_data.keys():
                if fuzzy_match_x(input, key):
                    return test_data[key]

        raise Exception(fuzzy_match_x_error_msg)

    ######################################################

    @tool(
        override_tool_names.get("get_documents_to_annotate"),
        parse_docstring=True,  # Fallback on docstring
        description=override_tool_descriptions.get("get_documents_to_annotate"),
    )
    def get_documents_to_annotate() -> str:
        """
        Returns the test set of documents that are interesting to annotate.
        This only returns the documents, not the correct annotations.
        Warning: this may return 1000s of documents at once, so use it sparingly.
        """
        if return_as_df:
            return pd.DataFrame({"input": test_data.keys()})
        else:
            return sep.join([fmt_x(i, input) for i, input in enumerate(test_data.keys())])

    @tool(
        override_tool_names.get("sample_documents_to_annotate"),
        parse_docstring=True,  # Fallback on docstring
        description=override_tool_descriptions.get("sample_documents_to_annotate"),
    )
    def sample_documents_to_annotate(n: int = 1) -> str:
        """
        Returns a random sample of documents that are interesting to annotate.
        This only returns the documents, not the correct annotations.

        Args:
            n (int): The number of documents to sample.
        """
        sample = np.random.choice(list(test_data.keys()), size=n, replace=False)
        if return_as_df:
            return pd.DataFrame({"input": sample})
        else:
            return sep.join([fmt_x(i, input) for i, input in enumerate(sample)])

    @tool(
        override_tool_names.get("get_correct_annotation"),
        parse_docstring=True,  # Fallback on docstring
        description=override_tool_descriptions.get("get_correct_annotation"),
    )
    def get_correct_annotation(input: str) -> str:
        """
        Returns the ground truth annotation for the given input text.

        Args:
            input (str): The input text. This must EXACTLY match the full document of interest.
        """
        out = _get_correct_annotation(input)
        if return_as_df:
            return pd.DataFrame({"input": [input], "output": [out]})
        else:
            return out

    @tool(
        override_tool_names.get("check_single_annotation"),
        parse_docstring=True,  # Fallback on docstring
        description=override_tool_descriptions.get("check_single_annotation"),
    )
    def check_single_annotation(input: str, predicted_output: str) -> str:
        """
        Checks if the given predicted output is correct for the given input text.

        Args:
            input (str): The input text. This must EXACTLY match the full document of interest.
            predicted_output (str): The predicted output to check.
        """
        correct_annotation = _get_correct_annotation(input)
        if fuzzy_match_y is not None:
            if fuzzy_match_y(predicted_output, correct_annotation):
                return "The predicted output is correct for this input."
            else:
                return "The predicted output is incorrect for this input."
        else:
            if predicted_output == correct_annotation:
                return "The predicted output is correct for this input."
            else:
                return "The predicted output is incorrect for this input."

    return [
        Action(
            fn=get_documents_to_annotate,
            name=f"View all {len(test_data)} documents to annotate",
            is_public=True,
            is_human=True,
        ),
        Action(
            fn=sample_documents_to_annotate,
            is_public=True,
            is_human=False,
            name="View sample documents to annotate",
        ),
        Action(
            fn=check_single_annotation,
            is_public=False,
            name="Check annotation",
            is_human=False,
        ),
        Action(
            fn=get_correct_annotation,
            is_public=False,
            name="Get correct annotation",
            is_human=True,
        ),
    ]


def get_annotator_output_action(
    models: List[str],
    anonymize_model_names: bool = True,
    override_tool_names: Dict[str, str] = {},
    override_tool_descriptions: Dict[str, str] = {},
) -> List[Action]:
    """
    Returns an action for calling a language model with a prompt on an input

    Args:
        models (List[str]): The models to use.
        anonymize_model_names (bool): Whether to anonymize the model names.
    """

    @tool(
        override_tool_names.get("get_annotator_output"),
        parse_docstring=True,  # Fallback on docstring
        description=override_tool_descriptions.get("get_annotator_output"),
    )
    def get_annotator_output(annotation_instructions: str, input: str) -> str:
        """
        Collects annotations for the given input from a set of annotators
        prompted with the annotation_instructions.

        Args:
            annotation_instructions (str): The instructions for the annotators.
            input (str): The input to annotate.
        """
        outputs = {}
        for i, model in enumerate(models):
            lotus.settings.configure(lm=LM(model=model, temperature=0.0))
            df = pd.DataFrame({"input": [input]}).sem_map(
                annotation_instructions + "\n\n{input}"
            )
            k = f"annotator_{i}" if anonymize_model_names else model
            outputs[k] = df["_map"].iloc[0]

        return outputs

    return [
        Action(
            fn=get_annotator_output,
            is_public=True,
            is_human=True,
            name="Get annotation based on instructions",
        )
    ]


def get_jupyter_actions(
    docker_image: str,
    ls_output: List[dict],
    root_dir: str = "",
    docker_container_id: str = None,
) -> Tuple[str, List[Action]]:
    """
    Get the actions for a jupyter notebook.

    Args:
        docker_image (str): The docker image to use.
        ls_output (List[dict]): The output of the ls command.

    Returns:
        A tuple of the id of the session and the actions.
    """
    id_str = str(uuid.uuid4())
    # Place the buffer file inside root_dir for consistent path resolution in local mode;
    # in docker mode, it will be copied into /sandbox by the runner
    if docker_image is not None:
        root_dir = "/sandbox"
    filename = (
        os.path.join(root_dir, f"_{id_str}.txt") if (root_dir not in [None, ""]) else os.path.join(os.getcwd(), f"_{id_str}.txt")
    )

    reset_jupyter_session(filename)

    @tool(parse_docstring=True)
    def ls_files() -> str:
        """
        List the available files in the environment,
        along with a detailed description of their contents.
        For example, CSV file descriptions will also include a codebook of the columns.
        """
        return ls_output

    @tool(parse_docstring=True)
    def reset_notebook() -> str:
        """
        Reset the Jupyter notebook. This will clear all previous calls to this tool.
        """
        print("Resetting Jupyter notebook")
        reset_jupyter_session(filename)

    @tool(parse_docstring=True)
    def run_python_code(code: str) -> str:
        """
        Add a new cell to the Jupyter notebook and execute the cell.
        This tool behaves like you are adding a new cell to a continuous Jupyter notebook.
        Calls to this tool will depend on previous calls to this tool.
        WARNING: do not write any string literals in your code that contain newlines, even escaped newlines.
        Instead, to print hello newline world,write:
        print("hello")
        print("world")
        String literals with newlines will cause the cell to fail.

        Note: writing files is prohibited. The environment is read-only.

        Args:
            code (str): The Python code to run.
        """
        print("Running Python code")
        return run_jupyter_script(
            filename=filename,
            cell_code=code,
            docker_image=docker_image,
            docker_container_id=docker_container_id,
            root_dir=root_dir,
        )

    return filename, [
        Action(
            fn=ls_files,
            is_public=True,
            is_human=False,
            name="List files in environment",
        ),
        Action(
            fn=reset_notebook,
            is_public=True,
            is_human=False,
            name="Reset Jupyter notebook",
        ),
        Action(
            fn=run_python_code, is_public=True, is_human=False, name="Run Python code"
        ),
    ]


def get_query_theta_action(theta: str, retrieval_method="bm25") -> Action:
    """
    Retrieve chunks of the full specification of the task.

    Args:
        theta (str): The theta of the task.
    """
    from rank_bm25 import BM25Okapi

    # Split the theta into chunks
    chunks = [c.strip() for c in theta.split("<chunk>") if c.strip() != ""]

    if retrieval_method == "bm25":
        def retrieval_fn(query: str, documents: List[str]):
            return BM25Okapi(documents).get_top_n(query, documents, n=1)
    else:
        qa_model = init_langchain_model(retrieval_method)
        PROMPT = """Return relevant EXACT QUOTES from the context that answer the query:
-----------
CONTEXT
{context}

-----------
Query: {query}

Answer: """
        def retrieval_fn(query: str, documents: List[str]):
            return qa_model.invoke(
                PROMPT.format(context="\n".join(documents), query=query)
            ).content

    @tool(parse_docstring=True)
    def consult_full_task_specification(query: str) -> str:
        """
        If parts of the problem are underspecified, you can consult the full specification of the task
        using this function.

        Args:
            query (str): The query to consult the full specification for, e.g. "What is the input format?"
        """
        return retrieval_fn(query, documents=chunks)

    return Action(
        fn=consult_full_task_specification,
        is_public=False,
        is_human=True,
        name="Consult full task specification",
    )
