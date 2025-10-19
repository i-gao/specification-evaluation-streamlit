from typing import List, Tuple, Dict, Optional, Callable, Any, Iterator
from langchain_core.tools import tool
from langchain_core.tools.structured import StructuredTool
import os
from pathlib import Path

from data.actions import Action
from utils.misc import (
    download_file_from_google_drive,
    build_docker_image,
    check_docker_image_exists,
)
from utils.streamlit_types import FormElement, DisplayElement
import streamlit as st

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)  # data/ -> specification-benchmark/


class Specification:
    """
    A Specification object represents a task to solve.
    """

    def __init__(
        self,
        dataset_name: str,
        index: str,
        render_task_explanation: Optional[Callable[[], None]] = None,
        initial_specification: str = None,
        initial_specification_multimodal: List[DisplayElement] = None,
        commonsense_description: Optional[str] = None,
        actions: Optional[List[Action]] = None,
        msg_fmt_instructions: Optional[str] = None,
        prediction_fmt_instructions: Optional[str] = None,
        render_msg_fn: Optional[Callable[[str], None]] = None,
        render_msg_kwargs: Optional[List[str]] = None,
        name: Optional[str] = None,
        state_files: Optional[List[str]] = None,
        files_to_clean: Optional[List[str]] = None,
        container_ids: Optional[List[str]] = None,
        user_expertise_form: List[FormElement] = None,
        initial_shared_state: List[Tuple[str, DisplayElement]] = None,
        **kwargs: Any,
    ) -> None:
        """
        logistical args:
            render_task_explanation: Callable[[], None] = a function that renders the task explanation using Streamlit
            name: str = unique identifier for this task
            state_files: List[str] = a list of files that represent the task state
            files_to_clean: List[str] = a list of files to clean up when the specification is deleted
            container_ids: List[str] = a list of container ids to clean up when the specification is deleted
            y0: str = the baseline output for the task
            ystar: str = the ground truth output for the task
            user_expertise_form: List[FormElement] = FormElements which asks the user to assess their domain expertise

        execution args:
            actions: List[Action] = a list of Action objects associated with the task
                The validity_fn and reward_fn tools will be auto-added to this list
            render_msg_fn: Callable[[str], None] = a function that converts a message to a Streamlit component for user display
            render_msg_kwargs: List[str] = a list of kwargs to pass to the render_msg_fn function
        """
        self.dataset_name = dataset_name
        self.index = index

        # input validation
        for action in actions:
            assert isinstance(action, Action), (
                "All actions must be instances of the Action class"
            )
            assert isinstance(action.fn, StructuredTool), (
                "All action fns must be StructuredTools (from langchain), e.g. functions wrapped with the @tool decorator"
            )

        # save attributes
        self.commonsense_description = commonsense_description
        self.render_task_explanation = render_task_explanation
        self.initial_specification = initial_specification
        self.current_specification = initial_specification
        self.initial_specification_multimodal = initial_specification_multimodal
        self.user_expertise_form = user_expertise_form
        self.initial_shared_state = initial_shared_state

        self._actions = actions
        self.msg_fmt_instructions = msg_fmt_instructions
        self.prediction_fmt_instructions = prediction_fmt_instructions
        self._render_msg_fn = render_msg_fn
        self._render_msg_kwargs = render_msg_kwargs

        self.name = name
        self.state_files = state_files
        self.files_to_clean = files_to_clean
        self.container_ids = container_ids

        # common evaluation attributes (may be set by subclasses)
        self._validity_fn = getattr(self, "_validity_fn", None)
        self._validity_kwargs = getattr(self, "_validity_kwargs", None)
        self._validity_fn_tool_name = getattr(self, "_validity_fn_tool_name", None)
        self._validity_fn_tool_description = getattr(
            self, "_validity_fn_tool_description", None
        )
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return f"Specification(name={self.name})"

    def render_msg_fn(self, msg: str) -> None:
        """
        Returns the rendered message in a Streamlit component.
        """
        if self._render_msg_fn is None:
            st.write(msg)
        kwargs = (
            {k: getattr(self, k, None) for k in self._render_msg_kwargs}
            if self._render_msg_kwargs is not None
            else {}
        )
        self._render_msg_fn(msg, **kwargs)

    ################ evaluation ################

    def contains_solution(self, msg: str) -> bool:
        """
        Returns True if the message contains a solution to the task.
        Subclasses can override this method to provide a more specific implementation.
        """
        try:
            self.validity_fn(msg, raise_errors=True)
        except Exception as e:
            if "could not parse" in str(e).lower():
                return False
        return True

    def validity_fn(self, yhat: str, raise_errors: bool = False) -> Tuple[bool, dict]:
        """
        Returns the validity of the output yhat.
        """
        if self._validity_fn is None:
            return True, {}
        return self._validity_fn(
            yhat, raise_errors=raise_errors, **(self._validity_kwargs or {})
        )

    @property
    def validity_action(self) -> Action:
        """
        Returns the validity action.
        """
        if self._validity_fn is None:
            return None

        @tool(
            self._validity_fn_tool_name,
            parse_docstring=True,
            description=self._validity_fn_tool_description,
        )
        def check_if_solution_is_valid(solution_attempt: str) -> bool:
            """
            Checks if the given solution attempt is valid.
            Returns False if the solution attempt violates some key constraints.

            Args:
                solution_attempt (str): The solution attempt to check.
            """
            return self.validity_fn(solution_attempt, raise_errors=True)[0]

        return Action(
            fn=check_if_solution_is_valid,
            is_public=False,
            is_human=False,
            name="Validate solution",
        )

    ################ tools ################

    @property
    def actions(self) -> List[Action]:
        """
        Returns all actions associated with the task.
        """
        actions = self._actions + [self.validity_action]
        return [a for a in actions if a is not None]

    @property
    def all_tools(self) -> List[Action]:
        """
        Returns all actions associated with the task.
        """
        return [action.fn for action in self.actions]

    @property
    def public_tools(self) -> List[Action]:
        """
        Returns all public actions associated with the task.
        """
        return [action.fn for action in self.actions if action.is_public]

    @property
    def private_tools(self) -> List[Action]:
        """
        Returns all private actions associated with the task.
        """
        return [action.fn for action in self.actions if not action.is_public]

    ################ state ################

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the tool states for the task.
        """
        if self.state_files is None:
            return {}

        def _read(path):
            if not os.path.exists(path):
                return None
            return open(path, "r").read()

        return {
            "file_contents": [_read(f) for f in self.state_files],
            "filenames": self.state_files,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load the tool states for the task.
        """
        self.state_files = state["filenames"]
        for f, contents in zip(self.state_files, state["file_contents"]):
            if contents is None:
                continue
            with open(f, "w") as f:
                f.write(contents)

    def __del__(self) -> None:
        # Clean up files and container ids
        if hasattr(self, "files_to_clean") and self.files_to_clean is not None:
            for file in self.files_to_clean:
                try:
                    os.remove(file)
                except Exception:
                    pass
        if hasattr(self, "container_ids") and self.container_ids is not None:
            from llm_sandbox import SandboxSession

            for container_id in self.container_ids:
                try:
                    SandboxSession(container_id=container_id).close()
                except Exception:
                    pass


class CustomSpecification(Specification):
    """
    A CustomSpecification is a Specification where the task parameters are provided by a user,
    either explicitly or implicitly.
    The evaluation is also done by the user by rating the solution compared to a baseline.
    """

    def __init__(
        self,
        user_specification_form_initial: List[FormElement] = None,
        user_specification_form_final: List[FormElement] = None,
        user_specification_callback: Optional[
            Callable[[List[FormElement], dict], dict]
        ] = None,
        user_specification_callback_kwargs: Optional[List[str]] = None,
        validity_fn: Optional[Callable[[str], Tuple[bool, dict]]] = None,
        validity_kwargs: Optional[Dict[str, Any]] = None,
        validity_fn_tool_name: Optional[str] = None,
        validity_fn_tool_description: Optional[str] = None,
        y0: Optional[str] = None,
        render_comparison_fn: Optional[Callable[[str, str], None]] = None,
        render_evaluation_fn: Optional[Callable[..., Tuple[bool, dict]]] = None,
        render_evaluation_kwargs: Optional[Dict[str, Any]] = None,
        final_eval_likert_label: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        specification args:
            user_specification_form_initial: List[FormElement] = FormElements which asks the user to specify the task
            user_specification_form_final: List[FormElement] = FormElements which asks the user to specify the task
            user_specification_callback: Callable[[List[FormElement], dict, dict], dict] = a function that takes the result of the user_specification_form,
                the callback kwargs, and returns an updated dict of attributes (probably validity_kwargs and y0)
            user_specification_callback_kwargs: List[str] = names of additional attributes that should be passed to the user_specification_callback

        evaluation args:
            validity_fn: Callable[[str], Tuple[bool, dict]] = a function that checks if the output is valid.
                This should take a raise_errors flag and raise an error if raise_errors is True and the output is invalid.
            validity_kwargs: Dict[str, Any] = additional arguments for the validity function
            validity_fn_tool_name: str = name for the evaluation validity tool
            validity_fn_tool_description: str = description for the evaluation validity tool
            y0: str = the baseline output for the task
            render_comparison_fn: Callable[[str, str], None] = a function that converts a comparison to a Streamlit component for user display
                Will also be passed the validity_fn and validity_kwargs and render_msg_kwargs
            render_evaluation_fn: Callable[..., Tuple[bool, dict]] = a function that converts a final prediction to a Streamlit component for user display
                Will also be passed the validity_fn and validity_kwargs and render_msg_kwargs
            render_evaluation_kwargs: Dict[str, Any] = additional arguments for the render_evaluation function
        """
        super().__init__(**kwargs)

        # save attributes
        self.user_specification_form_initial = user_specification_form_initial
        self.user_specification_form_final = user_specification_form_final
        self._user_specification_callback = user_specification_callback
        self._user_specification_callback_kwargs = user_specification_callback_kwargs
        self._validity_fn = validity_fn
        self._validity_kwargs = validity_kwargs
        self._validity_fn_tool_name = validity_fn_tool_name
        self._validity_fn_tool_description = validity_fn_tool_description
        self.y0 = y0
        self._render_comparison_fn = render_comparison_fn
        self._render_evaluation_fn = render_evaluation_fn
        self._render_evaluation_kwargs = render_evaluation_kwargs
        self.final_eval_likert_label = final_eval_likert_label

    ################ functions ################

    def user_specification_callback(self, form_results: dict):
        """
        Runs the user specification callback and updates the specification attributes.
        """
        if self._user_specification_callback is None:
            return
        if self._user_specification_callback_kwargs is not None:
            callback_kwargs = {
                k: getattr(self, k) for k in self._user_specification_callback_kwargs
            }
        else:
            callback_kwargs = {}
        updates = self._user_specification_callback(form_results, callback_kwargs)
        if updates:
            for attr_name, attr_value in updates.items():
                setattr(self, attr_name, attr_value)

    def render_comparison_fn(self, y1: str, y2: str) -> None:
        """
        Renders the comparison in a Streamlit component.
        """
        if self._render_comparison_fn is None:
            st.write(f"**Plan A:**\n{y1}\n\n**Plan B:**\n{y2}")
        kwargs = (
            {k: getattr(self, k, None) for k in self._render_msg_kwargs}
            if self._render_msg_kwargs is not None
            else {}
        )
        self._render_comparison_fn(
            y1=y1,
            y2=y2,
            **kwargs,
            validity_fn=self._validity_fn,
            validity_kwargs=self._validity_kwargs,
        )

    def render_evaluation(self, final_prediction: str) -> Tuple[bool, dict]:
        """
        Delegates evaluation rendering to the dataset-provided function.
        Returns (completed, feedback).
        """
        if self._render_evaluation_fn is None:
            raise ValueError("render_evaluation_fn is not set for this specification")
        return self._render_evaluation_fn(
            final_prediction=final_prediction, **(self._render_evaluation_kwargs or {})
        )

    ################ tools ################

    @property
    def actions(self) -> List[Action]:
        """
        Returns all actions associated with the task.
        """
        actions = self._actions + [self.validity_action]
        return [a for a in actions if a is not None]


class FixedSpecification(Specification):
    """
    A FixedSpecification is a Specification where the task parameters are fixed, and the evaluation
    is computed automatically.
    """

    def __init__(
        self,
        full_specification: str,
        validity_fn: Optional[Callable[[str], Tuple[bool, dict]]] = None,
        validity_kwargs: Optional[Dict[str, Any]] = None,
        validity_fn_tool_name: Optional[str] = None,
        validity_fn_tool_description: Optional[str] = None,
        reward_fn: Optional[Callable[[str], Tuple[float, dict]]] = None,
        reward_kwargs: Optional[Dict[str, Any]] = None,
        reward_fn_tool_name: Optional[str] = None,
        reward_fn_tool_description: Optional[str] = None,
        ystar: Optional[str] = None,
        metric_name: Optional[str] = None,
        baseline_scores: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        """
        specification args:
            full_specification: str = the complete natural language description of the task

        evaluation args:
            validity_fn: Callable[[str], Tuple[bool, dict]] = a function that checks if the output is valid.
                This should take a raise_errors flag and raise an error if raise_errors is True and the output is invalid.
            validity_kwargs: Dict[str, Any] = additional arguments for the validity function
            validity_fn_tool_name: str = name for the evaluation validity tool
            validity_fn_tool_description: str = description for the evaluation validity tool
            reward_fn: Callable[[str], Tuple[float, dict]] = a function that evaluates the reward function R(y) that scores outputs
                This should take a raise_errors flag and raise an error if raise_errors is True and the output is invalid.
            reward_kwargs: Dict[str, Any] = additional arguments for the reward function
            reward_fn_tool_name: str = name for the evaluation reward tool
            reward_fn_tool_description: str = description for the evaluation reward tool
            ystar: str = the ground truth output for the task
            metric_name: str = name of the metric to use for evaluation
            baseline_scores: List[float] = a list of baseline scores for the task
        """
        super().__init__(**kwargs)

        # save attributes
        self.full_specification = full_specification
        self._validity_fn = validity_fn
        self._validity_kwargs = validity_kwargs
        self._validity_fn_tool_name = validity_fn_tool_name
        self._validity_fn_tool_description = validity_fn_tool_description
        self._reward_fn = reward_fn
        self._reward_kwargs = reward_kwargs
        self._reward_fn_tool_name = reward_fn_tool_name
        self._reward_fn_tool_description = reward_fn_tool_description
        self.ystar = ystar
        self.metric_name = metric_name
        self.baseline_scores = baseline_scores

    ################ functions ################

    def reward_fn(self, yhat: str, raise_errors: bool = False) -> Tuple[float, dict]:
        """
        Returns the reward for the given solution attempt.
        """
        if self._reward_fn is None:
            raise ValueError("Reward function is not set")

        is_valid, validity_metadata = self.validity_fn(yhat, raise_errors=raise_errors)
        if not is_valid:
            return float("-inf"), {}
        score, reward_metadata = self._reward_fn(
            yhat, raise_errors=raise_errors, **self._reward_kwargs
        )
        return score, {
            "validity_metadata": validity_metadata,
            "reward_metadata": reward_metadata,
        }

    @property
    def reward_action(self) -> Action:
        """
        Returns the reward action.
        """
        if self._reward_fn is None:
            return None

        @tool(
            self._reward_fn_tool_name,
            parse_docstring=True,
            description=self._reward_fn_tool_description,
        )
        def score_solution(solution_attempt: str) -> Tuple[float, dict]:
            """
            Scores the given solution attempt between [0, 100], where increasing values are better.

            Args:
                solution_attempt (str): The solution attempt to score.
            """
            return self.reward_fn(solution_attempt, raise_errors=True)[0]

        return Action(
            fn=score_solution, is_public=False, is_human=False, name="Score solution"
        )

    ################ tools ################

    @property
    def actions(self) -> List[Action]:
        """
        Returns all actions associated with the task.
        """
        actions = self._actions + [self.validity_action, self.reward_action]
        return [a for a in actions if a is not None]


#########################################################

# Asset download settings
DOWNLOAD_SETTINGS = {
    "chunk_size": 8192,
    "timeout": 600,  # 10 minutes
}


class SpecificationCollection:
    """Uses lazy loading by default. Call load_specs() to load all specs at once."""

    def __init__(
        self,
        dev: bool = False,
        skip_docker_check: bool = False,
        **kwargs: Any,
    ) -> None:
        # Check and download assets before loading dataset
        self._ensure_assets_available()
        if not skip_docker_check:
            self._ensure_docker_images_available()
        self.dev = dev

    @property
    def dataset_name(self) -> str:
        """
        Returns the programmatic name of the dataset (e.g. workout_planning)
        """
        raise NotImplementedError

    @property
    def dataset_pretty_name(self) -> str:
        """
        Returns the pretty name of the dataset (e.g. Workout Planning)
        """
        raise NotImplementedError

    @property
    def dataset_description(self) -> str:
        """
        Returns a short description of the dataset (e.g. The Workout Planning benchmark evaluates how well LMs can generate personalized workout plans which obey some constraints.)
        """
        raise NotImplementedError

    @property
    def task_names(self) -> List[str]:
        return [
            spec.name if spec.name is not None else f"fixed_task_{i + 1}"
            for i, spec in self.fixed_specs.items()
        ] + [
            spec.name if spec.name is not None else f"custom_task_{i + 1}"
            for i, spec in self.custom_specs.items()
        ]

    @property
    def assets_file_id(self) -> str:
        return None

    @property
    def default_docker_images(self) -> List[Dict[str, str]]:
        return None

    def __repr__(self) -> str:
        return f"SpecificationCollection(name={self.dataset_name}, dev={self.dev}, fixed_specs={len(self.fixed_specs)}, custom_specs={len(self.custom_specs)})"

    def __del__(self) -> None:
        for spec in self.fixed_specs.values():
            if hasattr(spec, "__del__"):
                spec.__del__()
        for spec in self.custom_specs.values():
            if hasattr(spec, "__del__"):
                spec.__del__()

    def _finish_init(self) -> None:
        """Check if the required attributes are set"""
        for attr in ["dev", "fixed_length", "custom_length"]:
            if getattr(self, attr) is None:
                raise ValueError(f"{attr} is not set")
        self.fixed_specs = {i: None for i in range(self.fixed_length)}
        self.custom_specs = {i: None for i in range(self.custom_length)}

    def load_fixed_specs(
        self, indexes: Optional[List[int]] = None, reload: bool = False
    ):
        if indexes is None:
            print(f"Loading all {self.fixed_length} specs")
            indexes = list(range(self.fixed_length))

        if any(i not in self.fixed_specs for i in indexes):
            raise ValueError(f"Indexes {indexes} not found in dataset")

        if not reload:
            # remove already loaded specs
            indexes = [i for i in indexes if self.fixed_specs[i] is None]
        self.fixed_specs.update(self._load_fixed_specs(indexes=indexes))

    def load_custom_specs(
        self, indexes: Optional[List[int]] = None, reload: bool = False
    ):
        if indexes is None:
            print(f"Loading all {self.custom_length} specs")
            indexes = list(range(self.custom_length))

        if any(i not in self.custom_specs for i in indexes):
            raise ValueError(f"Indexes {indexes} not found in dataset")

        if not reload:
            # remove already loaded specs
            indexes = [i for i in indexes if self.custom_specs[i] is None]
        self.custom_specs.update(self._load_custom_specs(indexes=indexes))

    def _load_fixed_specs(self, **kwargs: Any) -> Dict[int, FixedSpecification]:
        raise NotImplementedError

    def _load_custom_specs(self, **kwargs: Any) -> Dict[int, CustomSpecification]:
        raise NotImplementedError

    def _ensure_assets_available(self) -> None:
        """Ensure dataset assets are downloaded and available"""

        if self.assets_file_id is None:
            # no assets for the dataset
            return

        # Check if the assets are already downloaded
        assets_dir = Path(ROOT_DIR) / "data" / self.dataset_name / "assets"
        if assets_dir.exists() and any(assets_dir.iterdir()):
            return

        # Otherwise, download
        try:
            print(f"Downloading assets for {self.dataset_name}")
            download_file_from_google_drive(
                self.assets_file_id, str(assets_dir), unzip=True, **DOWNLOAD_SETTINGS
            )
        except Exception as e:
            print(f"Download failed for {self.dataset_name}: {e}")
            raise

    def _ensure_docker_images_available(self) -> None:
        """Ensure required Docker images are available"""
        if self.default_docker_images is None:
            return
        for docker_info in self.default_docker_images:
            try:
                if not check_docker_image_exists(docker_info["image_name"]):
                    print(f"Building docker image {docker_info['image_name']}")
                    build_docker_image(docker_info, root_dir=ROOT_DIR)
            except ImportError:
                print("Subprocess not available, skipping Docker check")
            except Exception as e:
                print(f"Failed to ensure Docker images available: {e}")
                raise

    def __len__(self) -> int:
        return self.fixed_length + self.custom_length

    def __getitem__(self, key: str, load_on_demand: bool = True) -> Specification:
        key = str(key)
        assert key.startswith("fixed_") or key.startswith("custom_"), (
            "Key must start with 'fixed_' or 'custom_'"
        )
        ix = int(key.split("_")[1])

        if key.startswith("fixed_"):
            assert ix < self.fixed_length, (
                f"Index {ix} is out of bounds for fixed specs"
            )
            spec = self.fixed_specs.get(ix, None)
            if spec is None and load_on_demand:
                self.load_fixed_specs(indexes=[ix])
                spec = self.fixed_specs[ix]
        elif key.startswith("custom_"):
            assert ix < self.custom_length, (
                f"Index {ix} is out of bounds for custom specs"
            )
            spec = self.custom_specs.get(ix, None)
            if spec is None and load_on_demand:
                self.load_custom_specs(indexes=[ix])
                spec = self.custom_specs[ix]
        return spec

    def __iter__(self) -> Iterator[Specification]:
        return iter(list(self.fixed_specs.values()) + list(self.custom_specs.values()))
