from typing import List, Tuple, Dict, Optional, Callable, Any, Iterator
from langchain_core.tools import tool
from langchain_core.tools.structured import StructuredTool
import os
from pathlib import Path

from data.actions import Action
from data.reward import linear_reward, Constraint
from utils.misc import (
    download_file_from_google_drive,
    build_docker_image,
    check_docker_image_exists,
)
from utils.streamlit_types import FormElement, DisplayElement
import streamlit as st
import numpy as np
from llm_sandbox import SandboxSession

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
        parse_solutions_fn: Optional[Callable[[str], List[str]]] = None,
        parse_solutions_and_options_fn: Optional[Callable[[str], List[str]]] = None,
        render_task_explanation: Optional[Callable[[], None]] = None,
        initial_specification: str = None,
        initial_specification_multimodal: List[DisplayElement] = None,
        commonsense_description: Optional[str] = None,
        actions: Optional[List[Action]] = None,
        msg_fmt_instructions: Optional[str] = None,
        prediction_fmt_instructions: Optional[str] = None,
        render_msg_fn: Optional[Callable[[str], None]] = None,
        render_msg_fn_txt: Optional[Callable[[str], None]] = None,
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
            parse_solutions_fn: Callable[[str], List[str]] = a function that parses out complete solutions from the message
            parse_solutions_and_options_fn: Callable[[str], List[str]] = a function that parses out both complete solutions and individual options from the message
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
        _actions = actions or []
        for action in _actions:
            assert isinstance(action, Action), (
                "All actions must be instances of the Action class"
            )
            assert isinstance(action.fn, StructuredTool), (
                "All action fns must be StructuredTools (from langchain), e.g. functions wrapped with the @tool decorator"
            )
        self._parse_solutions_fn = parse_solutions_fn
        self._parse_solutions_and_options_fn = parse_solutions_and_options_fn

        # save attributes
        self.commonsense_description = commonsense_description
        self.render_task_explanation = render_task_explanation
        self.initial_specification = initial_specification
        self.initial_specification_multimodal = initial_specification_multimodal
        self.user_expertise_form = user_expertise_form
        self.initial_shared_state = initial_shared_state

        self._actions = _actions
        self.msg_fmt_instructions = msg_fmt_instructions
        self.prediction_fmt_instructions = prediction_fmt_instructions
        self._render_msg_fn = render_msg_fn
        self._render_msg_fn_txt = render_msg_fn_txt
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
            return
        kwargs = (
            {k: getattr(self, k, None) for k in self._render_msg_kwargs}
            if self._render_msg_kwargs is not None
            else {}
        )
        self._render_msg_fn(msg, **kwargs)

    def render_msg_fn_txt(self, msg: str) -> str:
        """
        Returns the rendered message in a text format.
        """
        if self._render_msg_fn_txt is None:
            return msg
        kwargs = (
            {k: getattr(self, k, None) for k in self._render_msg_kwargs}
            if self._render_msg_kwargs is not None
            else {}
        )
        return self._render_msg_fn_txt(msg, **kwargs)

    def get_current_specification(self) -> str:
        if hasattr(self, "current_specification"):
            return self.current_specification
        raise NotImplementedError(
            "get_current_specification is not implemented for this specification"
        )

    ################ evaluation ################

    def contains_solution(self, msg: str) -> bool:
        """
        Returns True if the message contains a solution to the task.
        Subclasses can override this method to provide a more specific implementation.
        """
        return len(self.parse_solutions(msg)) > 0

    def parse_solutions(self, msg: str) -> List[str]:
        """
        Parses out complete solutions from the message.
        """
        if self._parse_solutions_fn is None:
            try:
                self.validity_fn(msg, raise_errors=True)
            except Exception as e:
                if "could not parse" in str(e).lower():
                    return []
            return [msg]
        return self._parse_solutions_fn(msg)

    def parse_solutions_and_options(self, msg: str) -> List[str]:
        """
        Parses out both complete solutions and individual options from the message.
        Falls back to parse_solutions if parse_solutions_and_options_fn is not provided.
        """
        if self._parse_solutions_and_options_fn is not None:
            return self._parse_solutions_and_options_fn(msg)
        # Fall back to parse_solutions if parse_solutions_and_options_fn is not provided
        return self.parse_solutions(msg)

    def validity_fn(
        self, yhat: str, raise_errors: bool = False, **kwargs
    ) -> Tuple[bool, dict]:
        """
        Returns the validity of the output yhat.
        """
        if self._validity_fn is None:
            return True, {}
        call_kwargs = dict(self._validity_kwargs or {})
        call_kwargs.update(kwargs)
        return self._validity_fn(yhat, raise_errors=raise_errors, **call_kwargs)

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
        render_search_interface_fn: Optional[Callable[..., None]] = None,
        render_search_interface_kwargs: Optional[Dict[str, Any]] = None,
        render_liked_items_fn: Optional[Callable[[set, Any], None]] = None,
        render_liked_items_kwargs: Optional[Dict[str, Any]] = None,
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
            render_search_interface_fn: Callable[..., None] = a function that renders the search interface for exploring items
            render_search_interface_kwargs: Dict[str, Any] = additional arguments for the render_search_interface function
            render_liked_items_fn: Callable[[set, Any], None] = a function that renders liked items in cards for final comparison
                Takes (liked_items: set, db_instance: Any) as arguments
            render_liked_items_kwargs: Dict[str, Any] = additional arguments for the render_liked_items function
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
        self._render_search_interface_fn = render_search_interface_fn
        self._render_search_interface_kwargs = render_search_interface_kwargs
        self._render_liked_items_fn = render_liked_items_fn
        self._render_liked_items_kwargs = render_liked_items_kwargs

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
            return
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

    def render_y0_yhat_evaluation(self, final_prediction: str) -> Tuple[bool, dict]:
        """
        Delegates y0/yhat comparison evaluation rendering to the dataset-provided function.
        Returns (completed, feedback).
        """
        if self._render_evaluation_fn is None:
            raise ValueError("render_evaluation_fn is not set for this specification")
        return self._render_evaluation_fn(
            final_prediction=final_prediction, **(self._render_evaluation_kwargs or {})
        )
    
    def render_evaluation(self, final_prediction: str) -> Tuple[bool, dict]:
        """
        Legacy alias for render_y0_yhat_evaluation for backwards compatibility.
        """
        return self.render_y0_yhat_evaluation(final_prediction)

    def render_search_interface(self, **kwargs) -> None:
        """
        Renders the search interface for exploring items.
        Merges render_search_interface_kwargs with provided kwargs.
        """
        if self._render_search_interface_fn is None:
            return
        merged_kwargs = dict(self._render_search_interface_kwargs or {})
        merged_kwargs.update(kwargs)
        self._render_search_interface_fn(**merged_kwargs)
    
    def render_liked_items(self, liked_items: set, **kwargs) -> None:
        """
        Renders liked items in cards for final comparison.
        Merges render_liked_items_kwargs with provided kwargs.
        The DB instance should already be in render_liked_items_kwargs.
        """
        if self._render_liked_items_fn is None:
            return
        merged_kwargs = dict(self._render_liked_items_kwargs or {})
        merged_kwargs.update(kwargs)
        # Get db_instance from merged_kwargs (should be set when spec is created)
        db_instance = merged_kwargs.get("db_instance")
        if db_instance is None:
            # Try to get from merged_kwargs by common names
            for key in ["catalog", "recipe_db", "travel_db"]:
                if key in merged_kwargs:
                    db_instance = merged_kwargs[key]
                    break
        if db_instance is None:
            # Try to get from spec attributes (e.g., self.db)
            db_instance = getattr(self, "db", None)
        self._render_liked_items_fn(liked_items, db_instance)

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

    def reward_fn(
        self, yhat: str, raise_errors: bool = False, **kwargs
    ) -> Tuple[float, dict]:
        """
        Returns the reward for the given solution attempt.
        """
        if self._reward_fn is None:
            raise ValueError("Reward function is not set")

        is_valid, validity_metadata = self.validity_fn(
            yhat, raise_errors=raise_errors, **kwargs
        )
        if not is_valid:
            return float("-inf"), {
                "validity_metadata": validity_metadata,
                "reward_metadata": {},
            }
        call_kwargs = dict(self._reward_kwargs or {})
        call_kwargs.update(kwargs)
        score, reward_metadata = self._reward_fn(
            yhat, raise_errors=raise_errors, **call_kwargs
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


class LinearFixedSpecification(FixedSpecification):
    """
    A LinearFixedSpecification is a FixedSpecification where the reward function is a linear function of the features.
    """

    def __init__(
        self,
        features: List[Constraint],
        weights: List[float],
        weight_covariance: np.ndarray = None,
        parse_y_fn: Callable[[str, bool], Any] = None,
        lam: float = 1e-2,
        **kwargs: Any,
    ) -> None:
        # reward and validity functions
        kwargs["reward_fn"] = self._linear_reward_fn
        kwargs["validity_fn"] = self._linear_validity_fn

        # If full_specification is not provided, we'll compute it from get_current_specification
        # after initialization. Temporarily remove it from kwargs if present.
        full_spec_provided = kwargs.pop("full_specification", None)
        super().__init__(**kwargs, full_specification="")

        self.weights = weights
        self.features = features
        # Track the current feature set for tool-based evaluations
        # Default to full feature set
        self.reset_feature_set_current_ids(minimal=False)
        if weight_covariance is not None:
            self.weight_covariance = weight_covariance
        else:
            self.weight_covariance = np.eye(len(weights))
        self._parse_y_fn = parse_y_fn
        self.lam = lam

        # Compute full_specification from get_current_specification if not provided
        if full_spec_provided is None:
            # Get all feature IDs for full specification
            all_feature_ids = [c.id for c in self.features]
            self.full_specification = self.get_current_specification(
                feature_set=all_feature_ids
            )
        else:
            self.full_specification = full_spec_provided

    def reset_feature_set_current_ids(self, minimal: bool = False):
        """
        Resets the current feature set to the full feature set or a minimal feature set.
        """
        if minimal:
            self._feature_set_current_ids = [
                c.id for c in self.features if c.is_minimal
            ]
        else:
            self._feature_set_current_ids = [c.id for c in self.features]

    def reveal_features(self, feature_ids: List[str]):
        """
        Reveals the features with the given IDs.
        """
        self._feature_set_current_ids.extend(feature_ids)

    def get_feature_by_id(self, id: str) -> Constraint:
        """
        Returns the feature with the given ID.
        """
        feature = next((f for f in self.features if f.id == id), None)
        if feature is None:
            raise ValueError(f"Feature with ID {id} not found")
        return feature

    def parse_y_fn(self, yhat: str, raise_errors: bool = False) -> Any:
        """
        Parses the solution attempt.
        """
        if self._parse_y_fn is None:
            return yhat
        return self._parse_y_fn(yhat, raise_errors=raise_errors)

    def project_weights(self, feature_set: List[str]) -> Tuple[List[float], List[bool]]:
        """
        Returns the projected weights and active mask for the feature set.
        """
        d = len(self.weights)
        if self.weight_covariance.shape != (d, d):
            raise ValueError(
                "weight_covariance must be a square (d x d) matrix matching len(weights)"
            )

        tokens = set((feature_set or []))
        # Match only on constraint IDs
        active_mask = [c.id in tokens for c in self.features]

        # Create diagonal selection matrix S from mask
        S = np.diag(np.array(active_mask, dtype=float))  # (d, d)
        Sigma = np.asarray(self.weight_covariance, dtype=float)  # (d, d)

        # Compute M_F = (S Σ S + λ I)^(-1) (S Σ)
        # This matches the torch implementation used elsewhere (voi/environment.py)
        inner = S @ (Sigma @ S) + self.lam * np.eye(d)
        rhs = S @ Sigma
        try:
            M_F = np.linalg.solve(inner, rhs)
        except np.linalg.LinAlgError:
            M_F = np.linalg.pinv(inner) @ rhs

        theta = np.asarray(self.weights, dtype=float)
        theta_proj = M_F @ theta  # (d,)
        projected_active_weights = theta_proj[active_mask].tolist()

        return projected_active_weights, active_mask

    def _linear_reward_fn(
        self,
        yhat: str,
        feature_set: List[str] = None,
        raise_errors: bool = False,
    ) -> Tuple[float, dict]:
        """
        Returns the reward for the given solution attempt.
        """
        # parse the solution attempt
        parsed_yhat = self.parse_y_fn(yhat, raise_errors=raise_errors)
        if parsed_yhat is None:
            if raise_errors:
                raise ValueError("Could not parse the solution attempt.")
            return float("-inf"), {"error": "Could not parse the solution attempt."}

        # get the constraints and weights for the feature set
        if feature_set is None:
            feature_set = getattr(self, "_feature_set_current_ids", None)
        weights, active_mask = self.project_weights(feature_set or [])
        features = [
            constraint
            for constraint, active in zip(self.features, active_mask)
            if active
        ]
        new_weights, new_features = [], []
        for constraint, weight in zip(features, weights):
            if not constraint.is_hard:
                new_features.append(constraint)
                new_weights.append(weight)
        try:
            (
                is_valid,
                score,
                min_unconstrained_score,
                max_unconstrained_score,
                metadata,
            ) = linear_reward(
                parsed_yhat,
                constraints=new_features,
                weights=(
                    np.array(new_weights, dtype=float) if len(new_weights) > 0 else None
                ),
                enforce_hard=True,
                raise_errors=raise_errors,
            )
        except Exception as e:
            if raise_errors:
                raise Exception(str(e))
            return float("-inf"), {"error": str(e)}

        # rescale from real numbers to [0, 1]
        if (
            min_unconstrained_score is None
            or max_unconstrained_score is None
            or max_unconstrained_score == min_unconstrained_score
        ):
            norm_score = 1.0
        else:
            norm_score = (score - min_unconstrained_score) / (
                max_unconstrained_score - min_unconstrained_score
            )
        return (
            norm_score * 100,  # rescale from [0, 1] to [0, 100]
            metadata,
        )

    def _linear_validity_fn(
        self,
        yhat: str,
        feature_set: List[str] = None,
        raise_errors: bool = False,
    ) -> Tuple[bool, dict]:
        """
        Returns the validity of the given solution attempt.
        """
        # parse the solution attempt
        parsed_yhat = self.parse_y_fn(yhat, raise_errors=raise_errors)
        if parsed_yhat is None:
            if raise_errors:
                raise ValueError("Could not parse the solution attempt.")
            return False, {"error": "Could not parse the solution attempt."}

        if feature_set is None:
            feature_set = getattr(self, "_feature_set_current_ids", None)
        _, active_mask = self.project_weights(feature_set or [])
        features = [
            constraint
            for constraint, active in zip(self.features, active_mask)
            if active
        ]
        new_features = []
        for constraint in features:
            if constraint.is_hard:
                new_features.append(constraint)
        try:
            (
                is_valid,
                score,
                min_unconstrained_score,
                max_unconstrained_score,
                metadata,
            ) = linear_reward(
                parsed_yhat,
                constraints=new_features,
                weights=None,
                enforce_hard=True,
                raise_errors=raise_errors,
            )
        except Exception as e:
            if raise_errors:
                raise Exception(str(e))
            return False, {"error": str(e)}
        return is_valid, metadata

    def compute_spontaneous_discovery_scores(
        self, yhats: List[str], negative_bonus: float = 1
    ) -> Dict[str, float]:
        """
        Compute an unnormalized discovery score q_j for each feature j given a set of candidate options yhats.

        Uses:
        - Importance I_theta(j) = |theta_j|
        - If len(yhats) == 1 with y: q_j = I_theta(j) * |phi_j(y)|
        - If len(yhats) > 1:       q_j = I_theta(j) * Var_O(phi_j)
        - If len(yhats) == 0:      q_j = 0

        phi_j(y) is taken as the constraint output (in [-1, 1]) for feature j on parsed y.

        Returns a dict mapping constraint.id -> score (float).
        """
        d = len(self.features)
        if len(yhats or []) == 0:
            return {c.id: 0.0 for c in self.features}

        # Parse yhats
        parsed: List[Any] = []
        for y in yhats:
            try:
                parsed.append(self.parse_y_fn(y, raise_errors=False))
            except Exception:
                continue
        parsed = [p for p in parsed if p is not None]
        num_options = len(parsed)

        # If all parsing failed, return zero scores
        if num_options == 0:
            return {c.id: 0.0 for c in self.features}

        # Build phi matrix (num_options, d) with np.nan for failures
        phi = np.full((num_options, d), np.nan, dtype=float)
        for i, y_parsed in enumerate(parsed):
            if y_parsed is None:
                continue
            for j, constraint in enumerate(self.features):
                try:
                    val = constraint(y_parsed)
                    phi[i, j] = float(val) if val is not None else np.nan
                except Exception:
                    phi[i, j] = np.nan

        theta = np.asarray(self.weights, dtype=float)

        scores: Dict[str, float] = {}
        for j, constraint in enumerate(self.features):
            qj = float(abs(theta[j]))
            varj = np.var(phi[:, j]) if num_options > 1 else 1.0
            vj = np.abs(constraint.oracle_value - phi[:, j]).max()
            scores[constraint.id] = float(qj * (varj * vj))

        # Hide already discovered features and non-discoverable features
        for c in self.features:
            if c.id in self._feature_set_current_ids or not c.is_discoverable:
                scores[c.id] = 0.0
        return scores

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the state of the specification.
        """
        return {
            **super().get_state(),
            "_feature_set_current_ids": self._feature_set_current_ids,
            "current_specification": self.get_current_specification(
                feature_set=self._feature_set_current_ids
            ),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load the state of the specification.
        Restores both the base state (files) and LinearFixedSpecification-specific state (_feature_set_current_ids).
        """
        # First, call parent to restore file state
        super().load_state(state)

        # Then restore LinearFixedSpecification-specific state
        if "_feature_set_current_ids" in state:
            self._feature_set_current_ids = state["_feature_set_current_ids"]

    def get_current_specification(
        self,
        feature_set: List[str] = None,
        base_spec: Optional[str] = None,
        highlight: List[str] = None,
    ) -> str:
        """
        Build a specification string that reveals the active features in natural language.

        - Hard constraints are listed under a "Constraints (must be satisfied)" section.
        - Soft preferences are listed under a "Preferences" section, sorted by projected weight.

        Returns the assembled specification string. This method is pure and does not mutate state.
        """
        if feature_set is None:
            feature_set = getattr(self, "_feature_set_current_ids", None)

        # Determine active mask and projected weights for provided feature_set
        projected_weights, active_mask = self.project_weights(feature_set or [])

        # Build maps for descriptions → weights (soft) and list for hard (compute on the fly)
        soft_items_map: Dict[str, float] = {}
        hard_items: List[str] = []

        active_constraints: List[Constraint] = [
            c for c, a in zip(self.features, active_mask) if a
        ]
        for constraint, weight in zip(active_constraints, projected_weights):
            desc = constraint.description or "(unspecified preference)"
            if highlight and constraint.id in highlight:
                desc = f"**{desc}**"
            if constraint.is_hard:
                hard_items.append(desc)
            else:
                # take max if duplicates share the same description
                soft_items_map[desc] = max(soft_items_map.get(desc, 0.0), float(weight))

        # Sort soft by weight (desc)
        soft_items: List[Tuple[str, float]] = sorted(
            [(d, w) for d, w in soft_items_map.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Omit items with 0 weight
        soft_items = [item for item in soft_items if item[1] != 0]

        # Build sections
        lines: List[str] = []
        if hard_items:
            lines.append("Constraints (must be satisfied):")
            for desc in sorted(set(hard_items)):
                lines.append(f"- {desc}")
            lines.append("")
        if soft_items:
            lines.append("Preferences (most important to least important; may not be feasible to satisfy all of these):")
            for desc, _w in soft_items:
                lines.append(f"- {desc}")

        revealed_block = "\n".join(lines).strip()

        # Build a base specification to append revealed section onto
        base = (
            base_spec if base_spec is not None else self.initial_specification
        ) or ""

        # Remove any existing revealed section heuristically
        markers = [
            "Constraints (must be satisfied):",
            "Preferences (most important to least important):",
        ]

        def _strip_revealed(text: str) -> str:
            if not text:
                return ""
            parts = text.split("\n")
            if any(m in text for m in markers):
                # keep everything up to the first marker
                new_parts = []
                for line in parts:
                    if any(m in line for m in markers):
                        break
                    new_parts.append(line)
                return "\n".join(new_parts).rstrip()
            return text.rstrip()

        base_clean = _strip_revealed(base)
        updated = base_clean
        if revealed_block:
            updated = (base_clean + "\n\n" + revealed_block).strip()

        return updated


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
        skip_docker_check: bool = True,
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
        if hasattr(self, "fixed_specs"):
            for spec in self.fixed_specs.values():
                if hasattr(spec, "__del__"):
                    spec.__del__()
        if hasattr(self, "custom_specs"):
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
