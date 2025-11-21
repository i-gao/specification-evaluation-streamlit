"""
Evaluation control flow for both custom and fixed specifications.

This module provides a configurable, step-based control flow for evaluations.
Each step is implemented as a separate function, and the order of steps can be
easily configured via the CUSTOM_EVALUATION_STEPS or FIXED_EVALUATION_STEPS constants.

Dependencies between steps are defined in STEP_DEPENDENCIES. If a prerequisite step
is not in the step list, the dependency is automatically skipped.
"""

import streamlit as st
import time
from typing import Callable, List, Dict, Set
from data.dataset import CustomSpecification, FixedSpecification
from evaluation.app.forms import final_prediction_evaluation
from evaluation.interaction_types import Grade
import evaluation.app.components as components
from evaluation.app.control_flow import (
    complete_final_specification,
    complete_y0_yhat_evaluation,
    complete_final_evaluation,
    save_session_data,
    _run_search_exploration,
    lock_interface,
    unlock_interface,
)


# Define the order and configuration of steps in the custom evaluation flow
# Each step is a tuple of (step_name, enabled)
# Step names must match function names in this module (without the "step_" prefix)
# Note: chat_evaluation is handled separately in evaluation_flow, so it's not included here
CUSTOM_EVALUATION_STEPS: List[tuple] = [
    ("final_prediction", True),
    ("final_evaluation_first", True),
    ("final_specification", True),
    ("y0_yhat_evaluation", True),
    ("search_exploration", True),
    ("final_evaluation_second", True),
]

# Define the order and configuration of steps in the fixed evaluation flow
# Each step is a tuple of (step_name, enabled)
# Step names must match function names in this module (without the "step_" prefix)
# Note: chat_evaluation is handled separately in evaluation_flow, so it's not included here
FIXED_EVALUATION_STEPS: List[tuple] = [
    ("final_prediction", True),
    ("fixed_evaluation", True),
]

# Define dependencies between steps
# Format: {step_name: [list of prerequisite step names]}
# If a prerequisite step is not in the step list, the dependency check is skipped
# Only include dependencies that are conceptually necessary
STEP_DEPENDENCIES: Dict[str, List[str]] = {
    "final_specification": ["chat_evaluation"],  # chat_evaluation is handled separately
    "final_prediction": [],  # No dependencies - can generate prediction independently
    "final_evaluation_first": ["final_prediction"],  # Need prediction to evaluate
    "y0_yhat_evaluation": ["final_prediction"],  # Need prediction to evaluate
    "search_exploration": [],  # No dependencies - can happen anytime after chat eval
    "final_evaluation_second": ["final_prediction"],  # Need prediction to evaluate
    "fixed_evaluation": ["final_prediction"],  # Need prediction to evaluate
}


def _check_step_dependencies(
    step_name: str, completed_steps: Set[str], all_steps: List[str]
) -> bool:
    """
    Check if all dependencies for a step are satisfied.
    
    Only checks dependencies for steps that come BEFORE the current step in the order.
    This allows steps to be reordered without creating circular dependencies.

    Args:
        step_name: Name of the step to check
        completed_steps: Set of step names that have been completed
        all_steps: List of all step names in the current flow (in order)

    Returns:
        True if all dependencies are satisfied, False otherwise
    """
    if step_name not in STEP_DEPENDENCIES:
        return True

    # Find the index of the current step in the flow
    try:
        current_step_index = all_steps.index(step_name)
    except ValueError:
        # Step not in flow, allow it
        return True

    dependencies = STEP_DEPENDENCIES[step_name]
    for dep in dependencies:
        # Skip dependency if the prerequisite step is not in the current flow
        if dep not in all_steps:
            continue
        
        # Only require dependency if it comes BEFORE the current step in the order
        try:
            dep_index = all_steps.index(dep)
            if dep_index < current_step_index:
                # Dependency comes before, so it must be completed
                if dep not in completed_steps:
                    return False
            # If dependency comes after, we don't require it (allows reordering)
        except ValueError:
            # Dependency not in flow, skip
            continue

    return True


def _get_completed_steps_from_state() -> Set[str]:
    """
    Get the set of completed steps based on session state flags.
    Maps state flags to step names.
    """
    completed = set()

    # Map state flags to step names
    if st.session_state.get("chat_evaluation_completed", False):
        completed.add("chat_evaluation")
    if st.session_state.get("final_specification_completed", False):
        completed.add("final_specification")
    if st.session_state.get("final_prediction", None) is not None:
        completed.add("final_prediction")
    if st.session_state.get("final_evaluation_first_completed", False):
        completed.add("final_evaluation_first")
    if st.session_state.get("y0_yhat_evaluation_completed", False):
        completed.add("y0_yhat_evaluation")
    if st.session_state.get("search_exploration_completed", False):
        completed.add("search_exploration")
    if st.session_state.get("final_evaluation_completed", False):
        completed.add("final_evaluation_second")
        completed.add("fixed_evaluation")  # Both use the same flag

    return completed


# ==================== CUSTOM EVALUATION STEPS ====================


def step_final_specification(
    custom_final_specification_form: Callable = None,
) -> bool:
    """
    Step: Show the spec's final specification form (if available).
    Only for custom specifications (fixed specs skip this step).

    Returns True if this step is completed, False otherwise.
    """
    if not st.session_state.chat_evaluation_completed:
        return False

    if st.session_state.final_specification_completed:
        return True

    # For fixed specs, this step is skipped (not in FIXED_EVALUATION_STEPS)
    if isinstance(st.session_state.spec, FixedSpecification):
        # Fixed specs don't have final specification forms
        complete_final_specification()
        return True

    # For custom specs
    if not isinstance(st.session_state.spec, CustomSpecification):
        complete_final_specification()
        return True

    if not st.session_state.spec.user_specification_form_final:
        complete_final_specification()
        return True

    if custom_final_specification_form is None:
        complete_final_specification()
        return True

    def should_show():
        return not st.session_state.final_specification_completed

    def on_completion(feedback):
        if st.session_state.spec.user_specification_callback is not None:
            st.session_state.spec.user_specification_callback(feedback)
        st.session_state.form_results["final_specification"] = feedback
        complete_final_specification()
        st.rerun()

    with st.container(key="narrow_body"):
        st.markdown("## Wrapping up the task...")
        st.markdown("Please answer these final questions about yourself.")
        custom_final_specification_form(
            should_show=should_show,
            on_completion=on_completion,
            user_specification_form_final=st.session_state.spec.user_specification_form_final,
        )

    return False


def step_final_prediction() -> bool:
    """
    Step: Generate the final prediction.
    Works for both custom and fixed specifications.

    Returns True if this step is completed, False otherwise.
    """
    # Only require chat_evaluation_completed (handled separately in evaluation_flow)
    if not st.session_state.chat_evaluation_completed:
        return False

    if st.session_state.get("final_prediction", None) is not None:
        return True

    lock_interface()
    with st.container(key="narrow_body"):
        st.markdown("## Wrapping up the task...")
        with st.spinner(
            "The assistant is generating final artifacts based on your chat session...",
            show_time=True,
        ):
            st.session_state.final_prediction = (
                st.session_state.policy.get_test_prediction()
            )
    unlock_interface()

    return True


def step_final_evaluation_first(
    custom_final_evaluation_form: Callable = None,
) -> bool:
    """
    Step: Show the final evaluation form (first time).
    Only for custom specifications.

    Returns True if this step is completed, False otherwise.
    """
    # Check dependencies dynamically
    all_steps = [name for name, _ in CUSTOM_EVALUATION_STEPS]
    completed_steps = _get_completed_steps_from_state()
    if not _check_step_dependencies(
        "final_evaluation_first", completed_steps, all_steps
    ):
        return False

    if st.session_state.get("final_evaluation_first_completed", False):
        return True

    if not isinstance(st.session_state.spec, CustomSpecification):
        st.session_state.final_evaluation_first_completed = True
        return True

    # Ensure evaluation state
    if st.session_state.evaluation_start_time is None:
        st.session_state.evaluation_start_time = time.time()
    if "final_evaluation" not in st.session_state.form_results:
        st.session_state.form_results["final_evaluation"] = {}
    
    # Show first evaluation form
    completed, feedback = final_prediction_evaluation(
        slider_label="Rate the overall quality of this artifact, from 0 (unusable) to 100 (perfect).",
        text_area_label="What would you change about the assistant's artifact?",
        submit_key="final_eval_first_form",
    )

    # If custom_final_evaluation_form is provided, display it
    if custom_final_evaluation_form is not None:

        def on_completion(feedback_update):
            if "final_evaluation_first" not in st.session_state.form_results:
                st.session_state.form_results["final_evaluation_first"] = {}
            st.session_state.form_results["final_evaluation_first"].update(
                feedback_update
            )

        custom_final_evaluation_form(on_completion=on_completion)

    if completed:
        st.session_state.final_evaluation_first_completed = True
        if feedback is None:
            feedback = {}
        if "final_evaluation_first" not in st.session_state.form_results:
            st.session_state.form_results["final_evaluation_first"] = {}
        st.session_state.form_results["final_evaluation_first"].update(feedback)
        st.rerun()

    return False


def step_y0_yhat_evaluation() -> bool:
    """
    Step: Show the y0/yhat evaluation.
    Only for custom specifications.

    Returns True if this step is completed, False otherwise.
    """
    # Check dependencies dynamically
    all_steps = [name for name, _ in CUSTOM_EVALUATION_STEPS]
    completed_steps = _get_completed_steps_from_state()
    if not _check_step_dependencies("y0_yhat_evaluation", completed_steps, all_steps):
        return False

    if st.session_state.y0_yhat_evaluation_completed:
        return True

    if not isinstance(st.session_state.spec, CustomSpecification):
        complete_y0_yhat_evaluation()
        return True

    final_prediction = st.session_state.final_prediction
    assert final_prediction is not None, "final_prediction is not set"

    render_fn = getattr(st.session_state.spec, "render_y0_yhat_evaluation", None)
    if not callable(render_fn):
        # Fallback to legacy name for backwards compatibility
        render_fn = getattr(st.session_state.spec, "render_evaluation", None)

    if not callable(render_fn):
        st.error(
            "This dataset does not implement render_y0_yhat_evaluation(final_prediction)."
        )
        st.stop()
        return False

    try:
        y0_yhat_done, result_metadata = render_fn(final_prediction)
        if result_metadata is not None:
            if "final_evaluation" not in st.session_state.form_results:
                st.session_state.form_results["final_evaluation"] = {}
            st.session_state.form_results["final_evaluation"].update(result_metadata)
    except Exception as e:
        st.error(f"Error in dataset render_y0_yhat_evaluation: {e}")
        st.stop()
        return False

    if y0_yhat_done:
        complete_y0_yhat_evaluation()
        st.rerun()
        return True

    return False


def step_search_exploration() -> bool:
    """
    Step: Search exploration.
    Only for custom specifications.

    Returns True if this step is completed, False otherwise.
    """
    # Check dependencies dynamically
    all_steps = [name for name, _ in CUSTOM_EVALUATION_STEPS]
    completed_steps = _get_completed_steps_from_state()
    if not _check_step_dependencies("search_exploration", completed_steps, all_steps):
        return False

    if st.session_state.search_exploration_completed:
        return True

    _run_search_exploration()

    if st.session_state.search_exploration_completed:
        st.rerun()
        return True

    return False


def step_final_evaluation_second(
    custom_final_evaluation_form: Callable = None,
    show_liked_items: bool = False,
) -> bool:
    """
    Step: Show the final evaluation form (second time).
    Only for custom specifications.

    Returns True if this step is completed, False otherwise.
    """
    # Check dependencies dynamically
    all_steps = [name for name, _ in CUSTOM_EVALUATION_STEPS]
    completed_steps = _get_completed_steps_from_state()
    if not _check_step_dependencies(
        "final_evaluation_second", completed_steps, all_steps
    ):
        return False

    if st.session_state.final_evaluation_completed:
        return True

    final_prediction = st.session_state.final_prediction
    assert final_prediction is not None, "final_prediction is not set"

    if show_liked_items:
        # Get liked items or assignments based on dataset
        dataset_name = st.session_state.dataset_selector
        liked_items = None
        has_exploration_data = False

        if dataset_name == "shopping":
            liked_items = st.session_state.get("liked_products", set())
            has_exploration_data = len(liked_items) > 0
        elif dataset_name == "meal_planning":
            liked_items = st.session_state.get("liked_recipes", set())
            has_exploration_data = len(liked_items) > 0
        elif dataset_name == "travel_planner":
            liked_items = st.session_state.get("liked_travel_items", set())
            has_exploration_data = len(liked_items) > 0
        elif dataset_name == "workout_planning":
            liked_items = st.session_state.get("liked_exercises", set())
            has_exploration_data = len(liked_items) > 0
        elif dataset_name == "email_organization":
            assignments = st.session_state.get("email_organization_assignments", {})
            has_exploration_data = len(assignments) > 0
        elif dataset_name == "file_organization":
            assignments = st.session_state.get("file_organization_assignments", {})
            has_exploration_data = len(assignments) > 0

        # Display liked items or organization assignments first
        if has_exploration_data:
            if dataset_name in ["email_organization", "file_organization"]:
                st.markdown(
                    "Review your organization choices from exploration, then compare with the assistant's final artifact."
                )
                st.markdown("### Your Organization Policy")

                if dataset_name == "email_organization":
                    from data.email_organization.streamlit_search_interface import (
                        render_user_policy,
                    )

                    assignments = st.session_state.get(
                        "email_organization_assignments", {}
                    )
                    emails_data = getattr(st.session_state.spec, "emails_data", [])
                    render_user_policy(assignments, emails_data)

                elif dataset_name == "file_organization":
                    from data.file_organization.streamlit_search_interface import (
                        render_user_policy,
                    )

                    assignments = st.session_state.get(
                        "file_organization_assignments", {}
                    )
                    files_data = getattr(st.session_state.spec, "files_data", [])
                    render_user_policy(assignments, files_data)
            else:
                st.markdown(
                    "Review the items you liked during exploration, then compare with the assistant's final artifact."
                )
                st.markdown("### Your Liked Items")
                if (
                    isinstance(st.session_state.spec, CustomSpecification)
                    and st.session_state.spec._render_liked_items_fn is not None
                ):
                    st.session_state.spec.render_liked_items(liked_items)
                else:
                    st.markdown("*No search interface available for this dataset.*")

    st.markdown("### Evaluate the assistant's artifact")

    # Final evaluation form (second time)
    completed, feedback = final_prediction_evaluation(
        slider_label="Rate the overall quality of this artifact from 0 (unusable) to 100 (perfect).",
        text_area_label="What would you change about the assistant's artifact?",
        submit_key="final_eval_second_form",
        show_validity_check=True,
    )

    # If custom_final_evaluation_form is provided, display it
    if custom_final_evaluation_form is not None:

        def on_completion(feedback_update):
            if "final_evaluation" not in st.session_state.form_results:
                st.session_state.form_results["final_evaluation"] = {}
            st.session_state.form_results["final_evaluation"].update(feedback_update)

        custom_final_evaluation_form(on_completion=on_completion)

    # If completed, save results (but don't mark final_evaluation_completed yet - that's done by orchestrator)
    if completed:
        if feedback is None:
            feedback = {}
        if "final_evaluation" not in st.session_state.form_results:
            st.session_state.form_results["final_evaluation"] = {}
        st.session_state.form_results["final_evaluation"].update(feedback)

        # Try to compute a Grade and display validity results
        try:
            is_valid, validity_metadata = st.session_state.spec.validity_fn(
                st.session_state.final_prediction
            )
            
            score = st.session_state.form_results["final_evaluation"].get("score", None)
            st.session_state.final_grade = Grade(
                prediction=st.session_state.final_prediction,
                score=score,
                correct=is_valid,
                eval_metadata=validity_metadata,
            )
        except Exception as e:
            # Show error if validity check failed
            with st.container(horizontal=True, horizontal_alignment="right"):
                text = f":red[:material/close:] Error checking validity: {str(e)}"
                st.markdown(
                    f'<div class="validation-container">\n\n{text}</div>',
                    unsafe_allow_html=True,
                )
            st.session_state.final_grade = None

        return True

    return False


# ==================== FIXED EVALUATION STEPS ====================


def step_fixed_evaluation(
    fixed_final_evaluation_form: Callable = None,
) -> bool:
    """
    Step: Run the final evaluation for a fixed specification.
    This calls save_session_data with the final prediction, which runs the reward_fn.

    Returns True if this step is completed, False otherwise.
    """
    # Check dependencies dynamically
    all_steps = [name for name, _ in FIXED_EVALUATION_STEPS]
    completed_steps = _get_completed_steps_from_state()
    if not _check_step_dependencies("fixed_evaluation", completed_steps, all_steps):
        return False

    if st.session_state.final_evaluation_completed:
        return True

    if st.session_state.get("final_prediction", None) is None:
        return False

    if not isinstance(st.session_state.spec, FixedSpecification):
        raise ValueError("Fixed evaluation can only be run for fixed specifications")

    results = save_session_data(skip_grading=False)
    st.session_state.final_grade = results.final_grade
    # Don't mark final_evaluation_completed here - that's done by orchestrator after all steps
    st.session_state.score_history.append(
        (
            st.session_state.final_grade.score,
            st.session_state.final_grade.prediction,
            st.session_state.final_grade.eval_metadata,
        )
    )

    components.score_tracker()
    st.write("Your final score is: ", st.session_state.final_grade.score)

    if fixed_final_evaluation_form is not None:

        def on_completion(feedback):
            st.session_state.form_results["final_evaluation"] = feedback
            st.rerun()

        fixed_final_evaluation_form(
            on_completion=on_completion,
        )
    else:
        if st.button("Next round", type="primary"):
            st.rerun()

    return True


# ==================== ORCHESTRATOR FUNCTIONS ====================


def run_custom_evaluation_flow(
    custom_final_specification_form: Callable = None,
    custom_final_evaluation_form: Callable = None,
    show_liked_items: bool = False,
) -> bool:
    """
    Main orchestrator function that runs the custom evaluation flow.

    This function iterates through the steps defined in CUSTOM_EVALUATION_STEPS
    and executes each enabled step in order. Each step must complete before
    the next step can run.

    Note: chat_evaluation is handled separately in evaluation_flow before this function is called.

    Returns True if all steps are completed, False otherwise.
    """
    if not st.session_state.interaction_completed:
        return False

    if not isinstance(st.session_state.spec, CustomSpecification):
        return False

    # Ensure chat evaluation is completed (prerequisite)
    if not st.session_state.get("chat_evaluation_completed", False):
        return False

    # Ensure timer start and evaluation state
    if st.session_state.evaluation_start_time is None:
        st.session_state.evaluation_start_time = time.time()

    # Get the step mapping
    step_functions = {
        "final_specification": step_final_specification,
        "final_prediction": step_final_prediction,
        "final_evaluation_first": step_final_evaluation_first,
        "y0_yhat_evaluation": step_y0_yhat_evaluation,
        "search_exploration": step_search_exploration,
        "final_evaluation_second": step_final_evaluation_second,
    }

    # Run through steps in order
    for step_name, enabled in CUSTOM_EVALUATION_STEPS:
        if not enabled:
            continue

        # Get the step function
        step_fn = step_functions.get(step_name)
        if step_fn is None:
            st.error(f"Unknown step: {step_name}")
            continue

        # Prepare arguments based on step name
        if step_name == "final_specification":
            completed = step_fn(custom_final_specification_form)
        elif step_name == "final_evaluation_first":
            completed = step_fn(custom_final_evaluation_form)
        elif step_name == "final_evaluation_second":
            completed = step_fn(custom_final_evaluation_form, show_liked_items)
        else:
            completed = step_fn()

        # If step is not completed, stop here
        if not completed:
            return False

    # All steps completed - mark final evaluation as complete
    complete_final_evaluation()
    return True


def run_fixed_evaluation_flow(
    fixed_final_evaluation_form: Callable = None,
) -> bool:
    """
    Main orchestrator function that runs the fixed evaluation flow.

    This function iterates through the steps defined in FIXED_EVALUATION_STEPS
    and executes each enabled step in order. Each step must complete before
    the next step can run.

    Note: chat_evaluation is handled separately in evaluation_flow before this function is called.

    Returns True if all steps are completed, False otherwise.
    """
    if not st.session_state.interaction_completed:
        return False

    if not isinstance(st.session_state.spec, FixedSpecification):
        return False

    # Ensure chat evaluation is completed (prerequisite)
    if not st.session_state.get("chat_evaluation_completed", False):
        return False

    # Get the step mapping
    step_functions = {
        "final_prediction": step_final_prediction,
        "fixed_evaluation": step_fixed_evaluation,
    }

    # Run through steps in order
    for step_name, enabled in FIXED_EVALUATION_STEPS:
        if not enabled:
            continue

        # Get the step function
        step_fn = step_functions.get(step_name)
        if step_fn is None:
            st.error(f"Unknown step: {step_name}")
            continue

        # Prepare arguments based on step name
        if step_name == "fixed_evaluation":
            completed = step_fn(fixed_final_evaluation_form)
        else:
            completed = step_fn()

        # If step is not completed, stop here
        if not completed:
            return False

    # All steps completed - mark final evaluation as complete
    complete_final_evaluation()
    return True
