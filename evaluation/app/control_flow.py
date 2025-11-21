import streamlit as st
import sys
from pathlib import Path
import time
import json
import os
from collections import defaultdict
from typing import Dict, Union, List, Callable
import copy

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.append(
        str(Path(__file__).parent.parent.parent)
    )  # evaluation/app/ -> evaluation/ -> root

from data.dataset import CustomSpecification
from user_simulator import get_simulator
from data import get_dataset, get_spec
from new_baselines import get_policy
from evaluation.interaction_types import save_interaction
from utils import seed_everything
from utils.model import is_openai_model
from evaluation.namer import get_experiment_name

# forms is imported elsewhere if needed; remove unused import here
import evaluation.app.authentication as authentication
import evaluation.app.components as components
from evaluation.qualitative_eval import COMPARISON_LIKERT, COMPARISON_LIKERT_NUMERIC  # noqa: F401

SAVE_INTERVAL = 60  # Save every minute
BENCHMARK_TIMES_FILE = Path(__file__).parent.parent.parent / "benchmark_times.jsonl"

"""
The control flow is separate from the app screens. The "stages" of the app are:

Start:
0. Onboarding
    - flag: onboarding_completed

For each round:
1. Instructions
    - shows: dataset screen
    - flag: instructions_completed
2. Brainstorming (optional; skipped if brainstorm_time is None)
    - shows: free-write textbox with countdown
    - flag: brainstorming_started
    - flag: brainstorming_completed
3. Presurvey (domain expertise survey)
    - shows: presurvey form
    - flag: presurvey_completed
4. Interaction
    - shows: chat interface
    - flag: interaction_started (prevents double starts)
    - flag: interaction_completed
    3a. Post-user message, pre-assistant response feedback (optional survey)
        - flag: waiting_for_message_feedback
5. Chat evaluation (cognitive load survey)
6. (optional) final specification survey
7. Final prediction evaluation (initial)
8. y0/yhat evaluation
9. Search exploration
10. Final prediction evaluation (final)

End:
11. Exit survey (optional survey)
    - shows: exit survey form
    - flag: exit_survey_completed

The control flow is responsible for updating the session state to the next stage.
The app screens are responsible for displaying the appropriate form if necessary.
"""
CONTROL_FLOW_KEYS = [
    "onboarding_completed",
    "instructions_completed",
    "brainstorming_started",
    "brainstorming_completed",
    "presurvey_completed",
    "interaction_started",
    "interaction_completed",
    "final_specification_completed",
    "final_evaluation_completed",
    "chat_evaluation_completed",
    "y0_yhat_evaluation_completed",
    "search_exploration_completed",
    "final_comparison_completed",
    "exit_survey_completed",
]
SESSION_STATE_ROUND_DEFAULTS = {
    "instructions_completed": False,
    "brainstorming_started": False,
    "brainstorming_completed": False,
    "presurvey_completed": False,
    "interaction_started": False,
    "interaction_completed": False,
    "final_specification_completed": False,
    "post_specification_survey_completed": False,
    "final_evaluation_completed": False,
    "final_evaluation_first_completed": False,
    "chat_evaluation_completed": False,
    "y0_yhat_evaluation_completed": False,
    "search_exploration_completed": False,
    "waiting_for_message_feedback": False,
    "waiting_for_spinner": False,
    "messages": [],
    "message_history": [],
    "tool_history": defaultdict(list),
    "user_costs": [],
    "form_results": {},
    "last_save_time": time.time(),
    "end_reason": "unknown",
    "total_cost": 0.0,
    "score_history": [],
    "final_prediction": None,
    "final_grade": None,
    "interaction_start_time": None,
    "brainstorm_start_time": None,
    "evaluation_start_time": None,
    "budget_exhausted": False,
}
ROUND_CONFIGS = [
    "human_id",
    "max_react_steps",
    "dataset_selector",
    "dataset_kwargs",
    "spec_selector",
    "interaction_budget",
    "model_selector",
    "policy_selector",
    "spec",
    "policy",
    "simulator",
    "config",
    "brainstorm_time",
    "search_exploration_time",
    "output_path",
]


def initialize_session_state(
    app_config_jsons: List[str] = None, page_config: dict = {}, **kwargs
):
    """
    Initialize the session state. Assumes that the user has already been authenticated.
    This should be called one time at the very top of the app; it will only run once.
    """
    if st.session_state.get("_initialized", False):
        return

    st.set_page_config(
        **page_config,
        menu_items={
            "Get help": "mailto:irena@cs.stanford.edu",
            "Report a bug": "mailto:irena@cs.stanford.edu",
        },
    )

    json_kwargs = {}
    if app_config_jsons is not None:
        for app_config_json in app_config_jsons:
            # assume this is in the current folder
            with open(Path(__file__).parent / app_config_json, "r") as f:
                app_config = json.load(f)
            for key, value in app_config.items():
                json_kwargs[key] = value
    json_kwargs.update(kwargs)

    # Mark as initialized upon success; save the initialization parameters
    st.session_state["_app_initialization_params"] = json_kwargs
    st.session_state["_initialized"] = True

    # Non-round control flow flags
    st.session_state.onboarding_completed = False
    st.session_state.exit_survey_completed = False

    # Initialize screen timestamps tracking (persists across rounds)
    if "screen_timestamps" not in st.session_state:
        st.session_state.screen_timestamps = []
    if "current_screen_start_time" not in st.session_state:
        st.session_state.current_screen_start_time = None

    reset_session_state_for_round(round_index=0)


def reset_session_state_for_round(round_index, save_user_progress: bool = True):
    """
    Initializes the session state for a new round.
    It will also save / load the user's progress if save_user_progress is True.
    """
    token = authentication.check_token()
    assert token is not None, "User must be authenticated"
    user_configs = authentication.get_user_configs(token)

    requested_round_index_gt_0 = round_index > 0

    # save / load the user's progress
    message_history = []
    if save_user_progress:
        user_progress_path = f"streamlit_logs/user_progress/{token}.json"
        try:
            user_progress = st.session_state.connection.read(user_progress_path)
        except FileNotFoundError:
            user_progress = {}

        saved_round_index = -1
        for i in range(len(user_progress)):
            # check if round index i is in user_progress
            if str(i) in user_progress:
                message_history.append(
                    (
                        {
                            "dataset_name": user_configs[i]["dataset_selector"],
                            "spec_index": user_configs[i]["spec_selector"],
                            "dataset_kwargs": user_configs[i]["dataset_kwargs"],
                        },
                        user_progress[str(i)]["messages"],
                    )
                )
                saved_round_index = i
            else:
                break

        print("Saved round index", saved_round_index, "round index", round_index)

        # user progressed further than we currently are; jump to round index where they left off
        if round_index < saved_round_index + 1:
            print("Fast forwarding to round index", saved_round_index + 1)
            round_index = saved_round_index + 1

    # look for a exit_survey file
    if round_index == -1 and not st.session_state.exit_survey_completed:
        exit_survey_path = f"streamlit_logs/exit_surveys/{token}.json"
        try:
            st.session_state.connection.read(exit_survey_path)
            complete_exit_survey()
        except FileNotFoundError:
            pass

    # Clean up previous session state
    for key in ROUND_CONFIGS:
        if key in st.session_state:
            del st.session_state[key]

    # Clean up dynamically created state keys (tool results)
    keys_to_remove = []
    for key in st.session_state.keys():
        if key.startswith("tool_result_") or key.startswith("button_"):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del st.session_state[key]

    # Reset dataset page state
    if "dataset_page" in st.session_state:
        del st.session_state["dataset_page"]

    # Clean up dataset-specific session state variables
    # We need to check all possible datasets since we don't know which one was used in the previous round
    # Collect exact keys from search interfaces and prefixes from render modules
    all_exact_keys = set()
    all_key_prefixes = set()

    # Collect search interface exact keys
    datasets_to_check = [
        (
            "shopping",
            "streamlit_search_interface",
            "SEARCH_INTERFACE_SESSION_STATE_KEYS",
        ),
        (
            "meal_planning",
            "streamlit_search_interface",
            "SEARCH_INTERFACE_SESSION_STATE_KEYS",
        ),
        (
            "travel_planner",
            "streamlit_search_interface",
            "SEARCH_INTERFACE_SESSION_STATE_KEYS",
        ),
        (
            "email_organization",
            "streamlit_search_interface",
            "SEARCH_INTERFACE_SESSION_STATE_KEYS",
        ),
        (
            "file_organization",
            "streamlit_search_interface",
            "SEARCH_INTERFACE_SESSION_STATE_KEYS",
        ),
        (
            "workout_planning",
            "streamlit_search_interface",
            "SEARCH_INTERFACE_SESSION_STATE_KEYS",
        ),
    ]

    for dataset_name, module_name, constant_name in datasets_to_check:
        try:
            module = __import__(
                f"data.{dataset_name}.{module_name}", fromlist=[constant_name]
            )
            keys = getattr(module, constant_name, None)
            if keys:
                all_exact_keys.update(keys)
        except (ImportError, AttributeError):
            pass

    # Collect render module key prefixes
    render_datasets_to_check = [
        ("email_organization", "streamlit_render", "RENDER_SESSION_STATE_KEY_PREFIXES"),
        ("file_organization", "streamlit_render", "RENDER_SESSION_STATE_KEY_PREFIXES"),
        ("shopping", "streamlit_render", "RENDER_SESSION_STATE_KEY_PREFIXES"),
        ("workout_planning", "streamlit_render", "RENDER_SESSION_STATE_KEY_PREFIXES"),
        ("meal_planning", "streamlit_render", "RENDER_SESSION_STATE_KEY_PREFIXES"),
        ("travel_planner", "streamlit_render", "RENDER_SESSION_STATE_KEY_PREFIXES"),
        ("design2code", "streamlit_render", "RENDER_SESSION_STATE_KEY_PREFIXES"),
    ]

    for dataset_name, module_name, constant_name in render_datasets_to_check:
        try:
            module = __import__(
                f"data.{dataset_name}.{module_name}", fromlist=[constant_name]
            )
            prefixes = getattr(module, constant_name, None)
            if prefixes:
                all_key_prefixes.update(prefixes)
        except (ImportError, AttributeError):
            pass

    # Also clean up search exploration specific keys (search_exploration_completed is in defaults, so it will be reset)
    search_exploration_keys = [
        "search_exploration_instructions_acknowledged",
        "search_exploration_start_time",
    ]
    all_exact_keys.update(search_exploration_keys)

    # Remove all exact keys
    for key in all_exact_keys:
        if key in st.session_state:
            del st.session_state[key]

    # Remove all session state keys that start with any of the prefixes
    keys_to_remove = []
    for key in st.session_state.keys():
        for prefix in all_key_prefixes:
            if key.startswith(prefix):
                keys_to_remove.append(key)
                break
    for key in keys_to_remove:
        del st.session_state[key]

    # Set session state variables to defaults
    # Use deep copy to prevent modifying the original SESSION_STATE_ROUND_DEFAULTS
    st.session_state.update(copy.deepcopy(SESSION_STATE_ROUND_DEFAULTS))
    st.session_state.update(st.session_state.get("_app_initialization_params", {}))

    # Set round index
    st.session_state.round_index = round_index
    st.session_state.message_history = message_history

    # Read user-specific configs based on the round_index
    st.session_state.total_rounds = len(user_configs)
    dataset_names = [config["dataset_selector"] for config in user_configs]
    st.session_state.dataset_names, st.session_state.dataset_descriptions = zip(
        *[
            (ds.dataset_pretty_name, ds.dataset_description)
            for ds in [get_cached_dataset(name) for name in dataset_names]
        ]
    )
    if user_configs:
        if round_index >= len(user_configs):
            st.session_state.round_index = -1
            st.rerun()
        print(user_configs[round_index])
        st.session_state.update(user_configs[round_index])

    # Validate that all required keys are set, either by the app config or the user config
    for key in [
        "dataset_selector",
        "dataset_kwargs",
        "spec_selector",
        "interaction_budget",
        "model_selector",
        "seed",
        "policy_selector",
        "max_react_steps",
        "output_dir",
        "reasoning_effort",
    ]:
        assert key in st.session_state, f"Key {key} not found in session state"

    # Seed the random number generator
    seed_everything(st.session_state.seed)

    st.session_state.spec = get_cached_spec(
        st.session_state.dataset_selector,
        st.session_state.spec_selector,
        **st.session_state.dataset_kwargs,
    )
    st.session_state.config = get_config()
    st.session_state.output_path = get_output_path(st.session_state.config)

    updated_policy_kwargs = {**st.session_state.config["policy_kwargs"]}
    updated_policy_kwargs["model_kwargs"] = {
        **updated_policy_kwargs["model_kwargs"],
        **get_model_api_key(st.session_state.model_selector),
    }

    st.session_state.policy = get_policy(
        st.session_state.policy_selector,
        spec=st.session_state.spec,
        checkpoint_file=os.path.join(
            st.session_state.output_path.replace(".json", "_policy_state.json")
        ),
        **updated_policy_kwargs,
    )
    st.session_state.simulator = get_simulator(
        "dummy",
        spec=st.session_state.spec,
        interaction_budget=st.session_state.interaction_budget,
        verbosity=1,
    )

    if requested_round_index_gt_0:
        # don't go back to onboarding
        # onboarding flag gets reset by the default values
        finish_onboarding()


########### Update control flow variables ######################


def log_control_flow_transition(
    transition_name: str, from_state: str = None, to_state: str = None
):
    """
    Log a control flow transition. Records when a control flow state changes.

    Args:
        transition_name: The name of the transition/function being called (e.g., "finish_onboarding")
        from_state: The previous state (optional, for clarity)
        to_state: The new state (optional, for clarity)
    """
    print(
        f"LOGGING CONTROL FLOW TRANSITION: {transition_name} (from: {from_state}, to: {to_state})"
    )
    current_time = time.time()

    # Initialize control_flow_transitions if it doesn't exist
    if "control_flow_transitions" not in st.session_state:
        st.session_state.control_flow_transitions = []

    # Record the transition
    st.session_state.control_flow_transitions.append(
        {
            "transition": transition_name,
            "from_state": from_state,
            "to_state": to_state,
            "timestamp": current_time,
            "round_index": st.session_state.get("round_index", None),
        }
    )


def log_screen_transition(new_screen: str):
    """
    Log a screen transition. Records when a screen is entered and when the previous screen was exited.

    Args:
        new_screen: The name of the screen being entered
    """
    print("LOGGING SCREEN TRANSITION TO", new_screen)
    current_time = time.time()

    # If we have a previous screen, mark it as exited
    if st.session_state.get("current_screen_start_time") is not None:
        previous_screen = st.session_state.get("current_screen", None)
        if previous_screen is not None:
            # Find the last entry for this screen that hasn't been exited yet
            for entry in reversed(st.session_state.get("screen_timestamps", [])):
                if (
                    entry["screen"] == previous_screen
                    and entry.get("exited_at") is None
                ):
                    entry["exited_at"] = current_time
                    entry["duration_seconds"] = current_time - entry["entered_at"]
                    break

    # Record the new screen entry
    if "screen_timestamps" not in st.session_state:
        st.session_state.screen_timestamps = []

    st.session_state.screen_timestamps.append(
        {
            "screen": new_screen,
            "entered_at": current_time,
            "exited_at": None,
            "round_index": st.session_state.get("round_index", None),
        }
    )
    st.session_state.current_screen = new_screen
    st.session_state.current_screen_start_time = current_time


def finish_onboarding():
    """Mark onboarding as completed"""
    if st.session_state.onboarding_completed:
        return
    log_control_flow_transition(
        "finish_onboarding",
        from_state="onboarding_completed=False",
        to_state="onboarding_completed=True",
    )
    st.session_state.onboarding_completed = True
    st.rerun()


def complete_instructions():
    """Mark instructions as completed"""
    if st.session_state.instructions_completed:
        return
    log_control_flow_transition(
        "complete_instructions",
        from_state="instructions_completed=False",
        to_state="instructions_completed=True",
    )
    st.session_state.instructions_completed = True


def start_brainstorming():
    """Start the brainstorming stage if configured"""
    complete_instructions()
    if getattr(st.session_state, "brainstorm_time", None) is None:
        # Skip brainstorming if no time is configured
        complete_brainstorming()
        return
    if st.session_state.brainstorming_started:
        return
    log_control_flow_transition(
        "start_brainstorming",
        from_state="brainstorming_started=False",
        to_state="brainstorming_started=True",
    )
    st.session_state.brainstorming_started = True
    st.session_state.brainstorm_start_time = time.time()
    st.rerun()


def complete_brainstorming():
    """Mark brainstorming complete and proceed"""
    if st.session_state.brainstorming_completed:
        return
    log_control_flow_transition(
        "complete_brainstorming",
        from_state="brainstorming_completed=False",
        to_state="brainstorming_completed=True",
    )
    st.session_state.brainstorming_completed = True
    # Proceed to presurvey immediately after brainstorming completes
    start_presurvey()


def start_presurvey():
    """Start the presurvey stage"""
    if st.session_state.presurvey_completed:
        return
    log_control_flow_transition(
        "start_presurvey",
        from_state="presurvey_completed=False",
        to_state="presurvey_started",
    )
    st.rerun()


def complete_presurvey():
    """Mark presurvey as completed"""
    if st.session_state.presurvey_completed:
        return
    log_control_flow_transition(
        "complete_presurvey",
        from_state="presurvey_completed=False",
        to_state="presurvey_completed=True",
    )
    st.session_state.presurvey_completed = True


def start_interaction():
    """Start the interaction stage"""
    if st.session_state.interaction_started:
        return
    assert len(st.session_state.messages) == 0, (
        "Messages must be empty to start an interaction"
    )
    complete_presurvey()
    log_control_flow_transition(
        "start_interaction",
        from_state="interaction_started=False",
        to_state="interaction_started=True",
    )
    st.session_state.interaction_started = True
    st.session_state.interaction_start_time = time.time()
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": None,
            "sent_time": time.time(),
            "response_time": None,
        }
    )
    st.rerun()


def end_interaction(end_reason: str):
    """End the interaction stage"""
    if st.session_state.interaction_completed:
        return
    log_control_flow_transition(
        "end_interaction",
        from_state="interaction_completed=False",
        to_state="interaction_completed=True",
    )
    st.session_state.interaction_completed = True
    st.session_state.end_reason = end_reason
    st.rerun()


def complete_chat_evaluation():
    """Mark chat evaluation as completed"""
    if st.session_state.chat_evaluation_completed:
        return
    log_control_flow_transition(
        "complete_chat_evaluation",
        from_state="chat_evaluation_completed=False",
        to_state="chat_evaluation_completed=True",
    )
    st.session_state.chat_evaluation_completed = True


def complete_post_specification_survey():
    """Mark post specification survey as completed"""
    if st.session_state.get("post_specification_survey_completed", False):
        return
    log_control_flow_transition(
        "complete_post_specification_survey",
        from_state="post_specification_survey_completed=False",
        to_state="post_specification_survey_completed=True",
    )
    st.session_state.post_specification_survey_completed = True


def complete_final_specification():
    """Mark final specification as completed"""
    if st.session_state.final_specification_completed:
        return
    log_control_flow_transition(
        "complete_final_specification",
        from_state="final_specification_completed=False",
        to_state="final_specification_completed=True",
    )
    st.session_state.final_specification_completed = True


def complete_y0_yhat_evaluation():
    """Mark y0/yhat evaluation as completed"""
    if st.session_state.y0_yhat_evaluation_completed:
        return
    log_control_flow_transition(
        "complete_y0_yhat_evaluation",
        from_state="y0_yhat_evaluation_completed=False",
        to_state="y0_yhat_evaluation_completed=True",
    )
    st.session_state.y0_yhat_evaluation_completed = True


def complete_search_exploration():
    """Mark search exploration as completed"""
    if st.session_state.search_exploration_completed:
        return
    log_control_flow_transition(
        "complete_search_exploration",
        from_state="search_exploration_completed=False",
        to_state="search_exploration_completed=True",
    )
    st.session_state.search_exploration_completed = True


def complete_final_evaluation():
    """Mark final evaluation as completed"""
    if st.session_state.final_evaluation_completed:
        return
    log_control_flow_transition(
        "complete_final_evaluation",
        from_state="final_evaluation_completed=False",
        to_state="final_evaluation_completed=True",
    )
    st.session_state.final_evaluation_completed = True


def complete_exit_survey():
    """Mark exit survey as completed"""
    if st.session_state.exit_survey_completed:
        return
    log_control_flow_transition(
        "complete_exit_survey",
        from_state="exit_survey_completed=False",
        to_state="exit_survey_completed=True",
    )
    st.session_state.exit_survey_completed = True


######### HELPER FUNCTIONS #########


def get_model_api_key(model_name: str) -> dict:
    """
    Determine which API key to use based on the model type.

    Args:
        model_name: The name of the model to check.

    Returns:
        dict: Dictionary containing the appropriate api_key parameter for model_kwargs.
    """
    if is_openai_model(model_name, api_key=st.secrets.get("openai_api_key")):
        # Use OpenAI API key from secrets
        return {"api_key": st.secrets.get("openai_api_key")}
    else:
        # Use Anthropic API key from secrets (assumes non-OpenAI models are Anthropic)
        return {"api_key": st.secrets.get("anthropic_api_key")}


def check_intermediate_save():
    """
    Check if enough time has passed to save the session data
    """
    # Save session data if enough time has passed
    current_time = time.time()
    if current_time - st.session_state.last_save_time >= SAVE_INTERVAL:
        save_session_data(skip_grading=True)
        st.session_state.last_save_time = current_time


def save_session_data(save_user_progress: bool = True, **kwargs):
    """
    Get current session data and save it to a JSON file.
    Args:
        save_user_progress: Whether to save the user's progress (only saves if the interaction is completed)
        kwargs: Additional kwargs to pass to save_interaction
    """
    assert all(hasattr(st.session_state, k) for k in ROUND_CONFIGS)

    # Save user progress if the interaction is completed and save_user_progress is True
    if save_user_progress and st.session_state.interaction_completed:
        token = authentication.check_token()
        user_progress_path = f"streamlit_logs/user_progress/{token}.json"
        try:
            user_progress = st.session_state.connection.read(user_progress_path)
        except FileNotFoundError:
            user_progress = {}
        user_progress[st.session_state.round_index] = {
            "output_filename": st.session_state.output_path,
            "round_index": st.session_state.round_index,
            "messages": st.session_state.messages,
        }
        st.session_state.connection.write(user_progress_path, json.dumps(user_progress))

    if st.session_state.interaction_completed:
        st.session_state.message_history.append(
            [
                {
                    "dataset_name": st.session_state.dataset_selector,
                    "spec_index": st.session_state.spec_selector,
                    "dataset_kwargs": st.session_state.dataset_kwargs,
                },
                st.session_state.messages,
            ]
        )

    # Infill the history into the simulator for save_interaction
    st.session_state.simulator.infill_history(
        messages=st.session_state.messages,
        tool_history=st.session_state.tool_history,
    )

    # Prepare extra_metadata with screen timestamps
    # Close any open screen entries before saving
    current_time = time.time()
    screen_timestamps = st.session_state.get("screen_timestamps", []).copy()
    for entry in screen_timestamps:
        if entry.get("exited_at") is None:
            entry["exited_at"] = current_time
            entry["duration_seconds"] = current_time - entry["entered_at"]

    # Filter to include timestamps for the current round, or global screens (round_index None or -1)
    # when saving the last round
    current_round_index = st.session_state.round_index
    round_screen_timestamps = [
        entry
        for entry in screen_timestamps
        if entry.get("round_index") == current_round_index
        or (current_round_index == -1 and entry.get("round_index") in [None, -1])
    ]

    # Filter control flow transitions for the current round
    control_flow_transitions = st.session_state.get("control_flow_transitions", [])
    round_control_flow_transitions = [
        entry
        for entry in control_flow_transitions
        if entry.get("round_index") == current_round_index
        or (current_round_index == -1 and entry.get("round_index") in [None, -1])
    ]

    extra_metadata = kwargs.get("extra_metadata", {})
    extra_metadata["screen_timestamps"] = round_screen_timestamps
    extra_metadata["control_flow_transitions"] = round_control_flow_transitions
    kwargs["extra_metadata"] = extra_metadata

    # Save the interaction
    return save_interaction(
        simulator=st.session_state.simulator,
        policy=st.session_state.policy,
        grader=st.session_state.simulator,
        output_path=st.session_state.output_path,
        config=st.session_state.config,
        end_reason=st.session_state.end_reason,
        form_results=st.session_state.form_results,
        final_prediction=st.session_state.get("final_prediction", None),
        final_grade=st.session_state.get("final_grade", None),
        connection=st.session_state.connection,
        **kwargs,
    )


def get_cached_dataset(dataset_name, **dataset_kwargs):
    """Get cached dataset"""
    with st.spinner("Loading your tasks..."):
        try:
            return get_dataset(
                dataset_name, **dataset_kwargs, allow_multimodal_actions=True
            )
        except Exception as e:
            st.error(f"Error loading {dataset_name} dataset: {str(e)}")
            st.stop()


def get_cached_spec(dataset_name, spec_index, **dataset_kwargs):
    """Get cached spec"""
    print(f"Getting cached spec for {dataset_name} {spec_index} {dataset_kwargs}")
    with st.spinner("Loading your tasks..."):
        return get_spec(
            dataset_name,
            spec_index,
            **dataset_kwargs,
            allow_multimodal_actions=True,
        )


def get_output_path(config: Dict[str, Union[dict, str]]):
    """Get output path"""
    output_filename = get_experiment_name(include_datetime=True, **config) + ".json"
    os.makedirs(config["output_dir"], exist_ok=True)
    return os.path.join(config["output_dir"], output_filename)


def get_config():
    """Builds the config that is sent to get_experiment_name and save_interaction"""
    return {
        "dataset": st.session_state.dataset_selector,
        "dataset_kwargs": st.session_state.dataset_kwargs,
        "spec_index": st.session_state.spec_selector,
        "policy": st.session_state.policy_selector,
        "policy_model": st.session_state.model_selector,
        "policy_kwargs": {
            "max_react_steps": st.session_state.max_react_steps,
            "model_name": st.session_state.model_selector,
            "verbosity": 2 if st.secrets.get("debug_mode", False) else 0,
            "interaction_budget": st.session_state.interaction_budget,
            "actions": st.session_state.spec.public_tools,
            "prediction_fmt_instructions": st.session_state.spec.prediction_fmt_instructions,
            "commonsense_instructions": st.session_state.spec.commonsense_description,
            "msg_fmt_instructions": st.session_state.spec.msg_fmt_instructions,
            "cost_type": st.session_state.cost_type,
            "model_kwargs": {
                "reasoning_effort": st.session_state.reasoning_effort,
                **getattr(st.session_state, "model_kwargs", {}),
            },
        },
        "interaction_budget": st.session_state.interaction_budget,
        "simulator": "human",
        "simulator_model": st.session_state.human_id,
        "simulator_kwargs": {},
        "seed": st.session_state.seed,
        "user_first": True,
        "max_react_steps": st.session_state.max_react_steps,
        "output_dir": st.session_state.output_dir,
    }


def get_total_cost():
    """
    Get the total completed cost of the conversation according to cost_type.
    Uses recorded segment costs and does NOT include live in-progress time.
    """
    user_completed, assistant_completed = _get_completed_segment_costs()
    cost_type = st.session_state.get("cost_type", "user")
    if cost_type == "user":
        return user_completed
    if cost_type == "policy":
        return assistant_completed
    if cost_type == "both":
        return user_completed + assistant_completed
    raise ValueError(f"Invalid cost type: {cost_type}")


def lock_interface():
    if st.session_state.waiting_for_spinner:
        return

    st.session_state.waiting_for_spinner = True
    st.rerun()


def unlock_interface():
    if not st.session_state.waiting_for_spinner:
        return

    st.session_state.waiting_for_spinner = False
    st.rerun()


@st.cache_resource(show_spinner=False)
def get_expected_message_time(
    filter_to_model: bool = False, filter_to_max_react_steps: bool = True
):
    """
    Get the expected message time based on the benchmark times.
    Assumes the benchmark times t_i ~ N(mu, sigma^2)
    and return T s.t. P(t <= T) = 0.95
    """
    try:
        benchmark_times = st.session_state.connection.read(BENCHMARK_TIMES_FILE)
    except Exception:
        return None
    if filter_to_max_react_steps:
        benchmark_times = [
            time
            for time in benchmark_times
            if time["max_react_steps"] == st.session_state.max_react_steps
        ]
    if len(benchmark_times) == 0:
        return None

    import numpy as np

    benchmark_times = [time["time"] for time in benchmark_times]
    mu = np.mean(benchmark_times)
    sigma = np.std(benchmark_times)
    T = mu + 1.96 * sigma
    return T


################## CHAT FLOW ######################


def _get_current_speaker():
    """
    If the last message was sent by the assistant, return "user", and vice versa.
    If the last message is None, return "user".
    """
    if len(st.session_state.messages) == 0:
        return "user"
    return (
        "user" if st.session_state.messages[-1]["role"] == "assistant" else "assistant"
    )


def _get_last_msg_by_role(role: str):
    """
    Get the content of the last message sent by the assistant.
    """
    if len(st.session_state.messages) == 0:
        return None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == role:
            return msg["content"]
    return None


def _get_last_msg_index_by_role(role: str):
    """
    Get the index of the last message sent by the specified role.
    Returns None if no message with that role is found.
    """
    if len(st.session_state.messages) == 0:
        return None
    for i in range(len(st.session_state.messages) - 1, -1, -1):
        if st.session_state.messages[i]["role"] == role:
            return i
    return None


def _get_completed_segment_costs() -> tuple:
    """
    Return (user_completed_seconds, assistant_completed_seconds) from recorded segments.
    Uses alternating entries in st.session_state.user_costs: user at even indices, assistant at odd indices.
    Does not include live (in-progress) segments.
    """
    segment_costs = st.session_state.get("user_costs", [])
    user_completed = sum(segment_costs[0::2]) if segment_costs else 0.0
    assistant_completed = sum(segment_costs[1::2]) if len(segment_costs) > 1 else 0.0
    return float(user_completed), float(assistant_completed)


def _get_live_segment_seconds() -> float:
    """
    Return the live (in-progress) segment duration since the last message sent_time,
    or 0.0 if the interaction is completed or there are no messages yet.
    """
    if st.session_state.get("interaction_completed", False):
        return 0.0
    if len(st.session_state.get("messages", [])) == 0:
        return 0.0
    return max(0.0, time.time() - st.session_state.messages[-1]["sent_time"])


def _get_spent_seconds(cost_type: str) -> float:
    """
    Compute spent seconds based on cost_type in {"user", "policy", "both"}.
    Includes live segment time only for the current speaker when appropriate.
    Falls back to wall-clock since interaction start if cost_type is unknown.
    """
    user_completed, assistant_completed = _get_completed_segment_costs()
    live = _get_live_segment_seconds()
    is_user_turn = _get_current_speaker() == "user"

    if cost_type == "user":
        return user_completed + (live if is_user_turn else 0.0)
    if cost_type == "policy":
        return assistant_completed + (live if not is_user_turn else 0.0)
    if cost_type == "both":
        return user_completed + assistant_completed + live

    # Fallback to wall-clock
    start_time = st.session_state.get("interaction_start_time", time.time())
    return max(0.0, time.time() - start_time)


def get_countdown_params() -> tuple:
    """
    Return (start_time, time_to_remove) suitable for components.countdown(start_time, time_to_remove).

    The countdown function calculates: remaining = target_time - (now - start_time - time_to_remove).
    When interaction hasn't started: returns (now, 0.0) so remaining = budget.
    When interaction has started: returns (interaction_start_time, time_to_remove) where
    time_to_remove = wall_elapsed - spent, so remaining = budget - spent.
    """
    now = time.time()
    start_time_sec = st.session_state.get("interaction_start_time", None)
    budget = float(st.session_state.get("interaction_budget", 0.0))  # Used conceptually, passed as target_time
    if start_time_sec is None:
        # Not started yet; show full budget
        # Return (now, 0.0) so that countdown calculates: budget - (now - now - 0) = budget
        return now, 0.0

    wall_elapsed = max(0.0, now - start_time_sec)
    spent = _get_spent_seconds(st.session_state.get("cost_type", "user"))
    time_to_remove = max(0.0, wall_elapsed - float(spent))
    # Return start_time_sec (not end_time) so that countdown calculates:
    # elapsed = now - start_time_sec - time_to_remove = spent
    # remaining = budget - spent
    return start_time_sec, time_to_remove


def _log_user_message(message: str, collect_feedback: bool = False):
    """
    Add the user message to the messages list and update the total time
    """
    # Calculate user response time
    user_sent_time = time.time()
    user_cost = user_sent_time - st.session_state.messages[-1]["sent_time"]
    st.session_state.user_costs.append(user_cost)
    st.session_state.total_cost = get_total_cost()

    # Add user message to state
    st.session_state.messages.append(
        {
            "role": "user",
            "content": message,
            "sent_time": user_sent_time,
            "response_time": user_cost,
        }
    )
    if collect_feedback:
        st.session_state.waiting_for_message_feedback = True
    check_intermediate_save()
    # _check_interaction_budget() # only use budget exhausted after the last assistant msg
    st.rerun()


def _log_assistant_message(message: str):
    """
    Add the assistant message to the messages list and update the total time
    """
    print("Logging assistant message", message)
    assistant_sent_time = time.time()
    assistant_cost = assistant_sent_time - st.session_state.messages[-1]["sent_time"]
    st.session_state.user_costs.append(assistant_cost)
    st.session_state.total_cost = get_total_cost()
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": message,
            "sent_time": assistant_sent_time,
            "response_time": assistant_cost,
        }
    )
    _check_interaction_budget()
    unlock_interface()


def _check_interaction_budget():
    """
    Refresh the total cost and check if the interaction budget has been exceeded
    """
    st.session_state.total_cost = get_total_cost()
    if st.session_state.total_cost >= st.session_state.interaction_budget:
        # Mark budget as exhausted and return to let the UI present an end button
        st.session_state.budget_exhausted = True


@st.fragment(run_every=2)
def brainstorm_countdown():
    """
    Countdown timer for brainstorming
    """
    components.countdown(
        start_time=st.session_state.brainstorm_start_time,
        target_time=st.session_state.get("brainstorm_time", 0),
    )


@st.fragment(run_every=2)
def interaction_countdown():
    start_time, to_remove = get_countdown_params()
    components.countdown(
        start_time=start_time,
        time_to_remove=to_remove,
        target_time=st.session_state.interaction_budget,
    )


def search_exploration_button():
    """
    Show "Finished exploring" button next to the countdown
    """
    button_clicked = st.button("Finished exploring", type="primary")
    search_exploration_time = getattr(st.session_state, "search_exploration_time", None)
    # Calculate remaining time
    elapsed_time = time.time() - st.session_state.search_exploration_start_time
    remaining_time = max(0, search_exploration_time - elapsed_time) if search_exploration_time else 0

    if button_clicked:
        # Validate that enough time has passed
        if remaining_time > 0:
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            search_duration_minutes = search_exploration_time / 60
            st.error(
                f"Please spend at least {search_duration_minutes:.1f} minutes exploring before continuing. "
                f"You have {minutes}:{seconds:02d} remaining."
            )
        else:
            # Time has elapsed, proceed
            complete_search_exploration()
            st.rerun()


@st.fragment(run_every=2)
def search_exploration_countdown():
    """
    Countdown for search exploration stage, displayed in the header.
    Duration is loaded from st.session_state.search_exploration_time (in seconds).
    Also shows the "Finished exploring" button when countdown is active.
    """
    if st.session_state.get("search_exploration_start_time") is None:
        return

    # Get search duration from config - return early if not set
    search_exploration_time = getattr(st.session_state, "search_exploration_time", None)
    if search_exploration_time is None:
        return

    # Show countdown
    components.countdown(
        start_time=st.session_state.search_exploration_start_time,
        target_time=search_exploration_time,
        label="seconds remaining",
    )


def chat_flow(
    show_quick_actions: bool = False,
    show_raw_message: bool = False,
    message_feedback_form: Callable = None,
    autovalidate: bool = False,
    autoscore: bool = False,
    show_response_time: bool = False,
    show_end_conversation_button: bool = False,
    show_frustration_button: bool = False,
):
    """
    Chat flow. Handles the conversation between the user and the assistant.
    This function runs if the interaction has started, but may still run after the interaction has ended (to collect post-user message feedback)
    Subsequent screens need to
    """
    if not st.session_state.interaction_started:
        return

    ##################################

    # Display conversation
    components.chat_conversation(
        st.session_state.messages,
        show_quick_actions=show_quick_actions,
        show_raw_message=show_raw_message,
        autovalidate=autovalidate,
        autoscore=autoscore,
        show_response_time=show_response_time,
    )

    # Display the input box for the user to send a message
    print("Current speaker", _get_current_speaker())
    if not st.session_state.interaction_completed and _get_current_speaker() == "user":
        with st.container(horizontal=True):
            show_end_button = st.session_state.get("budget_exhausted", False) or (
                show_end_conversation_button
                and st.session_state.policy.wants_to_end_conversation
            )
            # Disable input when budget is exhausted
            if user_msg := st.chat_input(
                "Type your message here...",
                key="chat_input",
                disabled=st.session_state.get("budget_exhausted", False),
            ):
                _log_user_message(
                    user_msg, collect_feedback=(message_feedback_form is not None)
                )
            # Show end button if it should be shown (takes priority)
            if show_end_button:
                btn_txt = "End conversation"
                if st.session_state.get("budget_exhausted", False):
                    btn_txt += " (time exhausted)"
                if st.button(btn_txt, type="primary"):
                    if st.session_state.get("budget_exhausted", False):
                        end_interaction("budget_exhausted")
                    else:
                        end_interaction("user_end")
            # Show frustration button if enabled and end button is not shown
            elif show_frustration_button:
                if st.button("Give Up", type="primary"):
                    # Get the last assistant message
                    last_assistant_msg = _get_last_msg_by_role("assistant")
                    click_time = time.time()

                    # Log to form results
                    if "frustration_feedback" not in st.session_state.form_results:
                        st.session_state.form_results["frustration_feedback"] = []
                    st.session_state.form_results["frustration_feedback"].append(
                        {
                            "last_assistant_message": last_assistant_msg,
                            "click_time": click_time,
                            "timestamp": time.strftime(
                                "%Y-%m-%d %H:%M:%S", time.localtime(click_time)
                            ),
                        }
                    )

                    # Show popup message
                    st.toast(
                        "Please complete the task. If you are happy with the result, tell the assistant to end the conversation."
                    )

    # Display message feedback form if appropriate
    # Note: the feedback time currently DOES count towards the interaction budget
    if st.session_state.waiting_for_message_feedback:
        # Note: we do not check if the interaction is over; the interaction may have just ended,
        # but we still want feedback on the last message

        # Check to make sure it's really appropriate to wait for feedback
        msg_to_evaluate = _get_last_msg_by_role("assistant")
        msg_index = _get_last_msg_index_by_role("assistant")
        if (
            (_get_current_speaker() != "assistant")
            or (msg_to_evaluate is None)
            or (msg_index is None)
        ):
            st.session_state.waiting_for_message_feedback = False
            st.rerun()

        # Display the message feedback form
        def should_show():
            return st.session_state.waiting_for_message_feedback

        def on_completion(feedback):
            if "per_msg_feedback" not in st.session_state.form_results:
                st.session_state.form_results["per_msg_feedback"] = []
            st.session_state.form_results["per_msg_feedback"].append(
                {
                    "msg_to_evaluate": msg_to_evaluate,
                    "feedback": feedback,
                }
            )
            st.session_state.waiting_for_message_feedback = False

            # modify sent time of the last message to account for the feedback time
            st.session_state.messages[-1]["sent_time"] = time.time()
            st.rerun()

        message_feedback_form(
            message_index=msg_index,
            should_show=should_show,
            on_completion=on_completion,
        )

    # Call generate() to get assistant response
    if (
        not st.session_state.interaction_completed
        and not st.session_state.waiting_for_message_feedback
        and _get_current_speaker() == "assistant"
    ):
        lock_interface()
        expected_time = get_expected_message_time()

        with st.spinner(
            "The assistant is thinking hard..."
            + (
                f"\t :gray[:small[(estimated time: {expected_time:.0f} seconds)]]"
                if expected_time is not None
                else ""
            ),
            show_time=True,
        ):
            try:
                response = st.session_state.policy(
                    _get_last_msg_by_role("user"),
                    st.session_state.user_costs[-1],
                )
                if response is None:
                    st.stop()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()
            _log_assistant_message(response)


################## EVALUATION FLOW ######################


# Note: _run_post_specification_survey, _run_final_specification, _run_final_prediction,
# and _run_fixed_evaluation have been moved to evaluation_control_flow.py
# They are now accessed via run_fixed_evaluation_flow() and run_custom_evaluation_flow()


@st.fragment
def _run_search_exploration():
    """
    Run the search exploration stage with a countdown timer.
    First shows instructions, then the search interface.
    Duration is loaded from st.session_state.search_exploration_time (in seconds).
    If not set, this stage is skipped.
    """
    if st.session_state.search_exploration_completed:
        return

    # Get search duration from config - skip if not set
    search_exploration_time = st.session_state.get("search_exploration_time", None)
    if search_exploration_time is None:
        # Skip search exploration if not configured
        complete_search_exploration()
        return

    # Check if spec has search interface
    if not isinstance(st.session_state.spec, CustomSpecification):
        complete_search_exploration()
        return

    if st.session_state.spec._render_search_interface_fn is None:
        # No search interface for this dataset, skip this stage
        complete_search_exploration()
        return

    # Check if instructions have been acknowledged
    if "search_exploration_instructions_acknowledged" not in st.session_state:
        st.session_state.search_exploration_instructions_acknowledged = False

    # Show instructions page first
    if not st.session_state.search_exploration_instructions_acknowledged:
        search_duration_minutes = search_exploration_time / 60
        dataset_name = st.session_state.get("dataset_selector", None)

        # Get dataset-specific instructions
        if dataset_name == "email_organization":
            instructions = f"""
            You will now have **{search_duration_minutes:.1f} minutes** to explore the emails by yourself.
            
            During this time, you can:
            - Browse through the emails in your inbox
            - Create folders to organize emails
            - Assign emails to folders to design your ideal email organization
            
            **Please take your time to explore thoroughly and think about how you would like to organize these emails.**
            """
        elif dataset_name == "file_organization":
            instructions = f"""
            You will now have **{search_duration_minutes:.1f} minutes** to explore the files by yourself.
            
            During this time, you can:
            - Browse through the files in your system
            - Create folders to organize files
            - Assign files to folders to design your ideal file organization
            
            **Please take your time to explore thoroughly and think about how you would like to organize these files.**
            """
        else:
            # Default instructions for shopping, meal_planning, travel_planner, workout_planning, etc.
            instructions = f"""
            You will now have **{search_duration_minutes:.1f} minutes** to explore the options available to you by yourself.
            
            During this time, you can:
            - Search for items using natural language queries
            - Apply filters to narrow down results
            - Click the :material/favorite: button on items you like to add them to your liked list
            
            **Please take your time to explore thoroughly.**
            """

        with st.container(key="narrow_body"):
            st.markdown("## Free exploration")
            st.markdown(instructions)

            if st.button(
                "I understand, start exploration",
                type="primary",
                key="start_search_exploration",
            ):
                st.session_state.search_exploration_instructions_acknowledged = True
                st.session_state.search_exploration_start_time = time.time()
                st.rerun()
            return

    # Initialize search start time if not already set
    if "search_exploration_start_time" not in st.session_state:
        st.session_state.search_exploration_start_time = time.time()

    # Render the search interface (countdown and "Finished exploring" button are shown in header via search_exploration_countdown)
    st.session_state.spec.render_search_interface()


def _run_chat_evaluation(
    chat_evaluation_form: Callable = None,
):
    """
    Run the chat evaluation.
    """
    if st.session_state.chat_evaluation_completed:
        return

    # Check if we need to show the chat evaluation form
    if chat_evaluation_form is None:
        complete_chat_evaluation()
        return

    def should_show_chat():
        return not st.session_state.chat_evaluation_completed

    def on_completion_chat(feedback):
        complete_chat_evaluation()
        st.session_state.form_results["chat_evaluation"] = feedback
        st.rerun()

    def validate_chat(feedback):
        return all(item != "-" for item in feedback.values())

    # Display the chat history
    st.markdown("Review your chat session with the assistant.")
    with st.container(border=True, height=600):
        components.chat_conversation(
            st.session_state.messages,
            show_raw_message=False,
            empty_message_text="No messages were recorded in this chat session.",
            show_response_time=True,
        )

    st.markdown(
        "Based on how the assistant responded to your messages, answer the questions below."
    )

    # Display chat evaluation form
    chat_evaluation_form(
        should_show=should_show_chat,
        on_completion=on_completion_chat,
        validate=validate_chat,
    )


def evaluation_flow(
    show_debug_info: bool = False,
    show_start_over_button: bool = False,
    custom_final_specification_form: Callable = None,
    custom_final_evaluation_form: Callable = None,
    fixed_final_evaluation_form: Callable = None,
    chat_evaluation_form: Callable = None,
    post_specification_survey_form: Callable = None,
):
    """
    Evaluation flow upon the completion of an interaction
    """
    if not st.session_state.interaction_completed:
        return

    # Step 1: Chat evaluation (for both custom and fixed)
    _run_chat_evaluation(
        chat_evaluation_form=chat_evaluation_form,
    )

    # For custom specs, skip post_specification_survey and use custom flow
    if isinstance(st.session_state.spec, CustomSpecification):
        from evaluation.app.evaluation_control_flow import run_custom_evaluation_flow

        run_custom_evaluation_flow(
            custom_final_specification_form=custom_final_specification_form,
            custom_final_evaluation_form=custom_final_evaluation_form,
            show_liked_items=True,
        )
    else:
        # For fixed specs, use the fixed evaluation flow
        from evaluation.app.evaluation_control_flow import run_fixed_evaluation_flow

        run_fixed_evaluation_flow(
            fixed_final_evaluation_form=fixed_final_evaluation_form,
        )

    if st.session_state.final_evaluation_completed:
        save_session_data(skip_grading=True)  # just in case
        reset_session_state_for_round(st.session_state.round_index + 1)
        st.rerun()


def exit_survey_flow(exit_survey_form: Callable = None):
    """
    Exit survey flow
    """
    if exit_survey_form is None:
        complete_exit_survey()
        return

    def should_show():
        return not st.session_state.exit_survey_completed

    def on_completion(feedback):
        st.session_state.form_results["exit_survey"] = feedback

        token = authentication.check_token()
        user_exit_survey_path = f"streamlit_logs/exit_surveys/{token}.json"
        st.session_state.connection.write(
            user_exit_survey_path, json.dumps(st.session_state.form_results)
        )

        complete_exit_survey()
        st.rerun()

    exit_survey_form(should_show=should_show, on_completion=on_completion)
