import streamlit as st
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent.parent))

from data.dataset import FixedSpecification, CustomSpecification
from evaluation.app.control_flow import (
    start_interaction,
    initialize_session_state,
    chat_flow,
    start_brainstorming,
    evaluation_flow,
    exit_survey_flow,
    end_interaction,
    finish_onboarding,
    complete_brainstorming,
)
import evaluation.app.authentication as authentication
import evaluation.app.forms as forms
import evaluation.app.components as components
import evaluation.app.files as files

APP_CONFIG_JSON = ["shared_app_config.json"]
SCREENS = [
    "welcome_screen",
    "dataset_screen",
    "brainstorming_screen",
    "presurvey_screen",
    "chat_screen",
    "evaluation_screen",
    "exit_survey_screen",
    "end_screen",
]
DEBUG_MODE = st.secrets.get("debug_mode", False)

STYLESHEET = """
<style>
    /* Layout */
    .st-key-narrow_body {
        max-width: 60vw;
        margin: 0 auto;
    }

    /* Colors */
    div[data-testid="stTextInputRootElement"] input, div[data-baseweb="select"] div, div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
        background-color: #f0f0ec !important;
    }
    .stFormSubmitButton button:hover, .stButton button:hover {
        background-color: #EAC9BD !important;
    }
    div[data-testid="stAlert"] div[data-baseweb="notification"] {
        background-color: #EAC9BD !important;
        color: inherit;
    }
    a {
        font-weight: 500;
    }
    a .btn {
        background-color: #EAC9BD;
        color: #ffffff;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-decoration: none;
        font-size: 0.9rem;
    }

    /* hide the first radio button so that radio groups look like they are unselected initially */
    div[role="radiogroup"] >  :first-child{
           display: none !important;
    }

    /* Chat messages */
    .validation-container {
        padding-top: 0.5rem;
        text-align: right;
        font-size: 0.85em;
        font-style: italic;
        color: #777;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span, [data-testid="stChatMessage"] li {
        font-size: 0.9em !important;
    }

    /* Floating header */
    .st-key-floating_header {
        padding: 0.8em;
        position: fixed;
        background: #fdfdf8;
        width: 100vw;
        top: 0;
        left: 0;
        z-index: 99999999;
        height: 4em;
        box-shadow: 0 0 10px 0 rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
    }
    [role="dialog"], [data-testid="stFullScreenFrame"] {
        margin-top: 4em;
    }
    [data-testid="stFullScreenFrame"] .stElementToolbar {
        top: 0 !important;
    }
    .stAppHeader, .stMain, .stSidebar {
        padding-top: 8em;
    }
    .stMainBlockContainer {
        padding-top: 0;
    }
    .st-key-floating_header > :first-child {
        /* The progress bar */
        flex: 5;
        flex-shrink: 1;
        text-align: left;
    }
    .st-key-floating_header > :last-child:not(:first-child) {
        /* The countdown */
        flex: 2;
        flex-shrink: 1;
        text-align: right;
        white-space: nowrap;  /* prevents line breaks */
        overflow: hidden;     /* (optional) hides overflow */
        text-overflow: ellipsis; /* (optional) adds "..." if clipped */
    }

    /** Floating button **/
    .st-key-floating_button {
        position: fixed; 
        right: 2rem; 
        z-index: 99999998;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.5rem;
        width: auto;
        bottom: 0;
        padding-bottom: 4rem;
    }
    .st-key-floating_button a, .st-key-floating_button button {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        background: #EAC9BD;
        color: #3d3a2a;
        border: none;
        width: 12vw;
        text-overflow: ellipsis;
    }
    .st-key-floating_button a:hover, .st-key-floating_button button:hover {
        background: #d4b5a7;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
    }

    /** Inline documents **/
    .inline-document {
        padding: 2em;
        border: 1px solid #e0e0e0;
        border-radius: 0.5em;
        margin-bottom: 1em;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        font-family: monospace;
        cursor: text;
    }
    .inline-document h1 {
        font-size: 1.3em;
    }
    .inline-document h2 {
        font-size: 1.1em;
    }
    .inline-document h3 {
        font-size: 1.0em;
    }
    .inline-document h4 {
        font-size: 0.9em;
    }

    /** Tables in render_msg_fn **/
    table {
        width: 100%;
        border-radius: 0.5em;
    }
    table :not(h1, h2, h3, h4, h5, h6) {
        font-size: 0.85rem;
    }
    details {
        padding: 1em;
        border: 1px solid #e0e0e0;
        border-radius: 0.5em;
        margin-bottom: 1em;
        font-size: 0.85rem;
        line-height: 1.2;
        color: #555;
    }
    details summary {
        cursor: pointer;
    }
    details[open] summary {
        margin-bottom: 1em;
    }

    /** Dataset specific styles **/
    .st-key-shopping table td {
        /* data cells */
        border-left: none;
        border-right: none;
        padding: 1rem;
    }
    .st-key-shopping table thead * {
        /* header cells */
        border: none !important;
        color: #555;
        font-weight: 500;
    }
    .st-key-workout_planning table {
        text-align: center;
    }
    .st-key-workout_planning table thead *, .st-key-travel_planner table thead *, .st-key-meal_planning table thead * {
        /* header cells */
        color: #555;
        font-weight: 500;
    }
    .st-key-workout_planning blockquote {
        color: #3d3a2a !important;
    }

    /* Floating jump button */
    .st-key-floating_jump_button {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        z-index: 99999998;
        background: #EAC9BD;
        color: #3d3a2a;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 1.2rem;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .st-key-floating_jump_button:hover {
        background: #d4b5a7;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
    }

</style>
"""


@st.dialog("Task Instructions", width="large")
def task_instructions_dialog():
    st.session_state.spec.render_task_explanation()


def header():
    """
    Fixed app header
    """
    with st.container(key="floating_header", horizontal=True):
        components.experiment_progress_bar()

        if st.session_state.current_screen in [
            "chat_screen",
            "brainstorming_screen",
            "presurvey_screen",
        ]:
            if st.session_state.spec.render_task_explanation is not None:
                st.button(
                    "Task Instructions",
                    on_click=task_instructions_dialog,
                    type="primary",
                )

        # if not DEBUG_MODE and st.session_state.current_screen == "chat_screen":
        #     components.countdown()


def authentication_screen():
    """
    Welcome screen
    """
    # Check if already has valid token
    token = authentication.check_token()
    if token:
        return

    st.title("Welcome to your AI-assisted Creative Space!")
    st.write(
        "In this experiment, you will work together with an AI to co-create useful artifacts for yourself."
    )

    authentication.require_token()


def welcome_screen():
    """
    Welcome screen before any dataset is selected
    """
    if st.session_state.current_screen != "welcome_screen":
        return

    with st.container(key="narrow_body"):
        st.write(
            "## In this experiment, you will work together with an AI to co-create useful artifacts for yourself."
        )
        st.write(
            f"You will work on {st.session_state.total_rounds} different tasks with the same AI assistant. Each task takes around {st.session_state.interaction_budget // 60} minutes."
        )
        st.write("Here's a preview of the tasks you will work on:")

        with st.container(border=True):
            tab_names = [
                f"{i + 1} | {name}"
                for i, name in enumerate(st.session_state.dataset_names)
            ]
            tabs = st.tabs(tab_names)
            for tab, description in zip(tabs, st.session_state.dataset_descriptions):
                with tab:
                    st.write(f"<small>*{description}*</small>", unsafe_allow_html=True)

        if st.button("Let's get started!", type="primary"):
            st.session_state.current_screen = "dataset_screen"
            finish_onboarding()


def dataset_screen():
    """
    Dataset screen after a dataset is selected
    Before starting the actual interaction, explain the task
    """
    if st.session_state.current_screen != "dataset_screen":
        return

    header()

    if isinstance(st.session_state.spec, FixedSpecification):
        components.floating_tools()

    with st.container(key="narrow_body"):
        st.write(
            f"## Task {st.session_state.round_index + 1} ({st.session_state.dataset_names[st.session_state.round_index]}): Instructions"
        )

        # Initialize page state if not exists
        if "dataset_page" not in st.session_state:
            st.session_state.dataset_page = 0

        page_index = st.session_state.dataset_page

        if page_index == 0:
            # Page 1: General task explanation
            st.session_state.spec.render_task_explanation()
            round_time = (
                st.session_state.interaction_budget
                + st.session_state.evaluation_minimum
            )
            st.markdown(
                f"You will have {round_time} seconds ({round_time // 60} minutes) to complete the task. It is okay if you do not completely finish the task within this time. Both your typing time and the assistant's thinking time count towards the budget."
            )

            # Show different buttons based on spec type
            if isinstance(st.session_state.spec, FixedSpecification):
                # Fixed spec: show Next button to go to initial specification
                if st.button("Next page", type="primary"):
                    st.session_state.dataset_page = 1
                    st.rerun()
            else:
                # Custom spec: show Begin button directly
                if st.button("Begin the task", type="primary"):
                    # Always route through brainstorming controller; it will skip itself if unconfigured
                    st.session_state.current_screen = "brainstorming_screen"
                    start_brainstorming()

        elif page_index == 1 and isinstance(st.session_state.spec, FixedSpecification):
            # Page 2: Initial specification (only for fixed specs)
            components.render_specification()

            if st.button("Begin the task", type="primary"):
                # Always route through brainstorming controller; it will skip itself if unconfigured
                st.session_state.current_screen = "brainstorming_screen"
                start_brainstorming()


def presurvey_screen():
    """
    Presurvey screen
    """
    if st.session_state.current_screen != "presurvey_screen":
        return

    header()

    with st.container(key="narrow_body"):
        components.render_specification_banner()
        st.write("Before we begin, please answer a few questions about yourself.")

        def validate(form_values):
            return not any(
                v == "-" for v in form_values["expertise"].values()
            ) and not any(v == "-" for v in form_values["specification"].values())

        def on_completion(form_values):
            if (
                isinstance(st.session_state.spec, CustomSpecification)
                and st.session_state.spec.user_specification_callback is not None
            ):
                st.session_state.spec.user_specification_callback(
                    form_values["specification"]
                )
            st.session_state.form_results["presurvey"] = form_values
            st.session_state.current_screen = "chat_screen"
            start_interaction()

        if isinstance(st.session_state.spec, CustomSpecification):
            user_specification_form_initial = (
                st.session_state.spec.user_specification_form_initial
            )
        else:
            user_specification_form_initial = None

        forms.presurvey(
            should_show=lambda: not st.session_state.presurvey_completed,
            on_completion=on_completion,
            user_expertise_form=st.session_state.spec.user_expertise_form,
            user_specification_form_initial=user_specification_form_initial,
            validate=validate,
        )


def _custom_chat_screen():
    """
    Chat screen for custom specifications
    """
    if st.session_state.current_screen != "chat_screen":
        return
    if not isinstance(st.session_state.spec, CustomSpecification):
        return

    st.markdown(
        ":small[Work with the AI assistant to complete the following task. Remember, create artifacts that are **maximally useful and realistic for *you***!]"
    )
    st.markdown(
        ":small[You should not need to access any external websites. The AI only needs to show you a final recommendation; it does not have the ability to do anything else (e.g. place orders in the real world).]"
    )
    components.render_specification_banner()

    chat_flow(
        show_raw_message=DEBUG_MODE,
        autovalidate=False,
        show_response_time=True,
        show_end_conversation_button=True,
    )


def _fixed_chat_screen():
    """
    Chat screen for fixed specifications
    """
    if st.session_state.current_screen != "chat_screen":
        return
    if not isinstance(st.session_state.spec, FixedSpecification):
        return

    st.markdown(
        ":small[Work with the AI assistant to complete the task and maximize your score.]"
    )

    components.score_tracker()
    components.floating_tools()

    tabs = st.tabs(
        [
            "Chat",
            "Task details (copied from previous page)",
        ]
    )
    with tabs[0]:
        chat_flow(
            collect_feedback=False,
            show_quick_actions=True,
            show_raw_message=DEBUG_MODE,
            autovalidate=True,
        )
    with tabs[1]:
        components.render_specification()


def chat_screen():
    """
    Chat screen
    """
    if st.session_state.current_screen != "chat_screen":
        return

    header()
    if DEBUG_MODE:
        if st.button("End and save conversation"):
            print("Ending interaction")
            st.session_state.current_screen = "evaluation_screen"
            end_interaction("user_end")

    if isinstance(st.session_state.spec, CustomSpecification):
        _custom_chat_screen()
    else:
        _fixed_chat_screen()


def evaluation_screen():
    """
    Evaluation screen
    """
    if st.session_state.current_screen != "evaluation_screen":
        return

    header()

    components.render_specification_banner()

    evaluation_flow(
        chat_evaluation_form=forms.interaction_evaluation,
        custom_final_specification_form=forms.custom_final_specification,
    )


def brainstorming_screen():
    """
    Brainstorming stage between instructions and presurvey.
    Shows current specification, a reflection prompt, a large text box, and a countdown.
    Submission is enabled only after brainstorm_time seconds have elapsed.
    """
    if st.session_state.current_screen != "brainstorming_screen":
        return

    header()

    with st.container(key="narrow_body"):
        # Banner with current specification
        components.render_specification_banner()

        # Start brainstorming timer if not already started
        if getattr(st.session_state, "brainstorm_start_time", None) is None:
            st.session_state.brainstorm_start_time = __import__("time").time()

        # Countdown display
        # components.brainstorm_countdown()

        def validate(form_values):
            # Check if enough time has passed for brainstorming
            brainstorm_time = getattr(st.session_state, "brainstorm_time", None)
            if brainstorm_time is None:
                return True  # No time requirement

            brainstorm_start_time = st.session_state.get("brainstorm_start_time", None)
            if brainstorm_start_time is None:
                return False  # Timer not started

            import time as _time

            elapsed = _time.time() - brainstorm_start_time
            if elapsed < brainstorm_time:
                st.error(
                    f"Please spend at least {brainstorm_time} seconds reflecting before continuing."
                )
                return False

            return True

        def on_completion(form_values):
            st.session_state.form_results.setdefault("brainstorm", {})
            st.session_state.form_results["brainstorm"].update(form_values)
            complete_brainstorming()

        forms.brainstorming(
            should_show=lambda: not st.session_state.brainstorming_completed,
            on_completion=on_completion,
            validate=validate,
        )


def exit_survey_screen():
    """
    Exit survey screen
    """
    if st.session_state.current_screen != "exit_survey_screen":
        return

    exit_survey_flow()


def end_screen():
    """
    End screen
    """
    if st.session_state.current_screen != "end_screen":
        return

    st.cache_resource.clear()

    with st.container(key="narrow_body"):
        st.markdown("## You have finished the experiment")
        st.markdown("Thank you for your participation! Here is your exit code:")
        st.code(st.session_state.exit_code)


def main():
    st.markdown(STYLESHEET, unsafe_allow_html=True)

    #########################
    # Initialization
    #########################

    files.setup_connection()

    authentication_screen()

    initialize_session_state(
        APP_CONFIG_JSON,
        page_config={
            "layout": "wide",
            "page_title": "Co-Creation Experiment",
            "page_icon": "🤝",
        },
        current_screen="welcome_screen",
    )

    #########################
    # Screens
    #########################

    # Since chat_flow makes some changes to control flow without changing the screen,
    # We need to check and change screen here
    if st.session_state.round_index == -1 and st.session_state.exit_survey_completed:
        # round index is -1, which means we've gone through all rounds
        # exit_survey is done, so we go to end
        st.session_state.current_screen = "end_screen"
    elif (
        st.session_state.round_index == -1
        and not st.session_state.exit_survey_completed
    ):
        # round index is -1, which means we've gone through all rounds
        # exit_survey not done, so we go to exit_survey
        st.session_state.current_screen = "exit_survey_screen"
    elif (
        st.session_state.interaction_completed
        and not st.session_state.final_evaluation_completed
    ):
        # interaction is completed, but final evaluation is not, so we go to evaluation
        st.session_state.current_screen = "evaluation_screen"
    elif (
        st.session_state.interaction_started
        and not st.session_state.interaction_completed
    ):
        # interaction is started, but not completed, so we go to chat
        st.session_state.current_screen = "chat_screen"
    elif (
        st.session_state.instructions_completed
        and not st.session_state.interaction_started
    ):
        # Route through brainstorming controller, which will skip itself if unconfigured or already completed
        if not st.session_state.brainstorming_completed:
            st.session_state.current_screen = "brainstorming_screen"
        else:
            st.session_state.current_screen = "presurvey_screen"
    elif (
        st.session_state.onboarding_completed
        and not st.session_state.instructions_completed
    ):
        # onboarding is completed, but instructions are not completed
        # so we go to dataset screen, which will then mark instructions completed
        st.session_state.current_screen = "dataset_screen"

    welcome_screen()
    with st.container(
        key=(
            st.session_state.dataset_selector
            if hasattr(st.session_state, "dataset_selector")
            else "dataset"
        )
    ):
        dataset_screen()
        brainstorming_screen()
        presurvey_screen()
        chat_screen()
        evaluation_screen()
    exit_survey_screen()
    end_screen()


if __name__ == "__main__":
    main()
