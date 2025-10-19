import streamlit as st
import json
from typing import List
from user_simulator.user import ToolCall
from data.dataset import FixedSpecification
import time
import re
import pandas as pd
from utils.streamlit_types import display_element_to_streamlit, DisplayElement
import uuid

########### Components ##########


def sidebar_tools():
    """
    Expander in the sidebar that provides the human acess to all tools
    """
    assert "simulator" in st.session_state, "Simulator not found in session state"

    with st.sidebar.expander("Tools", expanded=True):
        available_actions = st.session_state.simulator.actions
        if not len(available_actions):
            st.markdown("*No tools available for this task*")
            return

        action_names = [action.name for action in available_actions if action.is_human]
        selected_action_name = st.selectbox(
            "Select tool",
            options=action_names,
            key="selected_action",
            disabled=st.session_state.waiting_for_spinner,
            help="Select a tool to invoke"
            if not st.session_state.waiting_for_spinner
            else "Waiting for something else to finish...",
        )

        # Find the selected action
        selected_action = None
        for action in available_actions:
            if action.name == selected_action_name:
                selected_action = action
                break

        # Render the tool
        if selected_action:
            _make_tool_form(selected_action)


def render_display_elements(display_elements: List[DisplayElement]):
    """
    Render a list of display elements
    """
    for element in display_elements:
        st_fn, st_kwargs = display_element_to_streamlit(element)
        st_fn(**st_kwargs)


def render_specification():
    """Get specification"""
    if st.session_state.spec.initial_specification_multimodal is not None:
        render_display_elements(st.session_state.spec.initial_specification_multimodal)
    else:
        st.markdown(st.session_state.spec.current_specification)


def render_specification_banner():
    """
    Render a banner with the current specification, if available.
    """
    text = getattr(st.session_state.spec, "current_specification", None)
    if text:
        st.info("*Your task:* " + text)


def sidebar_specification():
    """
    Display the task signature in the sidebar
    """

    if st.session_state.interaction_started:
        text = getattr(st.session_state.spec, "current_specification", "")
        if text is None:
            return
        text = f"<div class='no-copy monospace'><pre class='literal'>\n\n{text}</pre></div>"
        render_fn = st.html

    else:
        text = (
            "*After the interaction starts, you will be able to see task details here.*"
        )
        render_fn = st.markdown

    with st.sidebar.expander("How to maximize your score", expanded=True):
        render_fn(text)


def sidebar_instructions():
    """
    Display the dataset instructions in a collapsed sidebar
    """
    with st.sidebar.expander(
        "Instructions (copied from previous screen)", expanded=False
    ):
        st.session_state.spec.render_task_explanation()


@st.fragment
def chat_bubble(
    message: dict,
    show_quick_actions: bool = False,
    show_raw_message: bool = False,
    autovalidate: bool = False,
    autoscore: bool = False,
    show_response_time: bool = False,
):
    """
    Display a chat bubble for a message
    """
    if message["content"] is None:
        return

    with st.chat_message(message["role"]):
        if message["role"] == "user":
            if message.get("dialog_content", None):

                @st.dialog(message["content"], width="large")
                def dialog_content():
                    render_display_elements([message["dialog_content"]])

                st.button(
                    f"*Attached content:* {message['content']}",
                    on_click=dialog_content,
                    width="content",
                    key=f"dialog_{uuid.uuid4()}",
                    icon="📎",
                    type="secondary",
                )

            else:
                st.markdown(
                    message["content"],
                    unsafe_allow_html=True,
                )
        else:
            st.session_state.spec.render_msg_fn(message["content"])

            if show_raw_message:
                with st.expander("View raw assistant message"):
                    st.code(message["content"], wrap_lines=True)
            if show_response_time and message.get("response_time", None) is not None:
                st.html(
                    f"<div style='text-align: right; font-size: 0.8em; color: #777; font-style: italic'>Response time: {message['response_time']:.0f} seconds</div>"
                )
            if (
                autovalidate or show_quick_actions
            ) and st.session_state.spec.contains_solution(message["content"]):
                is_valid, error_msg = validate_message(message["content"])
                with st.container(horizontal=True, horizontal_alignment="right"):
                    text = (
                        ":green[:material/check:] This output is valid and passes basic checks."
                        if is_valid
                        else f":red[:material/close:] {error_msg}"
                    )
                    st.markdown(
                        f'<div class="validation-container">\n\n{text}</div>',
                        unsafe_allow_html=True,
                    )
                    if show_quick_actions:
                        create_quick_actions(message["content"])


def chat_conversation(
    messages: List[dict],
    show_quick_actions: bool = False,
    show_raw_message: bool = False,
    autovalidate: bool = False,
    autoscore: bool = False,
    empty_message_text: str = "No messages to display. Start chatting below! Remember, the assistant initially knows nothing about your goals.",
    show_response_time: bool = False,
):
    """
    Display a dialog of messages
    """
    # render the initial shared state for the user's sake
    if st.session_state.spec.initial_shared_state is not None:
        for label, obj in st.session_state.spec.initial_shared_state:
            chat_bubble(
                {
                    "role": "user",
                    "content": label,
                    "dialog_content": obj,
                },
                show_response_time=show_response_time,
            )

    # render the conversation history
    nonempty_messages = [m for m in messages if m["content"] is not None]
    if len(nonempty_messages) == 0:
        with st.container(
            horizontal_alignment="center",
            vertical_alignment="center",
            height=300,
            border=False,
        ):
            st.write(
                f"<center><i>{empty_message_text}</i></center>",
                unsafe_allow_html=True,
            )
        return
    for message in nonempty_messages:
        chat_bubble(
            message,
            show_quick_actions=show_quick_actions,
            show_raw_message=show_raw_message,
            autovalidate=autovalidate,
            autoscore=autoscore,
            show_response_time=show_response_time,
        )


def experiment_progress_bar():
    """
    Display a progress bar for the experiment
    """
    progress = (st.session_state.round_index + 1) / (st.session_state.total_rounds)
    text = f"Task {st.session_state.round_index + 1} / {st.session_state.total_rounds}: {st.session_state.dataset_names[st.session_state.round_index]}"
    with st.container(key="progress_bar"):
        st.progress(progress, text=text)


def validate_message(message_content: str):
    """
    Validate a message
    """
    try:
        is_valid = st.session_state.spec.validity_fn(
            message_content, raise_errors=True
        )[0]
        return is_valid, ""
    except Exception as e:
        return False, str(e)


def score_message(message_content: str):
    """
    Score a message
    """
    try:
        score = st.session_state.spec.reward_fn(message_content, raise_errors=True)[0]
        st.session_state.score_history.append((score, message_content, None))
        return score, ""
    except Exception as e:
        st.session_state.score_history.append((float("-inf"), message_content, str(e)))
        return float("-inf"), str(e)


def create_quick_actions(message_content: str):
    """
    Creates "Quick actions" for assistant messages that call
    the reward_fn / validity_fn tools.

    Args:
        message_content: The content of the message to evaluate
    """
    # Create a container for the evaluation button that's visually separate from the message
    with st.container(horizontal_alignment="right", horizontal=True):
        result = _invoke_tool(
            {"solution_attempt": message_content},
            button_text="Score solution",
            requested_tool_name="score_solution",
            help_text="Click to score the solution in this message",
            button_type="primary",
        )
        if result is not None:
            st.write(
                f"<div class = 'validation-container'>Score: {result}</div>",
                unsafe_allow_html=True,
            )


@st.fragment(run_every=2)
def countdown():
    """
    Countdown timer
    """
    with st.container(key="countdown"):
        st.write(
            f"{st.session_state.interaction_budget - (time.time() - st.session_state.interaction_start_time):.0f} seconds remaining",
        )


@st.fragment(run_every=1)
def brainstorm_countdown():
    """
    Countdown timer for brainstorming stage
    """
    if getattr(st.session_state, "brainstorm_time", None) is None:
        return
    if st.session_state.get("brainstorm_start_time", None) is None:
        return
    with st.container(key="countdown"):
        remaining = st.session_state.brainstorm_time - (
            time.time() - st.session_state.brainstorm_start_time
        )
        st.write(f"{max(0, remaining):.0f} seconds remaining")


######### Helper functions ################


def _get_tool_state_key(tool_name: str, kwargs: dict):
    """
    Get the state key for the tool.
    """
    kwargs_hash = hash(str(sorted(kwargs.items())))
    return f"tool_result_{tool_name}_{kwargs_hash}"


def _get_target_tool(requested_tool_name: str):
    """
    Get the target action for the requested tool name.
    """
    available_actions = st.session_state.simulator.actions
    target_tool = None
    for action in available_actions:
        if action.fn.name == requested_tool_name:
            target_tool = action.fn
            break
    if target_tool is None:
        raise ValueError(f"Tool {requested_tool_name} not found")
    return target_tool


def _invoke_tool_inner(
    tool_name: str,
    kwargs: dict,
    is_score_solution: bool,
    state_key: str,
):
    with st.spinner("Executing tool..."):
        try:
            result = _get_target_tool(tool_name).invoke(kwargs)
        except Exception as e:
            result = str(e)
            status = "error"
        else:
            status = "success"

    st.session_state[state_key] = result
    st.session_state.tool_history[st.session_state.policy.turn_count].append(
        ToolCall(
            name=tool_name,
            kwargs=json.dumps(kwargs),
            response=str(result),
            status=status,
        )
    )

    # Special logic for score_solution
    if is_score_solution:
        if status == "success":
            st.session_state.score_history.append(
                (str(result), kwargs["solution_attempt"], None)
            )
        else:
            st.session_state.score_history.append(
                ("-inf", kwargs["solution_attempt"], str(result))
            )


def _invoke_tool(
    kwargs: dict,
    requested_tool_name: str,
    button_text: str = "Execute tool",
    help_text: str = "Click to invoke the tool",
    button_type: str = "primary",
) -> str:
    """
    Displays a button to invoke the tool and returns the result
    Invokes the requested tool with the given kwargs.
    If the requested_tool_name is None, invoke the score_solution tool.

    Args:
        kwargs: Arguments to pass to the tool
        requested_tool_name: Name of the tool to invoke, or None for score_solution
        button_text: Text to display on the button
    """
    tool_name = requested_tool_name
    is_score_solution = False

    if (
        isinstance(st.session_state.spec, FixedSpecification)
        and tool_name == "score_solution"
    ):
        is_score_solution = True
        if st.session_state.spec._reward_fn_tool_name is not None:
            tool_name = st.session_state.spec._reward_fn_tool_name
    if (
        tool_name == "check_if_solution_is_valid"
        and st.session_state.spec._validity_fn_tool_name is not None
    ):
        tool_name = st.session_state.spec._validity_fn_tool_name

    state_key = _get_tool_state_key(tool_name, kwargs)
    disabled = state_key in st.session_state or st.session_state.waiting_for_spinner
    count_same_key = 0
    while f"button_{state_key}_{count_same_key}" in st.session_state:
        count_same_key += 1
    if not disabled:
        st.button(
            button_text,
            disabled=disabled,
            help=(
                help_text
                if not st.session_state.waiting_for_spinner
                else "Waiting for something else to finish..."
            ),
            key=f"button_{state_key}_{count_same_key}",
            on_click=lambda: _invoke_tool_inner(
                tool_name, kwargs, is_score_solution, state_key
            ),
            type=button_type,
        )
    return st.session_state.get(state_key, None)


def score_tracker():
    """
    Section that tracks the (score, solution) history
    """
    scores = [score for score, _, _ in st.session_state.score_history]

    # search for the floating point number in the score
    chart_data = [
        round(float(re.search(r"\d*(\.\d+)?", str(score)).group(0)), 1)
        if score != "-inf"
        else -1
        for score in scores
        if re.search(r"\d*(\.\d+)?", str(score))
    ]
    delta = (
        f"{chart_data[-1] - chart_data[-2]}"
        if len(st.session_state.score_history) > 1
        else None
    )
    with st.sidebar:
        st.metric(
            label="Current score",
            value=scores[-1] if len(scores) > 0 else "n/a",
            delta=delta,
            chart_data=chart_data,
            chart_type="line",
            border=False,
            help="The score of the last recorded output from the assistant",
        )

        st.write("Solution attempts")

        with st.container(
            key="version_history",
            horizontal_alignment="center",
            vertical_alignment="center"
            if len(st.session_state.score_history) == 0
            else "top",
            border=True,
            height=300,
        ):
            if len(st.session_state.score_history) == 0:
                st.markdown(
                    ":small[*When a message in the chat is scored, it will appear in this box.*]",
                )
                return

            for i, (score, solution, error_msg) in reversed(
                list(enumerate(st.session_state.score_history))
            ):

                @st.dialog(f"Solution attempt {i + 1} (Score: {score})", width="large")
                def solution_dialog(solution: str):
                    if error_msg is not None:
                        st.error(error_msg)

                    st.session_state.spec.render_msg_fn(solution)

                st.button(
                    f":small[Attempt {i + 1} (Score: {score})]",
                    on_click=solution_dialog,
                    args=(solution,),
                    width="stretch",
                )


def _make_tool_form(action):
    """
    Make a form for a given tool
    """
    tool = action.fn

    # Display docstring
    st.markdown(f"*{tool.description}*")

    # Parse arguments from the tool
    args_schema = tool.args_schema.schema()["properties"]

    # Initialize kwargs outside the conditional block
    kwargs = {}

    if len(args_schema) > 0:
        st.markdown("**Arguments:**")

        # Create input fields for each argument
        for field_name, field_info in args_schema.items():
            field_type = field_info.get("type", "string")
            field_description = field_info.get("description", "")

            if field_type == "string":
                value = st.text_area(
                    f"{field_name}",
                    key=f"arg_{field_name}",
                    height=200,
                    disabled=st.session_state.waiting_for_spinner,
                    help="Waiting for something else to finish..."
                    if st.session_state.waiting_for_spinner
                    else field_description,
                )
                if value:
                    kwargs[field_name] = value
            elif field_type == "integer":
                value = st.number_input(
                    f"{field_name}",
                    key=f"arg_{field_name}",
                    step=1,
                    disabled=st.session_state.waiting_for_spinner,
                    help="Waiting for something else to finish..."
                    if st.session_state.waiting_for_spinner
                    else field_description,
                )
                kwargs[field_name] = value
            elif field_type == "number":
                value = st.number_input(
                    f"{field_name}",
                    key=f"arg_{field_name}",
                    step=0.1,
                    disabled=st.session_state.waiting_for_spinner,
                    help="Waiting for something else to finish..."
                    if st.session_state.waiting_for_spinner
                    else field_description,
                )
                kwargs[field_name] = value
            elif field_type == "boolean":
                value = st.checkbox(
                    f"{field_name}",
                    key=f"arg_{field_name}",
                    disabled=st.session_state.waiting_for_spinner,
                    help="Waiting for something else to finish..."
                    if st.session_state.waiting_for_spinner
                    else field_description,
                )
                kwargs[field_name] = value
            else:
                # Default to text input for unknown types
                value = st.text_input(
                    f"{field_name}",
                    key=f"arg_{field_name}",
                    disabled=st.session_state.waiting_for_spinner,
                    help="Waiting for something else to finish..."
                    if st.session_state.waiting_for_spinner
                    else field_description,
                )
                if value:
                    kwargs[field_name] = value

    # if not st.session_state.interaction_started:
    #     st.markdown("Tools cannot be run before the chat session starts.")
    #     return

    result = _invoke_tool(kwargs, tool.name)
    if result is not None:
        if isinstance(result, pd.DataFrame):
            st.dataframe(result)
        else:
            st.write(result)


def floating_tools():
    """
    Render tools as floating buttons in the bottom right corner
    """

    if not st.session_state.simulator.actions:
        return

    with st.container(key="floating_button"):
        # Single gear button that opens a popover with tool buttons
        with st.popover(
            ":material/construction: Tools",
            help="Helper tools for solving the task",
            use_container_width=False,
        ):
            # Define the dialog that renders a form for a given tool
            def render_tool_dialog(action):
                @st.dialog(f"{action.name}", width="large")
                def tool_dialog():
                    _make_tool_form(action)

                return tool_dialog

            # List all tools in a scrollable area
            with st.container(height=200, border=False):
                for action in st.session_state.simulator.actions:
                    if not action.is_human:
                        continue
                    dialog_fn = render_tool_dialog(action)
                    st.button(
                        f":small[{action.name}]",
                        on_click=dialog_fn,
                        width="stretch",
                        type="primary",
                    )


@st.fragment
def carousel(
    display_fns: List[callable],
    include_select_button: bool = False,
    select_on_click: callable = None,
    noun: str = "option",
    height: int = 700,
):
    """
    Given a list of display functions, display them in a carousel.
    A carousel has a previous button, a center display, and a next button.
    """
    if display_fns is None or len(display_fns) == 0:
        return

    # Create a fairly stable key for this carousel instance so multiple carousels can coexist
    instance_key = f"carousel_{abs(hash(tuple(id(fn) for fn in display_fns)))}"
    index_key = f"{instance_key}_index"

    if index_key not in st.session_state:
        st.session_state[index_key] = 0

    num_items = len(display_fns)

    def go_prev():
        st.session_state[index_key] = (st.session_state[index_key] - 1) % num_items

    def go_next():
        st.session_state[index_key] = (st.session_state[index_key] + 1) % num_items

    def go_to(i: int):
        st.session_state[index_key] = i % num_items

    # Layout: [Prev] [Content] [Next]
    with st.container(key=f"{instance_key}_container"):
        # Lightweight CSS for styling
        st.html(
            f"""
<style>
.st-key-{instance_key}_dots button {{
  height: 28px;
  min-width: 28px;
  padding: 2px 6px;
  border-radius: 999px;
  transition: transform 120ms ease, opacity 120ms ease;
}}
.st-key-{instance_key}_dots button:hover, .st-key-carousel-arrow-{instance_key}-right button:hover, .st-key-carousel-arrow-{instance_key}-left button:hover {{ transform: translateY(-1px); }}
.st-key-carousel-arrow-{instance_key}-right button, .st-key-carousel-arrow-{instance_key}-left button {{
  height: 36px;
  min-width: 36px;
  border-radius: 999px;
  text-align: center;
  width: 100%;
}}
.st-key-{instance_key}_center_content {{
    padding: 3em;
}}
</style>
"""
        )

        left, center, right = st.columns([1, 10, 1], vertical_alignment="center")
        with left:
            with st.container(key=f"carousel-arrow-{instance_key}-left"):
                st.button(
                    "",
                    key=f"{instance_key}_prev",
                    on_click=go_prev,
                    help="Previous",
                    disabled=st.session_state.get("waiting_for_spinner", False),
                    icon=":material/arrow_back_ios:",
                    type="tertiary",
                    width="content",
                )

        with center:
            st.write(
                f"<center><i><small>{noun.capitalize()} {st.session_state[index_key] + 1} / {num_items}</small></i></center>",
                unsafe_allow_html=True,
            )

            with st.container(key=f"{instance_key}_center", border=True, height=height):
                current_index = st.session_state[index_key]
                # Display the current item
                try:
                    # Extra wrapper for a card feel
                    with st.container(
                        key=f"{instance_key}_center_content", border=False
                    ):
                        display_fns[current_index]()
                except Exception as e:
                    st.error(f"Error rendering carousel item {current_index}: {e}")

            # Dot navigation
            with st.container(
                key=f"{instance_key}_dots",
                horizontal=True,
                horizontal_alignment="center",
            ):
                for i in range(num_items):
                    st.button(
                        "●" if i == st.session_state[index_key] else "○",
                        key=f"{instance_key}_dot_{i}",
                        on_click=go_to,
                        args=(i,),
                        help=f"Go to {noun.capitalize()} {i + 1}",
                        type="tertiary",
                        width=50,
                    )

        with right:
            with st.container(key=f"carousel-arrow-{instance_key}-right"):
                st.button(
                    "",
                    key=f"{instance_key}_next",
                    on_click=go_next,
                    help="Next",
                    disabled=st.session_state.get("waiting_for_spinner", False),
                    icon=":material/arrow_forward_ios:",
                    type="tertiary",
                    width="content",
                )

    if include_select_button:
        with st.container(
            key=f"{instance_key}_select", horizontal=True, horizontal_alignment="center"
        ):
            if st.button(
                f"Select {noun.capitalize()} {st.session_state[index_key] + 1}",
                key=f"{instance_key}_select_btn",
                type="secondary",
                help=f"Select the current {noun.lower()}",
            ):
                if select_on_click is not None:
                    select_on_click(st.session_state[index_key])
