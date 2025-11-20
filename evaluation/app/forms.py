import streamlit as st
import random
from typing import Callable, List
from utils.streamlit_types import FormElement, form_element_to_streamlit
from evaluation.qualitative_eval import (
    COMPARISON_LIKERT,
    ASSISTANT_INSTRUMENTS,
    INSTRUMENT_LIKERT,
    NASA_TLX_SCALES,
    NASA_TLX_MIN_VALUE,
    NASA_TLX_MAX_VALUE,
    NASA_TLX_STEP,
    MUST_HAVES_QUESTION,
    NICE_TO_HAVES_QUESTION,
)
from data import get_spec
import evaluation.app.components as components

"""
This file contains code to generate the form interfaces.
Functions take in:
- a should_show function that returns True if the form should be shown
    default: will show the form
- a validation callback that will be called with the form results
- a on_completion callback that will be called with the form results
    once the form is validated.
- **kwargs: additional arguments to pass to the form

Note that the form will always show unless on_completion modifies the result of should_show.
"""


def default_validation(form_values: dict) -> bool:
    """
    Default validation function that checks that all fields are filled.
    """
    return all(
        value is not None and value != "" and value != "-"
        for value in form_values.values()
    )


def presurvey(
    should_show: Callable = None,
    validate: Callable = default_validation,
    on_completion: Callable = None,
    user_expertise_form: List[FormElement] = None,
    include_trust_question: bool = False,
):
    """
    Form that appears at the beginning of the experiment.
    Elicits the user's expertise level.
    """
    if should_show is not None and not should_show():
        return

    form_values = {
        "expertise": {},
    }
    with st.form(key="presurvey_form"):
        for form_element in user_expertise_form:
            st_fn, st_kwargs, req = form_element_to_streamlit(form_element)
            o = st_fn(**st_kwargs)
            if form_element["input_type"] != "text":
                form_values["expertise"][form_element["label"]] = o

        if include_trust_question:
            form_values["trust"] = st.radio(
                "How much do you agree with this statement? 'I think working with the assistant will be more efficient than using a web browser to solve the task myself.'",
                options=["-"] + INSTRUMENT_LIKERT,
            )

        if st.form_submit_button("Submit", type="primary"):
            # Default validation: check that all fields are filled
            if validate is None:
                valid = True
                # Check expertise fields
                for key, value in form_values.get("expertise", {}).items():
                    if value == "-" or value == "" or value is None:
                        valid = False
                        break
                # Check trust question if included
                if valid and include_trust_question:
                    if (
                        form_values.get("trust") == "-"
                        or form_values.get("trust") == ""
                    ):
                        valid = False
            else:
                valid = validate(form_values)
            if not valid:
                st.error("Please fill in all fields correctly")
                return
            if on_completion is not None:
                on_completion(form_values)


def brainstorming(
    should_show: Callable = None,
    validate: Callable = default_validation,
    on_completion: Callable = None,
):
    """
    Brainstorming form that appears between instructions and presurvey.
    Shows a reflection prompt, a large text area, and enforces a countdown gate.
    The countdown duration is taken from st.session_state.brainstorm_time.
    """
    if should_show is not None and not should_show():
        return

    with st.form(key="brainstorming_form"):
        st.write(
            "Reflect on the task. How will you decide if the assistant's solution is good? What requirements, likes, and dislikes do you have?"
        )
        notes = st.text_area("Write your thoughts here", height=250)

        submitted = st.form_submit_button("Continue to presurvey", type="primary")
        if submitted:
            form_values = {"notes": notes}
            if validate is None:
                valid = True
            else:
                valid = validate(form_values)
            if not valid:
                st.error("Please write your thoughts before continuing")
                return
            if on_completion is not None:
                on_completion(form_values)


def message_feedback(
    message_index: int,
    should_show: Callable = None,
    validate: Callable = default_validation,
    on_completion: Callable = None,
):
    """
    Form that appears beneath a message from the assistant,
    asking the user to evaluate the message.

    Args:
        message_index: The index of the message in st.session_state.messages to evaluate
        should_show: Function that returns True if the form should be shown
        validate: Validation callback called with form results
        on_completion: Callback called with form results once validated
    """
    if should_show is not None and not should_show():
        return

    with st.form(key=f"feedback_form_{message_index}"):
        st.write(
            "**Please evaluate the most recent policy message.** Check all that apply:"
        )

        repetitive = st.checkbox("Contains repetitive content")
        nonsequitur = st.checkbox("Is non-sequitur")
        irrelevant = st.checkbox("Contains irrelevant content")
        expensive = st.checkbox("Contains a query which is hard to answer")
        ambiguous = st.checkbox(
            "Contains a query for which I don't have strong opinions (ambiguous question)"
        )
        filler = st.checkbox(
            "Is a filler message with no content (e.g. 'I'm thinking, please hang tight...')"
        )
        solution = st.checkbox("Contains a solution attempt to the task")
        explanation = st.checkbox("Explains something well to the user")

        feedback = {
            "repetitive": repetitive,
            "nonsequitur": nonsequitur,
            "irrelevant": irrelevant,
            "expensive": expensive,
            "ambiguous": ambiguous,
            "filler": filler,
            "solution": solution,
            "explanation": explanation,
            "message_index": message_index,
        }

        if st.form_submit_button("Submit Feedback", type="primary"):
            valid = validate is None or validate(feedback)
            if not valid:
                st.error("Please fill in all fields correctly")
                return
            if on_completion is not None:
                on_completion(feedback)


def message_thumbs_feedback(
    message_index: int,
    should_show: Callable = None,
    validate: Callable = default_validation,
    on_completion: Callable = None,
):
    """
    Form that appears beneath a message from the assistant,
    asking the user to rate the message with thumbs up/down.

    Uses Streamlit's st.feedback widget with "thumbs" option.
    Returns 0 for thumbs-down, 1 for thumbs-up, or None if not selected.

    Args:
        message_index: The index of the message in st.session_state.messages to evaluate
        should_show: Function that returns True if the form should be shown
        validate: Validation callback called with form results
        on_completion: Callback called with form results once validated

    Reference: https://docs.streamlit.io/develop/api-reference/widgets/st.feedback
    """
    if should_show is not None and not should_show():
        return

    feedback_key = f"thumbs_feedback_{message_index}"

    st.write("**Rate the most recent assistant message:**")

    # Define callback to handle feedback submission
    def handle_feedback():
        """Process feedback when user clicks thumbs up/down"""
        feedback_value = st.session_state.get(feedback_key)
        if feedback_value is None:
            return

        feedback = {
            "thumbs_rating": feedback_value,  # 0 = thumbs down, 1 = thumbs up
            "message_index": message_index,
        }

        # Validate if callback provided
        if validate is None:
            valid = True
        else:
            valid = validate(feedback)
        if not valid:
            return
        if on_completion is not None:
            on_completion(feedback)

    # Use st.feedback with thumbs option
    # Returns 0 for thumbs-down, 1 for thumbs-up, or None if not selected
    # The on_change callback handles processing when user clicks
    st.feedback(
        options="thumbs",
        key=feedback_key,
        on_change=handle_feedback,
    )


def custom_final_specification(
    should_show: Callable = None,
    validate: Callable = default_validation,
    on_completion: Callable = None,
    user_specification_form_final: List[FormElement] = None,
):
    if should_show is not None and not should_show():
        return

    if user_specification_form_final is None:
        if on_completion is not None:
            on_completion({})

    with st.form(key="custom_final_specification_form"):
        form_values = {}
        # User specification questions
        for form_element in st.session_state.spec.user_specification_form_final:
            st_fn, st_kwargs, req = form_element_to_streamlit(form_element)
            o = st_fn(**st_kwargs)
            if form_element["input_type"] != "text":
                form_values[form_element["label"]] = o

        if st.form_submit_button("Submit", type="primary"):
            # Default validation: check that all form elements are filled
            if validate is not None:
                valid = validate(form_values)
            else:
                valid = True
            if not valid:
                st.error("Please fill in all required fields correctly")
                return

            if on_completion is not None:
                on_completion(form_values)


def comparison_scoring(
    should_show: Callable = None,
    validate: Callable = default_validation,
    on_completion: Callable = None,
):
    if should_show is not None and not should_show():
        return

    score_answers = {}
    with st.form(key="custom_final_relative_scoring_form"):
        score_answers["free_write"] = st.text_area(
            "Describe the pros and cons of A vs. B in a few sentences."
        )

        # Show dataset specific questions if available
        if st.session_state.spec.user_evaluation_form:
            for form_element in st.session_state.spec.user_evaluation_form:
                st_fn, st_kwargs, req = form_element_to_streamlit(form_element)
                o = st_fn(**st_kwargs)
                if form_element["input_type"] != "text":
                    score_answers[form_element["label"]] = o

        # Show fixed questions
        score_answers["relative_score"] = st.radio(
            "Overall, do you prefer A or B?",
            options=[""] + COMPARISON_LIKERT,
            horizontal=True,
        )
        # score_answers["relative_score"] = st.slider(
        #     "Overall, how much **more** do you prefer Creation A **over** Creation B?",
        #     min_value=-100,
        #     max_value=100,
        #     value=0,
        #     format="A scores %.0f points more than B",
        #     help="0 means you prefer Creation A and Creation B equally, 100 means you prefer Creation A way more than Creation B, -100 means you prefer Creation B way more than Creation A",
        # )
        # score_answers["choice"] = st.radio(
        #     "Select the creation you prefer",
        #     options=["", "A", "B"],
        # )
        score_answers["confidence"] = st.radio(
            "Do you think that more exploration (with or without the assistant) could have led you to a better creation?",
            options=["-", "Yes", "Maybe", "No"],
        )

        if st.form_submit_button("Submit", type="primary"):
            # Default validation: check that all fields are filled
            if validate is None:
                valid = True
            else:
                valid = validate(score_answers)
            if not valid:
                st.error(
                    f"Please make sure to fill in all fields and spend at least {st.session_state.evaluation_minimum / 60:.1f} minutes on the evaluation."
                )
                return
            if on_completion is not None:
                on_completion(score_answers)


def post_specification_survey(
    should_show: Callable = None,
    validate: Callable = default_validation,
    on_completion: Callable = None,
):
    """
    Post-interaction specification survey form.
    Asks about must-haves and nice-to-haves for a good recommendation.
    """
    if should_show is not None and not should_show():
        return

    form_results = {}
    with st.form(key="post_specification_survey_form"):
        form_results["must_haves"] = st.text_area(
            MUST_HAVES_QUESTION,
            height=120,
        )
        form_results["nice_to_haves"] = st.text_area(
            NICE_TO_HAVES_QUESTION,
            height=120,
        )

        if st.form_submit_button("Submit", type="primary"):
            # Default validation: check that both text areas are filled
            if validate is None:
                valid = True
            else:
                valid = validate(form_results)
            if not valid:
                st.error("Please fill in both the must-haves and nice-to-haves fields")
                return
            if on_completion is not None:
                on_completion(form_results)


def assistant_instruments_survey(
    should_show: Callable = None,
    validate: Callable = default_validation,
    on_completion: Callable = None,
):
    """
    Custom score elicitation form and exit_survey form
    """
    if should_show is not None and not should_show():
        return

    form_results = {}
    with st.form(key="interaction_evaluation_form"):
        st.write("Answer the following questions about the assistant.")

        # assistant instruments
        flat_instruments = [
            item for sublist in ASSISTANT_INSTRUMENTS.values() for item in sublist
        ]
        random.shuffle(flat_instruments)

        for instrument in flat_instruments:
            form_results[instrument] = st.radio(
                instrument,
                options=["-"] + INSTRUMENT_LIKERT,
            )

        if st.form_submit_button("Submit", type="primary"):
            # Default validation: check that all instruments are answered (not "-")
            if validate is None:
                valid = True
            else:
                valid = validate(form_results)
            if not valid:
                st.error("Please answer all questions about the assistant")
                return
            if on_completion is not None:
                on_completion(form_results)


def final_prediction_evaluation(
    *,
    likert_label: str = None,
    stars_label: str = None,
    slider_label: str = None,
    text_area_label: str = None,
    submit_key: str = "custom_eval_second_page_form",
):
    """
    Generic second-page evaluation: render the final prediction, then ask 3 questions.
    Returns (completed, feedback) like dataset-specific renderers.
    """
    from utils.streamlit_types import FormElement, form_element_to_streamlit
    from evaluation.qualitative_eval import INSTRUMENT_LIKERT

    # Render the final prediction view
    st.markdown("Below is the assistant's final artifact for the task.")
    with st.container(border=True, height=700):
        st.session_state.spec.render_msg_fn(st.session_state.final_prediction)

    form_elements = [
        FormElement(
            input_type="text_area",
            label=MUST_HAVES_QUESTION,
            height=120,
        ),
        FormElement(
            input_type="text_area",
            label=NICE_TO_HAVES_QUESTION,
            height=120,
        ),
    ]
    if slider_label is not None:
        form_elements.append(
            FormElement(
                input_type="slider",
                label=slider_label,
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                help="0: Unusable, 100: Perfect",
            ),
        )
    if likert_label is not None:
        form_elements.append(
            FormElement(
                input_type="radio",
                label=likert_label,
                options=["-"] + INSTRUMENT_LIKERT,
            ),
        )
    if stars_label is not None:
        form_elements.append(
            FormElement(
                input_type="stars",
                label=stars_label,
            ),
        )
    if text_area_label is not None:
        form_elements.append(
            FormElement(
                input_type="text_area",
                label=text_area_label,
                height=120,
            ),
        )

    with st.form(key=submit_key):
        feedback = {}
        for element in form_elements:
            st_fn, st_kwargs, required = form_element_to_streamlit(element)
            value = st_fn(**st_kwargs)
            label = element.get("label", "question")
            feedback[label] = value
        submit = st.form_submit_button("Submit", type="primary")
        if submit:
            # Check all required fields and text areas
            for element in form_elements:
                label = element.get("label")
                value = feedback.get(label)
                # Text areas should not be empty
                if element.get("input_type") == "text_area":
                    if not value or value.strip() == "":
                        st.error(f"Please fill in the '{label}' field.")
                        return False, None
                # Required fields (radio buttons) should not be "-" or empty
                elif element.get("required", False):
                    if not value or value == "" or value == "-":
                        st.error("Please fill in all required fields.")
                        return False, None
            return True, feedback

    return False, None


def assistant_ranking_exit_survey(
    should_show: Callable = None,
    validate: Callable = default_validation,
    on_completion: Callable = None,
):
    """
    Exit survey form that shows all conversations and asks the user to rank
    which assistant they preferred (Assistant 1, 2, 3, etc. based on number of rounds).
    Also asks for a paragraph explanation.
    """
    if should_show is not None and not should_show():
        return

    # Get message history - each entry is [config_dict, messages_list]
    message_history = st.session_state.get("message_history", [])

    if len(message_history) == 0:
        st.warning("No conversation history found. Cannot display ranking survey.")
        if on_completion is not None:
            on_completion({})
        return

    num_rounds = len(message_history)

    with st.form(key="assistant_ranking_exit_survey_form", border=False):
        st.markdown("## Assistant Ranking Survey")
        st.markdown(
            f"You interacted with {num_rounds} assistant(s) across {num_rounds} round(s). "
            "Please review each conversation below and then rank which assistant you preferred."
        )

        # Store the original spec to restore later
        original_spec = st.session_state.get("spec", None)

        # Create tabs for each conversation
        tab_labels = [
            f"Assistant {round_idx + 1} - Round {round_idx + 1} ({config_dict.get('dataset_name', 'unknown')})"
            for round_idx, (config_dict, _) in enumerate(message_history)
        ]
        tabs = st.tabs(tab_labels)

        # Display each conversation in its tab
        for round_idx, (config_dict, messages) in enumerate(message_history):
            assistant_num = round_idx + 1
            dataset_name = config_dict.get("dataset_name", "unknown")
            spec_index = config_dict.get("spec_index", 0)
            dataset_kwargs = config_dict.get("dataset_kwargs", {})

            with tabs[round_idx]:
                # Temporarily load and set the spec for this conversation
                temp_spec = None
                try:
                    temp_spec = get_spec(
                        dataset_name,
                        spec_index,
                        **dataset_kwargs,
                        allow_multimodal_actions=True,
                    )
                    st.session_state.spec = temp_spec
                except Exception as e:
                    st.error(
                        f"Error loading spec for Assistant {assistant_num}: {str(e)}"
                    )
                    # Restore original spec before continuing
                    st.session_state.spec = original_spec
                    # Still try to display messages even if spec loading failed
                    st.markdown(
                        "*Note: Messages may not render correctly due to spec loading error.*"
                    )

                # Display the conversation
                if temp_spec is not None:
                    with st.container(border=True, height=700):
                        components.chat_conversation(
                            messages,
                            show_quick_actions=False,
                            show_raw_message=False,
                            autovalidate=False,
                            autoscore=False,
                            show_response_time=True,
                            empty_message_text=f"No messages recorded for Assistant {assistant_num}.",
                        )
                else:
                    # Fallback: display raw messages if spec couldn't be loaded
                    with st.container(border=True, height=700):
                        for msg in messages:
                            if msg.get("content") is not None:
                                with st.chat_message(msg.get("role", "user")):
                                    st.markdown(msg.get("content", ""))

        # Restore original spec
        st.session_state.spec = original_spec

        st.markdown("---")

        # Complete ranking question (A > B > C style)
        ranking = st.multiselect(
            "Rank the assistants from MOST to LEAST preferred:",
            [i for i in range(num_rounds)],
            format_func=lambda x: f"Assistant {x + 1}",
        )

        # Paragraph explanation
        explanation = st.text_area(
            "Please write a paragraph explaining your ranking:",
            height=150,
            placeholder="Explain what made you rank the assistants in this order.",
        )

        form_values = {
            "ranking": ranking,
            "explanation": explanation,
            "num_rounds": num_rounds,
        }

        if st.form_submit_button("Submit", type="primary"):
            # Default validation: ensure all assistants are ranked and explanation is provided
            valid = (
                len(form_values["ranking"]) == num_rounds
                and form_values["explanation"].strip() != ""
            )

            if not valid:
                if len(form_values["ranking"]) != num_rounds:
                    st.error(
                        f"Please rank all {num_rounds} assistant(s) from most to least preferred"
                    )
                else:
                    st.error(
                        "Please fill in all fields correctly (rank all assistants and write an explanation)"
                    )
                return
            if on_completion is not None:
                on_completion(form_values)


def nasa_tlx_survey(
    should_show: Callable = None,
    validate: Callable = default_validation,
    on_completion: Callable = None,
    include_pairwise_comparisons: bool = False,
):
    """
    NASA Task Load Index (NASA-TLX) survey form.

    The NASA-TLX is a subjective workload assessment tool that rates perceived workload
    across six dimensions: Mental Demand, Physical Demand, Temporal Demand, Performance,
    Effort, and Frustration Level.

    Each dimension is rated on a 0-100 scale with 5-point increments.

    Args:
        should_show: Function that returns True if the form should be shown
        validate: Validation callback called with form results
        on_completion: Callback called with form results once validated
        include_pairwise_comparisons: If True, includes pairwise comparison questions
            to weight the subscales (full NASA-TLX). If False, uses Raw TLX (no weighting).
    """
    if should_show is not None and not should_show():
        return

    form_results = {}
    with st.container(key="narrow_body"):
        with st.form(key="nasa_tlx_form", border=False):
            # Collect ratings for each of the 6 scales
            for scale_name, scale_info in NASA_TLX_SCALES.items():
                # Create slider with anchors
                st.write(f"**{scale_name}**: {scale_info['description']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"0: {scale_info['low_anchor']}")
                with col2:
                    st.caption(f"100: {scale_info['high_anchor']}")
                slider_value = st.slider(
                    scale_name,
                    min_value=NASA_TLX_MIN_VALUE,
                    max_value=NASA_TLX_MAX_VALUE,
                    value=50,  # Default to middle
                    step=NASA_TLX_STEP,
                    label_visibility="collapsed",
                    help=f"0: {scale_info['low_anchor']}, 100: {scale_info['high_anchor']}",
                    key=f"nasa_tlx_{scale_name}",
                )
                form_results[scale_name] = slider_value

            # Optional pairwise comparisons for weighting (full NASA-TLX)
            if include_pairwise_comparisons:
                st.markdown("### Pairwise Comparisons")
                st.markdown(
                    "For each pair below, select which dimension contributed more to your "
                    "workload during the task."
                )

                scale_names = list(NASA_TLX_SCALES.keys())

                # Generate all pairwise comparisons (15 total: C(6,2) = 15)
                all_comparisons = []
                for i in range(len(scale_names)):
                    for j in range(i + 1, len(scale_names)):
                        scale_a = scale_names[i]
                        scale_b = scale_names[j]
                        all_comparisons.append(
                            (scale_a, scale_b, f"{scale_a} vs {scale_b}")
                        )

                # Randomize order but keep it stable across page reloads
                pairwise_order_key = "nasa_tlx_pairwise_order"
                if pairwise_order_key not in st.session_state:
                    # Create a shuffled copy to avoid modifying the original
                    shuffled_comparisons = all_comparisons.copy()
                    random.shuffle(shuffled_comparisons)
                    st.session_state[pairwise_order_key] = shuffled_comparisons
                else:
                    # Use the stored order
                    shuffled_comparisons = st.session_state[pairwise_order_key]

                pairwise_results = {}
                for comparison_idx, (scale_a, scale_b, comparison_key) in enumerate(
                    shuffled_comparisons
                ):
                    choice = st.segmented_control(
                        "Which contributed more to your workload?",
                        options=["-", scale_a, scale_b],
                        selection_mode="single",
                        default=None,
                        key=f"pairwise_{comparison_idx}",
                    )
                    # Convert None to empty string for consistency with validation
                    pairwise_results[comparison_key] = (
                        choice if choice is not None else ""
                    )

                form_results["pairwise_comparisons"] = pairwise_results

            if st.form_submit_button("Submit", type="primary"):
                # Default validation: ensure all scales are rated
                if validate is None:
                    valid = True
                else:
                    valid = validate(form_results)

                if not valid:
                    st.error(
                        "Please rate all dimensions correctly"
                        + (
                            " and complete all pairwise comparisons"
                            if include_pairwise_comparisons
                            else ""
                        )
                    )
                    return
                if on_completion is not None:
                    on_completion(form_results)
