import streamlit as st
import random
from typing import Callable, List
from utils.streamlit_types import FormElement, form_element_to_streamlit
from evaluation.qualitative_eval import (
    COMPARISON_LIKERT,
    ASSISTANT_INSTRUMENTS,
    INSTRUMENT_LIKERT,
)

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




def presurvey(
    should_show: Callable = None,
    validate: Callable = None,
    on_completion: Callable = None,
    user_expertise_form: List[FormElement] = None,
    user_specification_form_initial: List[FormElement] = None,
    include_trust_question: bool = False,
):
    """
    Form that appears at the beginning of the experiment.
    Elicits the user's expertise level and initial specification (in the custom case).
    """
    if should_show is not None and not should_show():
        return

    form_values = {
        "expertise": {},
        "specification": {},
    }
    with st.form(key="presurvey_form"):
        for form_element in user_expertise_form:
            st_fn, st_kwargs, req = form_element_to_streamlit(form_element)
            o = st_fn(**st_kwargs)
            if form_element["input_type"] != "text":
                form_values["expertise"][form_element["label"]] = o

        if user_specification_form_initial is not None:
            for form_element in user_specification_form_initial:
                st_fn, st_kwargs, req = form_element_to_streamlit(form_element)
                o = st_fn(**st_kwargs)
                if form_element["input_type"] != "text":
                    form_values["specification"][form_element["label"]] = o

        if include_trust_question:
            form_values["trust"] = st.radio(
                "How much do you agree with this statement? 'I think working with the assistant will be more efficient than using a web browser to solve the task myself.'",
                options=["-"] + INSTRUMENT_LIKERT,
            )

        if st.form_submit_button("Submit", type="primary"):
            valid = validate is None or validate(form_values)
            if not valid:
                st.error("Please fill in all fields correctly")
                return
            if on_completion is not None:
                on_completion(form_values)


def brainstorming(
    should_show: Callable = None,
    validate: Callable = None,
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
        st.write("Reflect on the task. How will you decide if the assistant's solution is good? What requirements, likes, and dislikes do you have?")
        notes = st.text_area("Write your thoughts here", height=250)

        submitted = st.form_submit_button("Continue to presurvey", type="primary")
        if submitted:
            form_values = {"notes": notes}
            valid = validate is None or validate(form_values)
            if not valid:
                return
            if on_completion is not None:
                on_completion(form_values)

def message_feedback(
    should_show: Callable = None,
    validate: Callable = None,
    on_completion: Callable = None,
):
    """
    Form that appears beneath a message from the assistant,
    asking the user to evaluate the message.
    """
    if should_show is not None and not should_show():
        return

    with st.form(key=f"feedback_form_{len(st.session_state.messages) - 1}"):
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
            "message_index": len(st.session_state.messages) - 2,
        }

        if st.form_submit_button("Submit Feedback", type="primary"):
            valid = validate is None or validate(feedback)
            if not valid:
                st.error("Please fill in all fields correctly")
                return
            if on_completion is not None:
                on_completion(feedback)


def custom_final_specification(
    should_show: Callable = None,
    validate: Callable = None,
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
            valid = validate is None or validate(form_values)
            if not valid:
                st.error("Please fill in all fields correctly")
                return

            if on_completion is not None:
                on_completion(form_values)


def comparison_scoring(
    should_show: Callable = None,
    validate: Callable = None,
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
            options=["", "Yes", "Maybe", "No"],
        )

        if st.form_submit_button("Submit", type="primary"):
            valid = validate is None or validate(score_answers)
            if not valid:
                st.error(
                    f"Please make sure to fill in all fields and spend at least {st.session_state.evaluation_minimum / 60:.1f} minutes on the evaluation."
                )
                return
            if on_completion is not None:
                on_completion(score_answers)


def interaction_evaluation(
    should_show: Callable = None,
    validate: Callable = None,
    on_completion: Callable = None,
):
    """
    Custom score elicitation form and exit_survey form
    """
    if should_show is not None and not should_show():
        return

    form_results = {}
    with st.form(key="interaction_evaluation_form"):
        flat_instruments = [
            item for sublist in ASSISTANT_INSTRUMENTS.values() for item in sublist
        ]
        random.shuffle(flat_instruments)

        for instrument in flat_instruments:
            form_results[instrument] = st.radio(
                instrument,
                options=["-"] + INSTRUMENT_LIKERT,
            )

        form_results["comments"] = st.text_area(
            "(Optional) Additional comments",
        )

        if st.form_submit_button("Submit", type="primary"):
            valid = validate is None or validate(form_results)
            if not valid:
                st.error("Please fill in all fields correctly")
                return
            if on_completion is not None:
                on_completion(form_results)


def final_prediction_evaluation(
    *,
    likert_label: str = 'How much do you agree with this statement: "I would rather accept this solution as is than continue my search with the assistant for 10 more minutes."',
    stars_label: str = "Rate the overall quality.",
    text_area_label: str = "If you were to continue working the assistant for 10 more minutes, what would you want to change?",
    submit_key: str = "custom_eval_second_page_form",
):
    """
    Generic second-page evaluation: render the final prediction, then ask 3 questions.
    Returns (completed, feedback) like dataset-specific renderers.
    """
    from utils.streamlit_types import FormElement, form_element_to_streamlit
    from evaluation.qualitative_eval import INSTRUMENT_LIKERT

    # Render the final prediction view
    st.session_state.spec.render_msg_fn(st.session_state.final_prediction)

    form_elements = [
        FormElement(
            input_type="stars",
            label=stars_label,
        ),
        FormElement(
            input_type="radio",
            label=likert_label,
            options=["-"] + INSTRUMENT_LIKERT,
        ),
        FormElement(
            input_type="text_area",
            label=text_area_label,
            height=120,
        ),
    ]

    with st.form(key=submit_key):
        feedback = {}
        for element in form_elements:
            st_fn, st_kwargs, required = form_element_to_streamlit(element)
            value = st_fn(**st_kwargs)
            label = element.get("label", "question")
            feedback[label] = value
        submit = st.form_submit_button("Submit", type="primary")
        if submit:
            for element in form_elements:
                if element.get("required", False):
                    label = element.get("label")
                    if (
                        not feedback.get(label)
                        or feedback.get(label) == ""
                        or feedback.get(label) == "-"
                    ):
                        st.error("Please fill in all required fields.")
                        return False, None
            return True, feedback

    return False, None
