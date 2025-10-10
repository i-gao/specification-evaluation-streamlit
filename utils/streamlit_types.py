from typing import TypedDict, Literal, Optional, List, Tuple

"""
Types for datasets in data/ to specify Streamlit components to use for the form elements.
"""


class FormElement(TypedDict, total=False):
    input_type: Literal[
        "text_input",
        "text_area",
        "slider",
        "number_input",
        "checkbox",
        "radio",
        "selectbox",
        "select_slider",
        "multiselect",
        "text",
        "stars",
        "toggle",
    ]
    label: str
    required: bool


def form_element_to_streamlit(element: FormElement) -> Tuple[callable, dict]:
    """
    Convert a form element to a streamlit component.
    """
    import streamlit as st

    input_type = element["input_type"]
    required = element.get("required", False)

    if input_type == "text":
        element["body"] = element["label"]
        keys = ["body"]
        component = st.text
    elif input_type == "text_input":
        keys = ["label", "value", "help", "max_chars", "placeholder"]
        component = st.text_input
    elif input_type == "slider":
        keys = ["label", "min_value", "max_value", "value", "step", "help"]
        component = st.slider
    elif input_type == "number_input":
        keys = ["label", "min_value", "max_value", "value", "step", "help"]
        component = st.number_input
    elif input_type == "text_area":
        keys = ["label", "value", "height", "help", "max_chars", "placeholder"]
        component = st.text_area
    elif input_type == "checkbox":
        keys = ["label", "value", "help"]
        component = st.checkbox
    elif input_type == "radio":
        keys = ["label", "options", "index", "help", "horizontal"]
        component = st.radio
        if "-" != element["options"][0]:
            element["options"] = ["-"] + element["options"]
    elif input_type == "selectbox":
        keys = ["label", "options", "index", "help", "accept_new_options"]
        component = st.selectbox
    elif input_type == "select_slider":
        keys = ["label", "options", "value", "help"]
        component = st.select_slider
    elif input_type == "multiselect":
        keys = ["label", "options", "value", "help"]
        component = st.multiselect
    elif input_type == "toggle":
        keys = ["label", "value", "help"]
        component = st.toggle
    elif input_type == "stars":
        keys = ["label", "key", "options"]
        element["options"] = "stars"
        def stars(**kwargs):
            st.text(kwargs.pop("label"))
            return st.feedback(**kwargs)
        component = stars
    else:
        raise ValueError(f"Invalid input type: {element.get('input_type')}")

    kwargs = {}
    for k in keys:
        if k in element:
            kwargs[k] = element[k]

    return component, kwargs, required


class DisplayElement(TypedDict, total=False):
    input_type: Literal[
        "markdown",
        "dataframe",
        "table",
        "json",
        "code",
        "image",
    ]
    value: str


def display_element_to_streamlit(element: DisplayElement) -> Tuple[callable, dict]:
    """
    Convert a display element to a streamlit component.
    """
    import streamlit as st

    input_type = element["input_type"]
    value = element["value"]

    if input_type == "markdown":
        return st.markdown, {"body": value, "unsafe_allow_html": True}
    elif input_type == "dataframe":
        return st.dataframe, {
            "data": value,
            "hide_index": element.get("hide_index", False),
        }
    elif input_type == "table":
        return st.table, {"data": value}
    elif input_type == "json":
        return st.json, {"body": value}
    elif input_type == "code":
        return st.code, {"body": value}
    elif input_type == "image":
        return st.image, {"image": value}
    else:
        raise ValueError(f"Invalid input type: {element.get('input_type')}")
