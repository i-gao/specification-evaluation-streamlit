import functools
from utils.code_sandbox import run_python_script_with_json_input
from PIL import Image
import os
import html
from utils.misc import parse_code
from data.design2code.parser import is_html
import streamlit as st
from typing import List, Dict, Any
import subprocess

try:
    subprocess.run(["playwright", "install-deps"], check=True)
    subprocess.run(["playwright", "install"], check=True)
except Exception:
    pass


@functools.lru_cache(maxsize=50)
def _render_html(
    code: str, docker_image: str, docker_container_id: str, test_id: int, root_dir: str = "/sandbox"
) -> str:
    """
    Renders the given HTML code and returns the path to the rendered image.
    """
    try:
        # Prefer local reward_utils directory as working dir when not using Docker
        script_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward_utils")
        effective_root = (
            script_root if (docker_image is None and docker_container_id is None) else root_dir
        )
        print(effective_root)
        run_id, log, output_filenames = run_python_script_with_json_input(
            input_dict={"predicted_html": code, "test_id": test_id},
            command="python screenshot_single.py --html {input_filename} --png _solution_output.png",
            docker_image=docker_image,
            docker_container_id=docker_container_id,
            output_filenames=["_solution_output.png"],
            root_dir=effective_root,
        )
    except Exception as e:
        raise e

    output_file = output_filenames[0]
    output_path = (
        os.path.join(effective_root, output_file)
        if (docker_image is None and docker_container_id is None)
        else output_file
    )
    img = Image.open(output_path)
    try:
        os.remove(output_path)
    except Exception:
        pass
    return img


def render_output(
    msg: str,
    docker_image: str = None,
    docker_container_id: str = None,
    test_id: str = None,
    inject_styles_raw_code: str = "",
):
    """
    Convert a message containing HTML code to markdown.
    Shows both the code (in a collapsible section) and the rendered image.

    Args:
        msg: The message containing the HTML code
        docker_image: Docker image for rendering HTML (optional)
        docker_container_id: Docker container ID for rendering HTML (optional)
        test_id: Test ID for the design2code task (optional)
    """
    # Extract HTML code from the message
    html_code, start_end = parse_code(
        msg, language="html", return_start_end=True, return_none_if_no_code=True
    )
    if html_code is None:
        if is_html(msg):
            html_code = msg
            start_end = (0, len(msg))
        else:
            st.write(msg)
            return

    st.write(msg[: start_end[0]])

    # Create markdown with rendered image
    try:
        # Use the existing render function to get the image
        rendered_image = _render_html(
            html_code, docker_image, docker_container_id, test_id
        )

        # Put image in buffer for streamlit
        from io import BytesIO

        buffer = BytesIO()
        rendered_image.save(buffer, format="PNG")

        # Create image HTML with base64 data
        st.image(buffer, width="stretch", caption="Generated web design")
    except Exception:
        st.write(
            ":red-background[:material/error: *The website code had errors in it and did not compile.*]"
        )

    with st.expander("Raw HTML code"):
        st.code(html.escape(html_code))

    if start_end[1] < len(msg):
        st.write(msg[start_end[1] :])


def render_comparison(
    y1: str,
    y2: str,
    docker_image: str,
    docker_container_id: str,
    test_id: str,
    **kwargs,
) -> str:
    """
    Compare two HTML codes and return a markdown string.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## Design A")
        render_output(y1, docker_image, docker_container_id, test_id)
    with col2:
        st.markdown("## Design B")
        render_output(y2, docker_image, docker_container_id, test_id)

def render_eval(
    *,
    final_prediction: str,
    docker_image: str,
    docker_container_id: str,
    test_id: str,
):
    """
    Render the evaluation for the design2code task.
    """
    import streamlit as st
    from utils.streamlit_types import FormElement, form_element_to_streamlit

    st.markdown("## Evaluate the assistant's design2code task")
    with st.container(key="design2code_eval_display", width="stretch"):
        render_output(final_prediction, docker_image, docker_container_id, test_id)

    form_elements: List[FormElement] = [
        FormElement(
            input_type="text_area",
            label="Describe the pros and cons of the design in a few sentences.",
            height=120,
        ),
    ]

    with st.form(key="design2code_custom_eval_form"):
        feedback: Dict[str, Any] = {}
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
                    if not feedback.get(label):
                        st.error("Please fill in all required fields.")
                        return False, None
    return False, None