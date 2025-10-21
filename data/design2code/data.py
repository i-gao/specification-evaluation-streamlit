from datasets import load_dataset
import numpy as np
from collections import defaultdict
import tqdm
from typing import List, Tuple, Optional, Dict, Callable, Any
import dill
import sys
import os
import sys
import glob
from PIL import Image
import re
import json
from langchain_core.tools import tool
import uuid
import streamlit as st

from data.dataset import (
    SpecificationCollection,
    FixedSpecification,
    CustomSpecification,
    Specification,
    FormElement,
)
from utils.streamlit_types import DisplayElement
from utils.misc import get_recursive, parse_json, subset_data
from data.actions import Action

from utils.model import init_langchain_model, encode_image_as_user_msg
from utils.code_sandbox import run_python_script_with_json_input

from data.design2code.parser import is_html, _render_html, parse_html
import data.design2code.streamlit_render as renderer


DEV_FRAC = 0.25
DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))

FIXED_INSTRUCTIONS = """
### What you need to prompt the assistant to do
In this task, **your goal is to get the assistant to write code to generate a requested static webpage.** You will receive a description of the design. Your task is to get the assistant to write a page as close as possible to the design.

To put a blue rectangle on the page, ask the assistant to insert an image called "rick.jpg".

### The tricky part
The description of the design describes the idea for the webpage. However, some nitty-gritty details may not have been specified (e.g., colors, fonts). You will have to make some choices about these details.

The score for the generated webpage will be between 0 and 100. If the score is 100, the webpage exactly matches the design. If the score is 0, the webpage is completely different from the design.
"""

FMT_INSTRUCTIONS = "To render a preview of the webpage, include the HTML wrapped in ```html\n...\n``` tags."
COMMONSENSE_INSTRUCTIONS = "Websites must be static and cannot use networking / Javascript. To render a blue rectangle, you can use an image named 'rick.jpg'. All code will go into a single, self-contained HTML file."


def render_fixed_task_explanation():
    """Render the fixed task explanation for design2code."""
    st.markdown(FIXED_INSTRUCTIONS)
    st.markdown(COMMONSENSE_INSTRUCTIONS)


def render_custom_task_explanation():
    """Render the custom task explanation for design2code."""
    st.markdown("### What you need to prompt the assistant to do")
    st.markdown(
        "In this task, **your goal is to get the assistant to write code to create a personal website for you.** The website must be a single page. Think about what you might want a website for, and prompt the assistant to write the code to create it."
    )
    st.markdown(
        "When the assistant generates a webpage for you, you will see it displayed in the chat. If the assistant writes code that is not valid, you will see an error message like the following."
    )

    st.markdown(
        ":red-background[:material/error: *The website code had errors in it and did not compile.*]"
    )

    st.markdown(COMMONSENSE_INSTRUCTIONS)


class Design2CodeDataset(SpecificationCollection):
    """
    The Design2Code benchmark evaluates how well LMs can generate HTML code
    that matches a given design.

    The reward fn ranges from [0, 1] where 1 is the best score.

    Paper: https://arxiv.org/pdf/2403.03163
    Original code: https://github.com/NoviScl/Design2Code

    Dev set split: 10% of cases in dev set
    """

    @property
    def dataset_name(self) -> str:
        return "design2code"

    @property
    def dataset_pretty_name(self) -> str:
        return "Web Design"

    @property
    def dataset_description(self) -> str:
        return "Work with the assistant to **design a personal webpage.**"

    @property
    def assets_file_id(self) -> str:
        return "1KaUNq9GwDf6p3aBrvLvNP5JXr2kU61M4"

    @property
    def default_docker_images(self) -> List[Dict[str, str]]:
        return [
            {
                "image_name": "design2code",
                "dockerfile_path": "data/design2code/reward_utils/Dockerfile",
                "build_context": "data/design2code",
                "description": "Docker image for Design2Code code evaluation",
            }
        ]

    def _create_user_expertise_form(self) -> List[FormElement]:
        """Create the user expertise form for design2code."""
        return [
            FormElement(
                input_type="radio",
                label="How familiar are you with HTML and CSS?",
                options=["Beginner", "Intermediate", "Advanced", "Expert"],
                default="Intermediate",
                required=True,
                help="This helps us understand your web development experience level",
            )
        ]

    def _create_user_evaluation_form(self) -> List[FormElement]:
        """Create the user evaluation form for design2code."""
        return [
            FormElement(
                input_type="radio",
                label="Do you prefer the **layout** of design A or B?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Do you prefer the **colors** of design A or B?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Do you prefer the **fonts** of design A or B?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Do you prefer the **text** of design A or B?",
                options=["A", "neutral", "B"],
                required=True,
            ),
            FormElement(
                input_type="radio",
                label="Which page design would you be more likely to pay for: A or B?",
                options=["A", "neutral", "B"],
                required=True,
            ),
        ]

    def __init__(
        self,
        dev: bool = False,
        docker_image: str = None,
        judge_model_name: str = "gpt-4o-mini",
        fixed_indexes: Optional[List[int]] = None,
        custom_indexes: Optional[List[int]] = None,
        persist_docker_container: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(dev=dev, **kwargs)

        # Load all the tasks
        task_paths = list(
            sorted(glob.glob(f"{DATASET_ROOT}/assets/testset_final/*_blocks.pkl"))
        )
        task_paths = subset_data(task_paths, DEV_FRAC, 1.0, dev)
        self._task_paths = task_paths
        self._custom_intents = json.load(
            open(f"{DATASET_ROOT}/assets/custom_intents.json")
        )
        self.fixed_length = len(task_paths)
        self.custom_length = len(self._custom_intents)
        self._docker_image = docker_image
        # self._judge_model_name = judge_model_name
        # self._judge_model = init_langchain_model(judge_model_name)
        self._persist_docker_container = persist_docker_container

        # All subclasses must have these attributes set
        self._finish_init()

        if fixed_indexes is not None:
            self._load_fixed_specs(fixed_indexes)
        if custom_indexes is not None:
            self._load_custom_specs(custom_indexes)

    def _load_fixed_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, FixedSpecification]:
        if indexes is None:
            return {}

        specs = {}
        for ix in indexes:
            task_path = self._task_paths[ix]
            id = re.search(r"(\d+)_blocks.pkl", task_path).group(1)
            blocks = dill.load(open(task_path, "rb"))
            image = Image.open(task_path.replace("_blocks.pkl", ".png"))
            html = open(task_path.replace("_blocks.pkl", ".html")).read()
            summary = open(task_path.replace("_blocks.pkl", "_summary.txt")).read()

            signature = "The task is to write HTML code to generate the requested webpage design; note that one must replace all images with placeholder source 'rick.jpg'."
            signature += "\n\n" + summary

            theta = "(Note: if the summary above mentions dark blue rectangles, they are actually images with the src set to 'rick.jpg'.)\n<chunk>\n"
            theta = "Here are details about all text on the page. (Note that bounding boxes are in x_ratio, y_ratio, w_ratio, h_ratio format, where ratio is the ratio of the block's width/height to the width/height of the screen):\n"
            for block in blocks:
                theta += f"- `{block['text']}` in RGB color `{block['color']}` with bbox `{block['bbox']}`.\n<chunk>\n"

            # persist one docker container
            if self._persist_docker_container and self._docker_image is not None:
                from llm_sandbox import SandboxSession

                session = SandboxSession(image=self._docker_image)
                session.open()
                container_id = session.container.id
            else:
                container_id = None

            spec = FixedSpecification(
                dataset_name=self.dataset_name,
                index=f"fixed_{ix}",
                full_specification=theta,
                initial_specification=summary,
                validity_fn=validity_fn,
                validity_kwargs={
                    "test_id": id,
                    "docker_image": self._docker_image,
                    "docker_container_id": container_id,
                },
                validity_fn_tool_name="check_html_validity",
                validity_fn_tool_description="Check if the HTML code compiles and renders without errors",
                reward_fn=reward_fn,
                reward_kwargs={
                    "test_id": id,
                    "docker_image": self._docker_image,
                    "docker_container_id": container_id,
                },
                reward_fn_tool_name="score_html_design",
                reward_fn_tool_description="Score the HTML code against the design requirements",
                msg_fmt_instructions=FMT_INSTRUCTIONS,
                prediction_fmt_instructions=FMT_INSTRUCTIONS,
                commonsense_description=COMMONSENSE_INSTRUCTIONS,
                ystar=f"```html\n{html}\n```",
                # metric_name=None,  # Not provided
                # baseline_scores=None,  # Not provided
                render_task_explanation=render_fixed_task_explanation,
                # actions=get_actions(
                #     docker_image=self._docker_image,
                #     docker_container_id=container_id,
                #     test_id=id,
                #     text_description=summary,
                #     model_name=self._judge_model_name,
                #     judge_model=self._judge_model,
                # ),
                render_msg_fn=lambda msg: renderer.render_output(
                    msg,
                    docker_image=self._docker_image,
                    docker_container_id=container_id,
                    test_id=id,
                ),
                name=f"design2code_{ix}",
                container_ids=[container_id],
                user_expertise_form=self._create_user_expertise_form(),
            )
            spec._container_ids = [container_id]
            specs[ix] = spec
        return specs

    def _load_custom_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, CustomSpecification]:
        """Load custom specifications for design2code."""
        if indexes is None:
            return {}

        specs = {}
        for ix in indexes:
            # Generate a unique fake test_id for the custom spec so it can still render HTML
            fake_test_id = f"custom_{uuid.uuid4().hex[:8]}"
            custom_intent = self._custom_intents[ix]["intent"]
            y0_file = f"{DATASET_ROOT}/assets/y0/custom_{ix}_*.html"
            y0_files = glob.glob(y0_file)
            y0_files = [open(y0_file).read() for y0_file in y0_files]

            if self._persist_docker_container and self._docker_image is not None:
                from llm_sandbox import SandboxSession

                session = SandboxSession(image=self._docker_image)
                session.open()
                container_id = session.container.id
            else:
                container_id = None

            spec = CustomSpecification(
                dataset_name=self.dataset_name,
                index=f"custom_{ix}",
                commonsense_description=COMMONSENSE_INSTRUCTIONS,
                validity_fn=validity_fn,
                validity_kwargs={
                    "test_id": fake_test_id,
                    "docker_image": self._docker_image,
                    "docker_container_id": container_id,
                },
                validity_fn_tool_name="check_html_validity",
                validity_fn_tool_description="Check if the HTML code compiles and renders without errors",
                initial_specification=f"Create a {custom_intent}",
                msg_fmt_instructions=FMT_INSTRUCTIONS,
                prediction_fmt_instructions=FMT_INSTRUCTIONS,
                y0=y0_files,  # Not provided
                render_task_explanation=render_custom_task_explanation,
                actions=[],
                render_msg_fn=lambda msg: renderer.render_output(
                    msg,
                    docker_image=self._docker_image,
                    docker_container_id=container_id,
                    test_id=fake_test_id,
                ),
                render_comparison_fn=lambda y1,
                y2,
                **kwargs: renderer.render_comparison(
                    y1, y2, self._docker_image, container_id, fake_test_id, **kwargs
                ),
                render_evaluation_fn=lambda **kwargs: renderer.render_eval(
                    **kwargs,
                    docker_image=self._docker_image,
                    docker_container_id=container_id,
                    test_id=fake_test_id,
                ),
                render_evaluation_kwargs={"y0": y0_files},
                name=f"custom_design2code_{ix}",
                state_files=[],
                files_to_clean=[],
                container_ids=[container_id],
                user_expertise_form=self._create_user_expertise_form(),
                user_evaluation_form=self._create_user_evaluation_form(),
            )
            spec._container_ids = [container_id]
            specs[ix] = spec
        return specs


def validity_fn(
    yhat: str,
    test_id: int,
    docker_image: str,
    docker_container_id: str,
    raise_errors: bool = False,
) -> Tuple[bool, dict]:
    """
    Check if the HTML code compiles and renders without errors.

    Args:
        yhat: The predicted HTML code
        test_id: The test ID for the task
        docker_image: Docker image for rendering
        docker_container_id: Docker container ID for rendering
        raise_errors: Whether to raise errors on failure

    Returns:
        Tuple[bool, dict]: (is_valid, metadata)
    """
    parsed_yhat = parse_html(yhat)
    if parsed_yhat is None:
        if raise_errors:
            raise ValueError("Could not parse HTML code from message")
        return False, {
            "error": "No HTML code found",
            "violated_constraints": ["No HTML code found"],
        }

    is_valid, metadata = is_html(parsed_yhat), {"parsed_code": parsed_yhat}
    if not is_valid:
        if raise_errors:
            raise ValueError("Invalid HTML code")
        return False, {
            "error": "Invalid HTML code",
            "violated_constraints": ["Invalid HTML code"],
            **metadata,
        }
    else:
        return True, metadata


def reward_fn(
    yhat: str,
    test_id: int,
    docker_image: str,
    docker_container_id: str,
    raise_errors: bool = False,
) -> Tuple[float, dict]:
    """
    Evaluate the HTML code against the design and return a score.

    Args:
        yhat: The predicted HTML code
        test_id: The test ID for the task
        docker_image: Docker image for rendering
        docker_container_id: Docker container ID for rendering
        raise_errors: Whether to raise errors on failure

    Returns:
        Tuple[float, dict]: (score, metadata)
    """
    parsed_yhat = parse_html(yhat)
    if parsed_yhat is None:
        if raise_errors:
            raise ValueError("No code found in the response")
        return float("-inf"), {"evaluation_error": "No code found in the response"}

    # Execute
    try:
        # When not using Docker, run from local reward_utils directory so relative imports work
        reward_utils_dir = os.path.join(DATASET_ROOT, "reward_utils")
        local_root_dir = reward_utils_dir if (docker_image is None and docker_container_id is None) else "/sandbox"

        run_id, log, output_filenames = run_python_script_with_json_input(
            input_dict={"predicted_html": parsed_yhat, "test_id": test_id},
            command="python check_correctness.py {uuid}",
            docker_image=docker_image,
            docker_container_id=docker_container_id,
            input_filename="_solution_to_grade_{uuid}.json",
            output_filenames=["_solution_output_{uuid}.pkl"],
            root_dir=local_root_dir,
        )
    except Exception as e:
        if raise_errors:
            raise e
        return float("-inf"), {"evaluation_error": str(e)}

    output_file = output_filenames[0]
    output_path = (
        os.path.join(local_root_dir, output_file)
        if (docker_image is None and docker_container_id is None)
        else output_file
    )
    result = dill.load(open(output_path, "rb"))
    try:
        os.remove(output_path)
    except Exception:
        pass

    score = result["final_score"]
    metadata = {**result, "log": log}
    return score, metadata


def get_actions(
    docker_image: str,
    docker_container_id: str,
    test_id: int,
    text_description: str,
    model_name: str,
    judge_model,
) -> List[Action]:
    """
    Since OpenAI doesn't allow for image outputs from tools, we need to use another model
    call to get a text description of the generated webpage.
    """

    @tool(parse_docstring=True)
    def render_html(code: str) -> str:
        """
        Renders the given HTML code and returns the path to the rendered image.
        The code should be self-contained HTML code that can be executed in a browser.

        Args:
            code (str): The HTML code to render.
        """
        return _render_html(code, docker_image, docker_container_id, test_id)

    @tool(parse_docstring=True)
    def render_html_and_reflect_on_design(code: str) -> List[dict]:
        """
        Takes a screenshot of the webpage generated by the given code and compares
        it to the text description of the design. Returns a text reflection on the
        correctness of the generated webpage.
        The code should be self-contained HTML code that can be executed in a browser.

        Args:
            code (str): The HTML code to render.
        """
        code = parse_html(code)
        if code is None:
            raise ValueError("No code found in the input")
        image = _render_html(code, docker_image, docker_container_id, test_id)

        prompt = [
            {
                "role": "system",
                "content": "Compare the following screenshot of a webpage to the text description of the intended design. Go sentence by sentence through the text description, and reflect on how well the webpage matches the design. Do not offer suggestions for improvement.",
            },
            encode_image_as_user_msg(
                image=image, caption=text_description, model_name=model_name
            ),
        ]
        response = judge_model.invoke(prompt)

        try:
            os.remove("_solution_output.png")
        except:
            pass
        return response.content

    return [
        Action(
            fn=render_html_and_reflect_on_design,
            is_public=True,
            is_human=False,
            name="Render HTML and reflect on design",
        ),
        Action(
            fn=render_html,
            is_public=False,
            is_human=False,
            name="Render HTML",
        ),
    ]
