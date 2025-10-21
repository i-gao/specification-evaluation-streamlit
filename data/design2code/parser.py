from bs4 import BeautifulSoup
import functools
from utils.code_sandbox import run_python_script_with_json_input
from PIL import Image
import os
import html
from utils.misc import parse_code

def parse_html(s: str) -> str:
    c = parse_code(s, language="html")
    if c is None:
        if is_html(s):
            c = s
        else:
            return None
    c = html.unescape(c)
    return c


def is_html(s: str) -> bool:
    return bool(BeautifulSoup(s, "html.parser").find())


@functools.lru_cache(maxsize=50)
def _render_html(
    code: str, docker_image: str, docker_container_id: str, test_id: int
) -> str:
    """
    Renders the given HTML code and returns the path to the rendered image.
    """
    code = parse_html(code)
    if code is None:
        raise ValueError("Could not parse HTML code from message")
    try:
        # Use local reward_utils directory as working dir when running without Docker
        script_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward_utils")
        local_root_dir = script_root if (docker_image is None and docker_container_id is None) else "/sandbox"

        run_id, log, output_filenames = run_python_script_with_json_input(
            input_dict={"predicted_html": code, "test_id": test_id},
            command="python screenshot_single.py --html {input_filename} --png _solution_output.png",
            docker_image=docker_image,
            docker_container_id=docker_container_id,
            output_filenames=["_solution_output.png"],
            root_dir=local_root_dir,
        )
    except Exception as exc:
        raise exc

    output_file = output_filenames[0]
    output_path = (
        os.path.join(local_root_dir, output_file)
        if (docker_image is None and docker_container_id is None)
        else output_file
    )
    img = Image.open(output_path)
    try:
        os.remove(output_path)
    except Exception:
        pass
    return img
