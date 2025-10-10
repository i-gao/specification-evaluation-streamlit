import time
import torch
import numpy as np
from ast import literal_eval
from argparse import Action
import pickle
import json
import os
import re
from time import perf_counter
import pandas as pd
from collections import defaultdict
from typing import List, Any, Dict, Union, Tuple, Literal
import hashlib
import zlib
from collections import Counter
import random
from json_repair import repair_json
from difflib import SequenceMatcher

FILE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ParseKwargs(Action):
    """
    Helper function s.t. argparse can parse kwargs of the form --kwarg key1=value1 key2=value2
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for pair in values:
            key, value = pair.split("=")
            processed_value = infer_type(value)
            getattr(namespace, self.dest)[key] = processed_value


def infer_type(s):
    """
    If the str can be interpreted as a float or an int, convert it to that type.
    """
    try:
        return str_to_bool(s)
    except:
        pass
    try:
        return literal_eval(s)
    except:
        pass
    try:
        return str_to_torchdtype(s)
    except:
        pass
    try:
        return str_to_list(s)
    except:
        return s


def str_to_torchdtype(value):
    if not value.startswith("torch."):
        raise Exception(f"Invalid torch dtype: {value}")
    return getattr(torch, value.split(".")[1])


def str_to_list(value):
    """
    Helper function to parse a string of the form "[x,y,z]" into a list [x, y, z].
    Catches some cases where ast.literal_eval fails because the elements in the list
    contain non-standard characters.
    """
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    else:
        raise Exception(f"Invalid list value: {value}")

    return value.split(",")


def str_to_bool(value):
    """
    Function to parse boolean values from argparse, but allow for '*' and 'None'
    """
    if value.lower() in ("true", "t", "yes", "y"):
        return True
    elif value.lower() in ("false", "f", "no", "n"):
        return False
    elif value.lower() == "none":
        return None
    elif value.lower() == "*":
        return "*"
    else:
        raise Exception(f"Invalid boolean value: {value}")


def str_to_int(value):
    """
    Function to parse int values from argparse, but allow for '*' and 'None'
    """
    try:
        return int(value)
    except:
        if value.lower() == "none":
            return None
        elif value.lower() == "*":
            return "*"
        else:
            raise Exception(f"Invalid int value: {value}")


def seed_everything(seed: int):
    """
    Helper function to seed everything.
    """
    if type(seed) != int:
        return

    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_recursive(d, key, default=None):
    """
    Return nested attribute of dict
    """
    if key == "":
        return d
    i = key.find(".")
    if i < 0:
        return d.get(key, default)
    else:
        return get_recursive(d.get(key[:i], {}), key[i + 1 :], default)


def hash(x: object, type="md5"):
    """
    Hash an object.
    """
    # encode the object
    if isinstance(x, torch.Tensor):
        encoded = x.numpy().tobytes()
    elif isinstance(x, np.ndarray):
        encoded = x.tobytes()
    elif isinstance(x, str):
        encoded = x.encode("utf-8")
    elif isinstance(x, pd.DataFrame):
        encoded = pickle.dumps(x.to_dict(orient="records"))
    else:
        encoded = pickle.dumps(x)
    # hash the encoded object
    if type == "md5":
        return hashlib.md5(encoded).hexdigest()
    elif type == "sha256":
        return hashlib.sha256(encoded).hexdigest()
    else:
        return zlib.adler32(encoded)


def collate(list_of_dicts):
    """
    Collate a list of dictionaries into a single dictionary.
    """
    collated_dict = defaultdict(list)
    for d in list_of_dicts:
        for k, v in d.items():
            collated_dict[k].append(v)
    return collated_dict


class Stopwatch:
    """
    Context manager for timing a block of code
    Source: https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
    """

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.time = perf_counter() - self.time


def parse_json(json_str, return_start_end=False):
    """
    Parse a JSON string, returning None if it fails.
    If return_start_end is True, also returns a tuple of (start, end) of the JSON string.
    """
    if json_str is None:
        if return_start_end:
            return None, None
        else:
            return None

    # replace occurrences of 2+ \ with a single \
    json_str = re.sub(r"\\{1,}n", r"\n", json_str)  # \\\\n -> \n
    json_str = re.sub(r"\\{1,}'", "'", json_str)  # \\\' -> \'
    json_str = re.sub(r'\\{1,}"', '"', json_str)  # \\\" -> \"

    if "```json" in json_str:
        start_end = (json_str.find("```json"), json_str.rfind("```") + 3)
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        start_end = (json_str.find("```"), json_str.rfind("```") + 3)
        json_str = json_str.split("```")[1].split("```")[0].strip()
    elif "{" in json_str and "}" in json_str:
        start_end = (json_str.find("{"), json_str.rfind("}") + 1)
        bracket_start_end = (json_str.find("["), json_str.rfind("]") + 1)
        if (bracket_start_end[0] != -1) and (bracket_start_end[0] < start_end[0]):
            start_end = bracket_start_end
        json_str = json_str[start_end[0] : start_end[1]]
    else:
        start_end = None

    json_str = repair_json(json_str)

    try:
        js = json.loads(json_str)
    except:
        try:
            js = literal_eval(json_str)
        except:
            try:
                import demjson3

                js = demjson3.decode(json_str)
            except:
                js = None

    if return_start_end:
        return js, start_end
    else:
        return js


def parse_code(
    code_str, language: str = None, return_none_if_no_code=True, return_start_end=False
):
    """
    Parse a code string, returning None if it fails.
    """
    if language is not None:
        LANGUAGES = [language]
    else:
        LANGUAGES = ["python", "bash", "html", "plaintext", "json"]

    for lang in LANGUAGES:
        if f"```{lang}" in code_str:
            start_end = (
                code_str.find(f"```{lang}"),
                code_str.rfind("```") + len(f"```{lang}"),
            )
            code_str = code_str.split(f"```{lang}")[1].split("```")[0].strip()
            x = code_str.replace("\r\n", "\n")
            if return_start_end:
                return x, start_end
            else:
                return x

    if "```" in code_str:
        start_end = (code_str.find("```"), code_str.rfind("```") + 3)
        code_str = code_str.split("```")[1].split("```")[0].strip()
        x = code_str.replace("\r\n", "\n")
        if return_start_end:
            return x, start_end
        else:
            return x

    if return_none_if_no_code:
        if return_start_end:
            return None, None
        else:
            return None
    else:
        if return_start_end:
            return code_str, (0, len(code_str))
        else:
            return code_str


def parse_for_answer_tags(
    text,
    keyword="answer",
    return_start_end=False,
    return_none_if_not_found=False,
    return_all=False,
):
    """
    Looks for <keyword>X</keyword> in text and returns X.
    """
    match = re.findall(rf"<{keyword}>(.*?)</{keyword}>", text, re.DOTALL)
    if match:
        if return_start_end:
            return match[0].strip(), (
                text.find(f"<{keyword}>"),
                text.find(f"</{keyword}>") + len(f"</{keyword}>"),
            )
        if return_all:
            return [item.strip() for item in match]
        else:
            return match[0].strip()
    else:
        if return_none_if_not_found:
            if return_start_end:
                return None, None
            return None
        else:
            if return_start_end:
                return text, (0, len(text))
            return text


def parse_list(text) -> List[str]:
    """
    Looks for an ordered list of the form
        1. X
        2. Y
        3. Z
        ...
    or an unordered list of the form
        * X
        * Y
        * Z
    or
        - X
        - Y
        - Z
    and returns a list of strs [X, Y, Z]

    Also handles numbered lists without newlines:
        1. X 2. Y 3. Z
    FYI this mode may cause some issues with whitespaces being removed.
    """
    # First try the original pattern for multiline lists
    pattern = r"^\s*(?:\d+\.\s+|\d+\)\s+|\[\d+\]\s+|\* |- )([^\n]*?)(?=\s*\d+\.|$)"
    matches = re.findall(pattern, text, re.MULTILINE)

    # If we found matches but there are multiple numbered items on the same line,
    # use the single-line pattern instead
    if matches and len(re.findall(r"\d+\.", text)) > len(matches):
        matches = []

    # If no matches found, try single-line numbered list pattern
    if not matches:
        # Pattern for numbered items that might be on the same line
        # Matches: 1. item 2. item 3. item
        # Use a regex that captures content between numbered patterns
        single_line_pattern = r"\d+\.\s*([^0-9]*?)(?=\s*\d+\.|$)"
        matches = re.findall(single_line_pattern, text)
    return matches


def compute_majority_class_accuracy(y: np.ndarray):
    """
    Computes the accuracy achieved by always predicting the majority class.
    This is a deterministic baseline that predicts the most frequent class.

    Args:
        y: Array of true labels

    Returns:
        float: The accuracy of the majority class predictor
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    class_probabilities = counts / len(y)  # P(y)
    # The accuracy is simply the frequency of the most common class
    return np.max(class_probabilities)


def compute_random_sampling_accuracy(y: np.ndarray):
    """
    Computes the expected accuracy of a classifier that randomly samples predictions
    from the empirical class distribution.

    Args:
        y: Array of true labels

    Returns:
        float: The expected accuracy of the random sampling predictor
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    class_probabilities = counts / len(y)  # P(y)

    # For a random sampling classifier, the accuracy is the sum of squared probabilities
    return np.sum(class_probabilities**2)


def subset_data(
    x: Union[List[Any], Dict[str, Any]],
    dev_frac: float = 1,
    frac: float = 1,
    dev: bool = True,
    max_len: int = None,
    shuffle: bool = False,
    seed: int = None,
):
    """
    Subsets data into dev and test sets.
    Args:
        x: List of data
        dev_frac: Fraction of data to use for dev set
        frac: Fraction of subset to use (for debugging purposes to save compute)
        dev: Whether to subset dev set
        max_len: Maximum length of data to use
        shuffle: Whether to shuffle the data
        seed: Seed for shuffling
    Returns:
        Subset of data
    """
    was_dict = isinstance(x, dict)
    if was_dict:
        x = list(x.items())

    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(x)

    if dev:
        x = x[: int(dev_frac * len(x))]
    else:
        x = x[int(dev_frac * len(x)) :]

    x = x[: int(frac * len(x))]
    if max_len is not None:
        x = x[:max_len]

    if was_dict:
        x = {k: v for k, v in x}

    return x


def iou_score(A, B):
    """
    Computes the IoU of two lists. Duplicates are allowed; the IOU is computed as if
    every occurrence of an element in the list is a unique element.
    Example:
        A = [T, T, F] => [T1, T2, F1]
        B = [T, F, F] => [T1, F1, F2]
        intersection = [T1, F1]
        union = [T1, T2, F1, F2]
        iou = 2 / 4 = 0.5
    Example:
        A = [T, T, T, T]
        B = [T, T, F, F]
        intersection = [T, T]
        union = [T, T, T, T, F, F]
        iou = 2 / 6 = 0.333
    """
    # Count occurrences of each value in both lists
    count_A = Counter(A)
    count_B = Counter(B)

    # Compute intersection size (minimum count for each value)
    intersection_size = sum(min(count_A[val], count_B[val]) for val in set(A) & set(B))

    # Compute union size (sum of all counts)
    union_size = sum(count_A.values()) + sum(count_B.values()) - intersection_size

    # Return IoU score
    return intersection_size / union_size if union_size else 0.0


def add_section(
    header: str,
    text: str,
    style: Literal["markdown", "divider", "code"] = "divider",
    divider_char: str = "=",
    indent_level: int = 0,
) -> str:
    """
    Formats text as a section with a header.
    """
    text = text.strip() if text is not None else ""
    if style == "markdown":
        return "#" * (indent_level + 1) + " " + header + "\n" + text
    elif style == "divider":
        indent = "\t" * indent_level
        return (
            indent
            + divider_char * 10
            + " "
            + header
            + " "
            + divider_char * 10
            + "\n"
            + text
        )
    elif style == "code":
        indent = "\t" * indent_level
        return indent + "```" + header.lower() + "\n" + text + "\n```"


# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ORANGE = "\033[38;5;216m"  # A softer, more pale orange
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    NONE = ""


def print_debug(
    message: str, function_name: str, color: str = "NONE", indent_level: int = 0
):
    """Print a debug message with function context and color.

    Args:
        message: The message to print (can be multiline)
        function_name: Name of the function generating the message
        color: Color to use for the function name (default: "NONE")
        indent_level: Number of indentation levels (default: 0)
    """
    message = str(message)
    color_code = getattr(Colors, color.upper(), Colors.NONE)
    indent = "\t" * indent_level

    # Split message into lines and indent each line
    lines = message.split("\n")
    indented_lines = [f"{indent}{line}" for line in lines]
    indented_message = "\n".join(indented_lines)

    print(f"{color_code}[{function_name}] {indented_message}{Colors.ENDC}")


def import_from_string(class_path: str):
    """
    Import a class from a module path.
    Example:
        import_from_string("user_simulator.free_agent.FreeAgentUser")
        => FreeAgentUser
    Returns:
        The class
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    class_ = getattr(module, class_name)
    return class_


def _clean_for_json(obj):
    """
    Recursively clean an object to ensure it's JSON serializable.

    Args:
        obj: Any object to clean

    Returns:
        JSON serializable version of the object
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    elif hasattr(obj, "isoformat"):
        # Handle datetime objects
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):
        # Handle custom objects by converting to dict
        return _clean_for_json(obj.__dict__)
    elif hasattr(obj, "tolist"):
        # Handle numpy arrays
        return obj.tolist()
    elif hasattr(obj, "item"):
        # Handle numpy scalars
        return obj.item()
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        # Handle other iterables (sets, etc.)
        return [_clean_for_json(item) for item in obj]
    else:
        # Convert everything else to string
        return str(obj)


def check_if_file_ending_exists(ending: str, output_dir: str) -> bool:
    """
    Check if a file of the format {output_dir}/*{ending} exists.
    """
    for file in os.listdir(output_dir):
        if file.endswith(ending):
            return True
    return False


def remove_datetime(filename: str) -> str:
    """
    Remove the datetime from a filename.
    Example:
        2025-07-15-16-02_{filename}.json
        => {filename}.json
    """
    return re.sub(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}_", "", filename)


def fuzzy_match(
    x: str,
    k: str,
    threshold: float = 0.8,
    ignore_case: bool = False,
    ignore_newlines: bool = False,
    ignore_whitespace: bool = False,
    ignore_punctuation: bool = False,
    allow_substring: bool = False,
) -> bool:
    """
    Check if string x is similar to string k using sequence matching.

    Args:
        x: Input string
        k: Key string to compare against
        threshold: Similarity threshold (0.0 to 1.0)

    Returns:
        True if similarity >= threshold, False otherwise
    """
    import ftfy
    import string

    def _normalize(s: str) -> str:
        s = ftfy.fix_text(s)
        if ignore_case:
            s = s.lower()
        if ignore_newlines:
            s = s.replace("\n", "")
        if ignore_whitespace:
            s = re.sub(r"\s+", "", s)
        if ignore_punctuation:
            s = s.translate(str.maketrans("", "", string.punctuation))
        return s

    similarity = SequenceMatcher(None, _normalize(x), _normalize(k)).ratio()
    if similarity >= threshold:
        return True
    if allow_substring:
        return _normalize(x) in _normalize(k) and len(_normalize(x)) >= 50
    else:
        return False


def download_file_from_google_drive(
    file_id: str,
    output_path: str,
    unzip: bool = False,
    chunk_size: int = 8192,
    timeout: int = 600,
):
    """
    Download a file from Google Drive.
    """
    import gdown
    import zipfile

    os.makedirs(output_path, exist_ok=True)

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        gdown.download(url, output_path + f"/{file_id}.zip")
        print(f"File downloaded successfully to {output_path + f'/{file_id}.zip'}")
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

    if unzip:
        with zipfile.ZipFile(output_path + f"/{file_id}.zip", "r") as zip_ref:
            zip_ref.extractall(output_path)
        print(f"File unzipped successfully to {output_path}")
    return True


def build_docker_image(docker_info: dict, root_dir: str):
    """
    Build a docker image from a docker info dictionary.
    """
    import subprocess

    try:
        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                docker_info["image_name"],
                "-f",
                os.path.join(root_dir, docker_info["dockerfile_path"]),
                os.path.join(root_dir, docker_info["build_context"]),
            ],
        )
    except Exception as e:
        print(f"Error building docker image: {e}")
        return False
    return True


def check_docker_image_exists(image_name: str) -> bool:
    """
    Check if a docker image exists.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["docker", "images", "-q", image_name], capture_output=True
        )
        return len(out.stdout) > 0
    except subprocess.CalledProcessError:
        return False


def replace_tags_with_link(s: str, tag: str, href: str) -> str:
    """
    Given a string s = "<tag>X</tag>", return "<a href='href'>X</a>".
    """
    return re.sub(
        f"<{tag}>([^<>]*)</{tag}>", rf"<a class='btn' href='{href}'>\1</a>", s
    )


def parse_questions(content: str):
    """
    Parse questions from a string.
    This pattern finds complete question sentences that:
    - start at a sentence boundary (start of string, or after ., !, ? + space, or a newline),
    - start with a capital letter,
    - allow internal . / abbreviations like e.g. and parentheses,
    - do not cross a true sentence boundary of the form [.!?] <space> Capital
    """
    # Pattern explanation:
    # (?:^|(?<=[.!?])\s+|(?<=\n)\s*[-*•]?\s*) - Start of string, after sentence ending, or after newline with optional bullet
    # ([A-Z]) - Capture the first capital letter
    # ((?:(?![.!?]\s+[A-Z])[^?])*)  - Capture everything until ? while not crossing sentence boundaries
    # \? - The question mark itself
    pattern = re.compile(
        r'(?:^|(?<=[.!?])\s+|(?<=\n)\s*[-*•]?\s*)([A-Z](?:(?![.!?]\s+[A-Z])[^?\n])*\?)(?=\s|$|["\'])'
    )

    matches = pattern.findall(content)

    # Clean up the matches by stripping leading/trailing whitespace
    return [match.strip() for match in matches]
