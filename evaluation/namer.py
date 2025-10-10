import datetime
import json
from typing import Dict, List, Tuple, Any
import os

ARGS = [
    "dataset",
    "spec_index",
]
KWARGS = [
    "interaction_budget",
    "user_first",
    "simulator",
    "simulator_model",
    "policy",
    "policy_model",
    "max_react_steps",
    "dataset_kwargs",
    "seed",
]
ARG_SEP = "_"
LIST_SEP = ","
SPACE = "-"


def _clean(s) -> str:
    """
    Clean the value
    Examples:
    - _clean(None) -> "None"
    - _clean("a") -> "a"
    - _clean(["a", "b", "c"]) -> "a,b,c"
    - _clean(1) -> "1"
    """
    if type(s) in (list, tuple):
        return LIST_SEP.join([_clean(x) for x in s])
    elif type(s) == str:
        return (
            s.replace(" ", SPACE)
            .replace(ARG_SEP, SPACE)
            .replace("=", SPACE)
            .replace("/", SPACE)
            .replace(LIST_SEP, SPACE)
        )
    elif s == None:
        return "None"
    elif type(s) in [int, float, bool]:
        return str(s)
    else:
        raise ValueError(f"Invalid type: {type(s)}")


def _fmt(k, v) -> str:
    """
    Format the key and value as a string
    Examples:
    - _fmt(None, "a") -> "_a"
    - _fmt("k", "v") -> "_k=v"
    - _fmt("k", ["a", "b", "c"]) -> "_k=a,b,c"
    - _fmt("k", {"a": "b", "c": "d"}) -> "_a=b_c=d"
    """
    kstr = "" if k is None else _clean(k) + "="
    if type(v) == dict:
        out = ""
        for vk, vv in v.items():
            out += f"{ARG_SEP}{_clean(vk)}={_clean(vv)}"
        return out
    else:
        return f"{ARG_SEP}{kstr}{_clean(v)}"


def get_experiment_name(
    fill_missing_with_star: bool = False, include_datetime: bool = False, **kwargs
) -> str:
    """
    Given the arguments and keyword arguments, return a unique experiment name
    """
    name = (
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "_"
        if include_datetime
        else ""
    )
    for arg in ARGS:
        if arg not in kwargs:
            if fill_missing_with_star:
                kwargs[arg] = "*"
            else:
                raise ValueError(f"Missing argument: {arg}")
        name += _fmt(None, kwargs[arg])
    for kwarg in KWARGS:
        if kwarg not in kwargs:
            if fill_missing_with_star:
                kwargs[kwarg] = "*"
            else:
                raise ValueError(f"Missing keyword argument: {kwarg}")
        name += _fmt(kwarg, kwargs[kwarg])
    return name.replace(ARG_SEP, "", 1)

def _unclean(s: str) -> Any:
    """
    Convert a cleaned string back to its original value
    Examples:
    - _unclean("None") -> None
    - _unclean("a") -> "a"
    - _unclean("a,b,c") -> ["a", "b", "c"]
    - _unclean("1") -> 1
    - _unclean("True") -> True
    """
    if s == "None":
        return None
    elif s == "True":
        return True
    elif s == "False":
        return False
    elif s.replace(".", "").isdigit():  # Handle both int and float
        return float(s) if "." in s else int(s)
    elif LIST_SEP in s:
        return [_unclean(x) for x in s.split(LIST_SEP)]
    else:
        return s

def get_args(path: str) -> Dict[str, Any]:
    """
    Given a path saved exactly as by get_experiment_name,
    get the arguments from the path
    """
    experiment_name = os.path.basename(path).rsplit(".", 1)[0]
    
    # Split into parts, removing any datetime prefix if present
    parts = experiment_name.split(ARG_SEP)
    if len(parts) > 0:
        try:
            # Try to parse the first part as a datetime
            datetime.datetime.strptime(parts[0], "%Y-%m-%d-%H-%M")
            parts = parts[1:]  # Remove datetime prefix if valid
        except ValueError:
            pass  # Not a datetime, keep the part
    
    args = {}
    
    # Process positional arguments first
    for i, arg in enumerate(ARGS):
        if i >= len(parts):
            break
        value = parts[i]
        if value == "*":
            continue
        args[arg] = _unclean(value)
    
    # Process keyword arguments
    for key in KWARGS:
        if f"{_clean(key)}=" in experiment_name:
            value = experiment_name.split(f"{_clean(key)}=")[1].split(ARG_SEP)[0]
            if value == "*":
                continue
            args[key] = _unclean(value)
    
    return args
    