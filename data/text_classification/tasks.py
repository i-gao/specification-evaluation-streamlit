import pandas as pd
import os
import json
from typing import List, Tuple

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))

TASKS = ["bfrs", "ccc", "gvfc"]


def get_task(task_name: str) -> Tuple[pd.DataFrame, str, List[str], str, int]:
    """
    Returns (df: pd.DataFrame, description: str, label_set: List[str], codebook: str, num_labels_per_x: int)
    """
    df = pd.read_csv(f"{DATASET_ROOT}/assets/{task_name}.csv")
    task_info = json.load(open(f"{DATASET_ROOT}/assets/{task_name}_info.json"))

    df["input"] = df[task_info["input_column"]]
    df["label"] = df[task_info["label_column"]]

    df = df.dropna(subset=["input", "label"], how="any")

    return (
        df[["input", "label"]],
        task_info["description"],
        task_info["label_set"],
        task_info["codebook"],
        task_info["num_labels_per_x"],
    )
