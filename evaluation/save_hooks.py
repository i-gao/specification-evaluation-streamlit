import numpy as np

DEFAULT_SAVE_HOOKS = ["compute_monotonicity"]


def run_hook(hook: str, *args, **kwargs):
    """
    Run a hook on an interaction.
    """
    if hook == "compute_monotonicity":
        compute_monotonicity(*args, **kwargs)
    else:
        raise ValueError(f"Hook {hook} not found")


def compute_monotonicity(interaction):
    """
    Given a sequence of [(x, y)] points, measure "how monotonically increasing" the
    sequence is by sorting the x values and computing
    \sum_{i=1}^{n-1} 1{y_{i+1} >= y_i} / (n-1)

    This counts the proportion of times the y values are non-decreasing.
    """
    xs_ys = [
        (
            (turn.user_runtime_cost, turn.intermediate_grade.score)
            if turn.intermediate_grade is not None
            else (turn.user_runtime_cost, None)
        )
        for turn in interaction.turns
    ]
    # replace the last y with the final grade
    if interaction.final_grade is not None:
        xs_ys[-1] = (xs_ys[-1][0], interaction.final_grade.score)
    xs, ys = zip(*xs_ys)
    xs = np.cumsum(xs)
    # drop any ys that are still None
    if any(y is None for y in ys):
        raise ValueError("Some intermediate grades were missing")
    return np.sum(np.diff(ys) >= 0) / (len(ys) - 1)
