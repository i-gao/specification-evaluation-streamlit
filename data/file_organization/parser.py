from typing import List
from utils.misc import parse_for_answer_tags, parse_json


def parse_policy(yhat: str, raise_errors: bool = False):
    """Parse the policy JSON from the message."""
    policy_str = parse_for_answer_tags(
        yhat, keyword="policy", return_none_if_not_found=True
    )
    if policy_str is None:
        if raise_errors:
            raise ValueError(
                "Could not parse the policy. Wrap the JSON policy in <policy></policy> tags."
            )
        return None

    return parse_json(policy_str)


def parse_file_organization_solutions(msg: str) -> List[str]:
    """
    Extract solutions from a message for the file organization dataset.
    Treat <policy>...</policy> as a solution.
    """
    policy = parse_for_answer_tags(msg, keyword="policy", return_none_if_not_found=True)
    if policy:
        return [f"<policy>{policy}</policy>"]
    return []

