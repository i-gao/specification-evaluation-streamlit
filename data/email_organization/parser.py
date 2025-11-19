from typing import List
import json
from utils.misc import parse_for_answer_tags


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

    try:
        policy = json.loads(policy_str)
    except json.JSONDecodeError:
        policy = None
    if policy is None:
        if raise_errors:
            raise ValueError(
                "Could not parse the policy. Wrap the JSON policy in <policy></policy> tags."
            )
        return None
    return policy


def parse_email_organization_solutions(msg: str) -> List[str]:
    """
    Extract solutions from a message for the email organization dataset.
    Treat <policy>...</policy> as a solution.
    """
    policy = parse_for_answer_tags(msg, keyword="policy", return_none_if_not_found=True)
    if policy:
        return [f"<policy>{policy}</policy>"]
    return []

