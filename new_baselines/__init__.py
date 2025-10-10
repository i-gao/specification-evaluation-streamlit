from utils.misc import import_from_string
from new_baselines.policy import PolicyAction, PolicyConversationTurn, InteractionPolicy
from new_baselines.prompted_llm import RawLLM, ClarifyLLM, ExecutionLLM, PromptedLLM


POLICIES = [
    "raw_llm",
    "clarify_llm",
    "execution_llm",
]


def get_policy(policy_name: str, **kwargs):
    policy_modules = {
        "raw_llm": "new_baselines.prompted_llm.RawLLM",
        "clarify_llm": "new_baselines.prompted_llm.ClarifyLLM",
        "execution_llm": "new_baselines.prompted_llm.ExecutionLLM",
        "prompted_llm": "new_baselines.prompted_llm.PromptedLLM",
    }

    if policy_name not in policy_modules:
        raise ValueError(f"Unknown policy: {policy_name}")

    policy_class = import_from_string(policy_modules[policy_name])
    return policy_class(**kwargs)
