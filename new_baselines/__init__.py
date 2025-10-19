from utils.misc import import_from_string
from new_baselines.policy import PolicyAction, PolicyConversationTurn, InteractionPolicy
from new_baselines.single_llm import RawLLM, ClarifyLLM, ExecutionLLM
from new_baselines.brainstorm_llm import BreakItDownLLM


POLICIES = [
    "raw_llm",
    "clarify_llm",
    "execution_llm",
    "break_it_down_llm",
]


def get_policy(policy_name: str, **kwargs):
    policy_modules = {
        "raw_llm": "new_baselines.single_llm.RawLLM",
        "clarify_llm": "new_baselines.single_llm.ClarifyLLM",
        "execution_llm": "new_baselines.single_llm.ExecutionLLM",
        "prompted_llm": "new_baselines.single_llm.PromptedLLM",
        "break_it_down_llm": "new_baselines.brainstorm_llm.BreakItDownLLM",
    }

    if policy_name not in policy_modules:
        raise ValueError(f"Unknown policy: {policy_name}")

    policy_class = import_from_string(policy_modules[policy_name])
    return policy_class(**kwargs)
