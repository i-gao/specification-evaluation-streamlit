from utils.misc import import_from_string
from new_baselines.policy import PolicyAction, PolicyConversationTurn, InteractionPolicy
from new_baselines.single_llm import RawLLM, ClarifyLLM, ExecutionLLM
from new_baselines.brainstorm_llm import BreakItDownLLM
from new_baselines.exploration_llm import ExplorationLLM
from new_baselines.voi_llm import VOILLM


POLICIES = [
    "raw_llm",
    "clarify_llm",
    "execution_llm",
    "clarify_then_execute_llm",
    "break_it_down_llm",
    "tom_llm",
    "clarify_feature_llm",
    "exploration_llm",
    "lightweight_exploration_llm",
    "super_lightweight_exploration_llm",
    "epsilon_greedy_exploration_llm",
    "voi_llm",
    "nimble_llm",
    "adversarial_execution_llm",
]


def get_policy(policy_name: str, **kwargs):
    policy_modules = {
        "raw_llm": "new_baselines.single_llm.RawLLM",
        "continual_clarify_llm": "new_baselines.single_llm.ClarifyLLM",
        "execution_llm": "new_baselines.single_llm.ExecutionLLM",
        "clarify_llm": "new_baselines.single_llm.ClarifyThenExecuteLLM",
        "alternate_clarify_llm": "new_baselines.single_llm.AlternateClarifyThenExecuteLLM",
        "prompted_llm": "new_baselines.single_llm.PromptedLLM",
        "break_it_down_llm": "new_baselines.brainstorm_llm.BreakItDownLLM",
        "tom_llm": "new_baselines.tom_llm.TomLLM",
        "clarify_feature_llm": "new_baselines.exploration_llm.ClarifyWithFeatureTrackingLLM",
        "exploration_llm": "new_baselines.exploration_llm.ExplorationLLM",
        "lightweight_exploration_llm": "new_baselines.lightweight_exploration_llm.LightweightExplorationLLM",
        "super_lightweight_exploration_llm": "new_baselines.super_lightweight_exploration_llm.SuperLightweightExplorationLLM",
        "epsilon_greedy_exploration_llm": "new_baselines.epsilon_greedy_exploration_llm.EpsilonGreedyExplorationLLM",
        "voi_llm": "new_baselines.voi_llm.VOILLM",
        "nimble_llm": "new_baselines.single_llm.NimbleLLM",
        "adversarial_execution_llm": "new_baselines.single_llm.AdversarialExecutionLLM",
    }

    if policy_name not in policy_modules:
        raise ValueError(f"Unknown policy: {policy_name}")

    policy_class = import_from_string(policy_modules[policy_name])
    return policy_class(**kwargs)
