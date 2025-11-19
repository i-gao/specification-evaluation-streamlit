from utils.misc import import_from_string
from user_simulator.user import (
    UserSimulator,
    UserConversationTurn,
    UserAction,
    BudgetExceeded,
)
from user_simulator.dummy import DummyUser


def get_simulator(simulator_name: str, **kwargs):
    simulator_modules = {
        "dummy": "user_simulator.dummy.DummyUser",
        "full_specification": "user_simulator.fixed_knowledge.FullSpecificationUser",
        "open_full_specification": "user_simulator.fixed_knowledge.OpenFullSpecificationUser",
        "ystar": "user_simulator.fixed_knowledge.YStarUser",
        "open_ystar": "user_simulator.fixed_knowledge.OpenYStarUser",
        "feature_discovery": "user_simulator.feature_discovery_user.FeatureDiscoveryUser",
    }

    if simulator_name not in simulator_modules:
        raise ValueError(f"Unknown simulator: {simulator_name}")

    simulator_class = import_from_string(simulator_modules[simulator_name])
    return simulator_class(**kwargs)
