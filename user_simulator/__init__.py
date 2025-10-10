from utils.misc import import_from_string
from user_simulator.user import UserSimulator, UserConversationTurn, UserAction, BudgetExceeded
from user_simulator.dummy import DummyUser


def get_simulator(simulator_name: str, **kwargs):

    simulator_modules = {
        "full_specification": "user_simulator.full_specification.FullSpecificationUser",
        "ystar": "user_simulator.ystar.YStarUser",
        "dummy": "user_simulator.dummy.DummyUser",
    }

    if simulator_name not in simulator_modules:
        raise ValueError(f"Unknown simulator: {simulator_name}")

    simulator_class = import_from_string(simulator_modules[simulator_name])
    return simulator_class(**kwargs)
