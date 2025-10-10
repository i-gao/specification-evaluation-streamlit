from user_simulator.user import UserSimulator, UserConversationTurn, UserAction
from typing import List, Dict
from collections import defaultdict


class DummyUser(UserSimulator):
    """
    Dummy class which is used when the user is a real human
    (e.g. during user studies).

    Since the class inherits from UserSimulator, it can be used for grading and action retrieval.
    However, it does not have a __call__ method, so it cannot be used for interaction.

    At the end of the interaction, we infill the conversation history and action history by calling the infill_history method.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().__init__(**kwargs)

    def infill_history(
        self,
        messages: List[Dict[str, str]],
        tool_history: Dict[int, List[Dict]],
    ):
        """
        Args:
            messages: List of dictionaries of the form
            {'role': 'assistant' | 'user', 'content': str, 'sent_time': float}
            Expects the messages to be strictly alternating between assistant and user messages.

            tool_history: Dict of the form {i: [tool1, tool2, ...]}
            where tool1, tool2, ... are dictionaries of the form
            {
                "name": str,
                "kwargs": dict,
                "response": str,
                "status": str,
            } and i is the turn number.
        """
        self.conversation_history = []
        remaining_budget = self.interaction_budget

        for i in range(len(messages)):
            m = messages[i]
            if m["role"] == "user":
                continue
            if i == len(messages) - 1:
                # Last message is from the assistant
                user_response = None
                user_cost = None
            else:
                user_response = messages[i + 1]["content"]
                user_cost = messages[i + 1]["sent_time"] - m["sent_time"]
                remaining_budget -= user_cost

            self.conversation_history.append(
                UserConversationTurn(
                    assistant_msg=m["content"],
                    user_msg=user_response,
                    token_cost=None,
                    runtime_cost=user_cost,
                    remaining_budget=remaining_budget,
                    user_rationale=None,
                )
            )

        self.action_history = defaultdict(list)
        for i, tool_calls in tool_history.items():
            statuses = [tool["status"] for tool in tool_calls]
            self.action_history[i] = [
                UserAction(
                    content=None,
                    goal="respond",
                    prompt=None,
                    token_cost=None,
                    runtime_cost=None,
                    tool_calls=tool_calls,
                    status=(
                        "success"
                        if any(status == "success" for status in statuses)
                        or len(tool_calls) == 0
                        else "error"
                    ),
                )
            ]
