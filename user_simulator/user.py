from typing import List, Tuple, Dict, Optional, Callable, Literal, Union, Any, TypedDict
from collections import defaultdict
import string
import json
import sys



from data.dataset import Specification
from data.actions import Action
from utils.misc import (
    add_section,
    parse_for_answer_tags,
    Stopwatch,
    parse_list,
    fuzzy_match,
    print_debug,
)
from dataclasses import dataclass, asdict


class BudgetExceeded(Exception):
    """Exception raised when the interaction budget is exceeded."""

    def __init__(self, message: str = "Interaction budget exceeded"):
        self.message = message
        super().__init__(self.message)


@dataclass
class UserConversationTurn:
    """Represents a turn in the conversation with assistant and user messages"""

    assistant_msg: str  # Message from the assistant
    user_msg: str  # Message from the user
    token_cost: float  # Cost of the user's response in tokens
    runtime_cost: float  # Cost of the user's response in seconds
    remaining_budget: float  # Remaining budget after this turn
    user_rationale: Optional[str] = (
        None  # User's internal thinking process before generating the response
    )


class ToolCall(TypedDict):
    name: str
    kwargs: str
    response: str
    status: Literal["success", "error"]


@dataclass
class UserAction:
    """Represents an action in the conversation with assistant and user messages"""

    content: str  # Content of the thought
    tool_calls: List[ToolCall]  # Associated actions
    token_cost: float  # Cost of the thought in tokens
    runtime_cost: float  # Cost of the thought in seconds
    status: Literal["success", "error"]  # Status of the thought
    goal: str  # Goal of the thought
    prompt: Dict[str, str]  # Top-level user/system prompts used to generate the thought


class UserSimulator:
    """
    An abstract class for an agent which simulates the behavior of \mathcal U, the user.

    The user has access to:
    - interaction budget (C): the maximum cost of the conversation
    - actions: the set of task actions the user can take (public and private)
    - initial_specification: the task signature
    - prediction_fmt_instructions: instructions for formatting the output y
        - verbosity: whether to print verbose output
    """

    def __init__(
        self,
        spec: Specification,
        interaction_budget: float,
        verbosity: Literal[0, 1, 2] = 0,
        lambda_cost: float = 0.0,
        **kwargs,
    ):
        """
        Initialize the user simulator.

        Args:
            spec: The specification containing reward / validation function
            interaction_budget: interaction cost budget
            verbosity: Whether to print verbose output. 0: print nothing, 1: print function outputs but not prompts, 2: print everything
            lambda_cost: weight of cost in the reward function
            max_retries: maximum number of retries for each tool call
        """
        # Task information
        self.actions = spec.actions
        self.public_action_names = [action.name for action in spec.public_tools]
        self.validity_fn = getattr(spec, "validity_fn", None)
        self.reward_fn = getattr(spec, "reward_fn", None)
        self.fmt_instructions = spec.prediction_fmt_instructions
        self.lambda_cost = lambda_cost
        self.interaction_budget = interaction_budget
        self.verbosity = verbosity

        # State tracking
        self.conversation_history: List[UserConversationTurn] = []
        self.action_history: Dict[int, List[UserAction]] = defaultdict(list)

    ######## PROPERTIES ##########

    @property
    def total_cost(self) -> float:
        """
        Total cost of the conversation so far (sum of runtime_costs).

        Returns:
            float: The total cost of the conversation
        """
        return sum(
            [
                turn.runtime_cost
                for turn in self.conversation_history
                if turn.runtime_cost is not None
            ]
        )

    @property
    def remaining_budget(self) -> float:
        """
        Remaining budget for the conversation.

        Returns:
            float: The remaining budget
        """
        return self.interaction_budget - self.total_cost

    @property
    def turn_count(self) -> int:
        """
        Number of turns in the conversation.

        Returns:
            int: The number of turns
        """
        return len(self.conversation_history)

    def get_conversation_history(self) -> List[dict]:
        """
        Get the conversation history as a list of dicts.

        Returns:
            List[dict]: List of conversation turns as dictionaries
        """
        return [asdict(turn) for turn in self.conversation_history]

    def get_action_history(self) -> Dict[int, List[Dict]]:
        """
        Get the action history.

        Returns:
            Dict[int, List[Dict]]: Dictionary mapping turn numbers to lists of action dictionaries
        """
        return {
            k: [asdict(action) for action in actions]
            for k, actions in self.action_history.items()
        }

    ######## MAIN METHODS ##########

    def initial_message(self) -> Tuple[str, float]:
        """
        Generate the initial user message and its cost, to start the conversation.
        Returns:
            Tuple[str, float]: The user's initial message and its cost in seconds
        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError(
            "Subclasses must implement initial_message() for user-first conversations."
        )

    def grade(self, yhat: str) -> Tuple[bool, float, dict]:
        """
        Given a single yhat, grades it using the reward function.
        """
        try:
            s, d = self.score(yhat=yhat)
            b = not (s == float("-inf"))
        except Exception as e:
            b, d = self.validate(yhat=yhat)
            s = None
        return b, s, d

    def validate(self, yhat: str) -> Tuple[bool, dict]:
        """
        Given a single yhat, validates it using the validity function.

        Args:
            yhat: The response to validate

        Returns:
            Tuple[bool, dict]: A tuple containing:
                - bool: Whether the response is valid
                - dict: Validation metadata
        """
        assert self.validity_fn is not None, "Validity function is not set"
        b, d = self.validity_fn(yhat=yhat, raise_errors=False)
        return b, d

    def score(self, yhat: str) -> Tuple[float, dict]:
        """
        Given a single yhat, scores it using the reward function.

        Args:
            yhat: The response to score

        Returns:
            Tuple[float, dict]: A tuple containing:
                - float: Score (-inf if invalid)
                - dict: Evaluation metadata
        """
        assert self.reward_fn is not None, "Reward function is not set"

        s, d = self.reward_fn(yhat=yhat, raise_errors=False)

        cost_term = self.lambda_cost * self.total_cost
        return s - cost_term, d

    def reset(self) -> None:
        """
        Reset the user simulator to its initial state.
        """
        self.conversation_history = []
        self.action_history = defaultdict(list)

    def __call__(self, assistant_msg: str) -> Tuple[str, float]:
        """
        Process a message from the assistant and generate a response.

        Args:
            assistant_msg: The message from the assistant

        Returns:
            Tuple[str, float]: A tuple containing:
                - str: The user's response
                - float: The cost of the user's response in seconds

        Raises:
            BudgetExceeded: If the interaction budget is exceeded
        """
        if self.remaining_budget <= 0:
            raise BudgetExceeded()

        response, token_cost, runtime_cost = self._respond(assistant_msg)

        # build the rationale
        rationale = ""
        for action in self.action_history[self.turn_count]:
            rationale += f"Thought: {action.content}\n"
            rationale += f"Actions: {action.actions}\n\n"

        if self.verbosity:
            print_debug(
                f"Runtime cost: simulator took {runtime_cost}s to respond",
                "__call__",
                color="green",
            )

        self.conversation_history.append(
            UserConversationTurn(
                assistant_msg=assistant_msg,
                user_msg=response,
                token_cost=token_cost,
                runtime_cost=runtime_cost,
                user_rationale=rationale,
                remaining_budget=self.remaining_budget - runtime_cost,
            )
        )  # This will be appended even if the user runs out of budget

        if self.remaining_budget <= 0:
            raise BudgetExceeded()

        return response, runtime_cost

    def _respond(self, assistant_msg: str) -> Tuple[str, float, float]:
        """
        Process a message from the assistant and generate a response.

        Args:
            assistant_msg: The message from the assistant

        Returns:
            Tuple[str, float, float]: A tuple containing:
                - str: The user's response
                - float: The token cost of the response
                - float: The runtime cost of the response

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement this method")

    ######## HELPER METHODS ##########

    def _fmt_conversation_history(
        self, assistant_msg: Optional[str] = None, max_turns: int = 10
    ) -> str:
        """
        Format the conversation history as a string, including
        optionally the next assistant message.

        Args:
            assistant_msg: Optional next message from the assistant to include
            max_turns: Maximum number of turns to include.
                Will show max_turns -2 of initial turns, plus the most recent two turns.
        Returns:
            str: A formatted string of the conversation history
        """
        if len(self.conversation_history) == 0 and assistant_msg is None:
            return "No previous conversation history."

        lines = []
        n = len(self.conversation_history)
        if n <= max_turns:
            first_part = self.conversation_history
            last_part = []
        else:
            # Avoid overlap/duplication
            first_part = self.conversation_history[: max_turns - 2]
            last_part = self.conversation_history[-2:]

        # Insert omission line after the first part
        for turn in first_part:
            if turn.assistant_msg is not None:
                lines.append(
                    add_section("[HUMAN MESSAGE]", turn.assistant_msg, style="divider")
                )
            if turn.user_msg is not None:
                lines.append(
                    add_section("[YOUR RESPONSE]", turn.user_msg, style="divider")
                )
        if len(last_part) > 0:
            lines.append("... middle of conversation omitted for brevity ...")
            for turn in last_part:
                if turn.assistant_msg is not None:
                    lines.append(
                        add_section(
                            "[HUMAN MESSAGE]", turn.assistant_msg, style="divider"
                        )
                    )
                if turn.user_msg is not None:
                    lines.append(
                        add_section("[YOUR RESPONSE]", turn.user_msg, style="divider")
                    )

        if assistant_msg is not None:
            lines.append(
                add_section(f"[(NEW!) HUMAN MESSAGE]", assistant_msg, style="divider")
            )
        return "\n\n".join(lines)

    def get_simulator_system_message(self, assistant_msg: str) -> str:
        """
        System message for the simulator LLM.

        Args:
            assistant_msg: The most recent message from the assistant (not in the conversation history)

        Returns:
            str: The system message for the simulator

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement this method")
