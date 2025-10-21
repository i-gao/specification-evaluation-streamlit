from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Literal,
    Union,
    Callable,
    Any,
    TypedDict,
)
from collections import defaultdict
from dataclasses import dataclass, asdict
from data.actions import Action
from data.dataset import Specification
import os
from langchain_core.messages import BaseMessage
import json

@dataclass
class PolicyConversationTurn:
    """Represents a turn in the conversation with assistant and user messages"""

    assistant_msg: str  # Message from the assistant
    user_msg: Optional[str] = None  # Message from the user
    user_cost: Optional[float] = None  # Cost of the user's response
    assistant_cost: Optional[float] = None  # Cost of the assistant's response


class ToolCall(TypedDict):
    name: str
    kwargs: str
    response: str
    status: Literal["success", "error"]


@dataclass
class PolicyAction:
    """Represents a React thought from the assistant"""

    content: str  # Content of the thought
    tool_calls: List[ToolCall]  # Associated actions
    token_cost: float  # Cost of the thought in tokens
    runtime_cost: float  # Cost of the thought in seconds
    status: Literal["success", "error"]  # Status of the thought
    goal: str  # Goal of the thought
    prompt: Dict[str, str]  # Top-level user/system prompts used to generate the thought


DEFAULT_HOOKS = [
    "get_state",
]


class InteractionPolicy:
    """
    An abstract class for an assistant

    The policy has access to:
    - interaction budget (C): the maximum cost of the conversation
    - actions: the public set of task actions the policy can take
    - initial_specification: the task signature
    - prediction_fmt_instructions: instructions for formatting the output y
    - msg_fmt_instructions: instructions for formatting the message to the user
    - verbosity: whether to print verbose output
    """

    def __init__(
        self,
        interaction_budget: float,
        actions: List[Action] = None,
        initial_specification: str = None,
        prediction_fmt_instructions: str = None,
        msg_fmt_instructions: str = None,
        hooks: List[Union[str, Callable]] = DEFAULT_HOOKS,
        verbosity: Literal[0, 1, 2] = 0,
        checkpoint_file: Optional[str] = None,
        spec: Optional[Specification] = None,
        cost_type: Literal["user", "policy", "both"] = "user",
    ):
        """
        Initialize the interaction policy.

        Args:
            interaction_budget: interaction cost budget
            actions: Public set of task actions the policy can take
            hooks: List of hooks to run immediately after receiving a user message
            verbosity: Whether to print verbose output. 0: print nothing, 1: print function outputs but not prompts, 2: print everything
            checkpoint_file: File to save checkpoints. If None, checkpointing is disabled.
            specification: used for hooks and checkpointing only, not generation or prediction. If provided, the policy's save_checkpoint method will automatically call the specification's get_state method
        """
        self.interaction_budget = interaction_budget
        self.actions = actions
        self.verbosity = verbosity
        self._hooks = hooks
        self.checkpoint_file = checkpoint_file
        self.spec = spec
        self.initial_specification = initial_specification
        self.prediction_fmt_instructions = prediction_fmt_instructions
        self.msg_fmt_instructions = msg_fmt_instructions
        self.cost_type = cost_type
        if self.checkpoint_file is not None and self.spec is None:
            print(
                "Warning: checkpoint_file is set but spec is not. This may cause inaccurate checkpoints when the spec itself is stateful."
            )

        # State tracking
        self.has_seen_system_prompt: bool = False
        self.conversation_history: List[PolicyConversationTurn] = []
        self.wants_to_end_conversation: bool = False
        self.action_history: Dict[int, List[PolicyAction]] = defaultdict(list)
        self.hook_history: Dict[int, Dict[str, Any]] = defaultdict(dict)
        self._model_lock = False

    ######## PROPERTIES ##########

    @property
    def current_unanswered_msg(self) -> str:
        """
        Checks if the assistant has a message that has not been answered by the user.
        If so, return the message.
        Otherwise, return None.
        """
        if len(self.conversation_history) == 0:
            return None

        last_turn = self.conversation_history[-1]
        if last_turn.user_msg is None:
            return last_turn.assistant_msg
        else:
            return None

    @property
    def total_cost(self) -> float:
        """
        Total cost of the conversation so far.
        Note: this may not match the user's total cost if the user runs out of budget while trying to answer. If that happened on turn t, this function will return c_{t-1}

        Returns:
            float: The total cost of the conversation
        """
        return sum(
            [
                (
                    turn.user_cost
                    if self.cost_type == "user"
                    else (
                        turn.assistant_cost
                        if self.cost_type == "policy"
                        else turn.user_cost + turn.assistant_cost
                    )
                )
                for turn in self.conversation_history
                if (
                    turn.user_cost is not None
                    if self.cost_type == "user"
                    else (
                        turn.assistant_cost is not None
                        if self.cost_type == "policy"
                        else turn.user_cost is not None
                        and turn.assistant_cost is not None
                    )
                )
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
        Index of the current turn in the conversation.

        Returns:
            int: The index of the current turn
        """
        if self.current_unanswered_msg is None:
            return len(self.conversation_history)
        else:
            return len(self.conversation_history) - 1

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

    def reset(self) -> None:
        """
        Reset the policy to its initial state.
        """
        self.conversation_history = []
        self.action_history = defaultdict(list)
        self.wants_to_end_conversation = False
        self.hook_history = defaultdict(dict)
        self.agent_executor.clear_state()
        self.has_seen_system_prompt = False

    def __call__(
        self, user_response: Optional[str] = None, user_cost: Optional[float] = None
    ) -> str:
        """
        Generate the next message in the conversation.

        Args:
            user_response: The user's response to the assistant's message
                None if this is the first turn
            user_cost: The cost of the user's response
                None if this is the first turn

        Returns:
            str: The next message in the conversation

        Raises:
            AssertionError: If user_response and user_cost are not provided on all turns except the first
        """
        if self._model_lock:
            print("Model is locked")
            return
        
        if self.current_unanswered_msg is not None:
            assert user_response is not None and user_cost is not None, (
                f"User response and cost must be provided on all turns except the first. The assistant's last message was '{self.current_unanswered_msg}'"
            )

        if self.turn_count == 0:
            self.run_hooks()

        if user_response is not None:
            if self.current_unanswered_msg is None:
                # User goes first and this is the first turn
                self.conversation_history.append(
                    PolicyConversationTurn(
                        assistant_msg=None,
                        user_msg=user_response,
                        user_cost=user_cost,
                    )
                )
            else:
                # Otherwise
                self.conversation_history[-1].user_msg = user_response
                self.conversation_history[-1].user_cost = user_cost

        assistant_msg, wants_to_end_conversation = self.generate_message(user_response)
        assistant_cost = sum(
            [
                a.runtime_cost
                for a in self.action_history[self.turn_count]
                if a.runtime_cost is not None
            ]
        )
        self.conversation_history.append(
            PolicyConversationTurn(
                assistant_msg=assistant_msg,
                assistant_cost=assistant_cost,
            )
        )
        self.wants_to_end_conversation = wants_to_end_conversation

        self.run_hooks()

        return assistant_msg

    def run_hooks(self) -> None:
        """
        Run the hooks.
        """
        # hook_history[idx] refers to the state before the user response to idx - 1, and before assistant message idx
        for hook in self._hooks:
            if self.verbosity == 2:
                print(f"Running hook {hook}")
            out = self._run_hook(hook)
            if not isinstance(out, dict):
                out = {hook: out}
            self.hook_history[self.turn_count].update(out)

    def generate_message(self, user_response: Optional[str] = None) -> Tuple[str, bool]:
        """
        Generate the next message in the conversation.
        Returns:
            Tuple[str, bool]:
                - str: The next message in the conversation
                - bool: Whether the assistant wants to end the conversation

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_test_prediction(self) -> str:
        """
        Get the current prediction from the strong model.
        """
        raise NotImplementedError("Subclasses must implement this method")

    ######## CHECKPOINTING ##########

    def save_checkpoint(self, connection=None) -> None:
        """
        Save the current state of the policy to a checkpoint file.
        Critically, this does not save configs for the policy, so the policy later must be initialized with the same configs as the checkpoint.

        Args:
            filename: Name of the checkpoint file (will be saved in checkpoint_dir)
        """
        if self.checkpoint_file is None:
            raise ValueError(
                "Checkpoint file not set. Set checkpoint_file in __init__ to enable checkpointing."
            )
        os.makedirs(os.path.dirname(self.checkpoint_file) or ".", exist_ok=True)

        # To load by turn, rely on the hook history
        # To load the most recent state, rely on the other keys
        checkpoint_data = {
            "hook_history": self.hook_history,
            "wants_to_end_conversation": self.wants_to_end_conversation,
            "conversation_history": self.get_conversation_history(),
            "action_history": self.get_action_history(),
            "agent_executor_state": self.agent_executor.get_state(),
            "turn_idx": self.turn_count,
        }
        if self.spec is not None:
            checkpoint_data["specification_state"] = self.spec.get_state()

        # Save checkpoint
        if connection is not None:
            connection.write(self.checkpoint_file, json.dumps(checkpoint_data, indent=2))
        else:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

        if self.verbosity >= 1:
            print(f"Checkpoint saved to {self.checkpoint_file}")

    def load_checkpoint(self, turn_idx: int = None, connection=None) -> None:
        """
        Load the policy state from a checkpoint file.
        Critically, this does not save configs for the policy, so the policy must be initialized with the same configs as the checkpoint.

        Relies on the hook history to load the state by turn.

        Args:
            filename: Name of the checkpoint file (will be loaded from checkpoint_dir)
            turn_idx: If provided, only load the state up to this turn. If None, load the entire checkpoint.
        """
        if self.checkpoint_file is None:
            raise ValueError(
                "Checkpoint file not set. Set checkpoint_file in __init__ to enable checkpointing."
            )
        if not os.path.exists(self.checkpoint_file):
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_file}"
            )
        
        if connection is not None:
            # Use binary read method if available, otherwise fall back to base64 decoding
            checkpoint_data = json.loads(connection.read(self.checkpoint_file))
        else:
            with open(self.checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)

        if turn_idx is not None:
            # Restore a single turn based on the hook history
            # Remember this captures the state BEFORE the user and the assistant msg, i.e. 0 = blank slate
            assert turn_idx in checkpoint_data["hook_history"], (
                f"No hook history available for turn {turn_idx}"
            )
            self._restore_full_state(
                conversation_history=checkpoint_data["hook_history"][turn_idx][
                    "conversation_history"
                ],
                action_history=checkpoint_data["hook_history"][turn_idx][
                    "action_history"
                ],
                hook_history={
                    k: v
                    for k, v in checkpoint_data["hook_history"].items()
                    if k <= turn_idx
                },
                agent_executor_state=checkpoint_data["hook_history"][turn_idx][
                    "agent_executor_state"
                ],
                specification_state=checkpoint_data["hook_history"][turn_idx][
                    "specification_state"
                ],
                wants_to_end_conversation=checkpoint_data["hook_history"][turn_idx][
                    "wants_to_end_conversation"
                ],
            )
        else:
            # Use the last available state
            self._restore_full_state(
                conversation_history=checkpoint_data["conversation_history"],
                action_history=checkpoint_data["action_history"],
                hook_history=checkpoint_data["hook_history"],
                agent_executor_state=checkpoint_data["agent_executor_state"],
                specification_state=checkpoint_data["specification_state"],
                wants_to_end_conversation=checkpoint_data["wants_to_end_conversation"],
            )
            turn_idx = checkpoint_data["turn_idx"]

        if self.verbosity >= 1:
            print(
                f"Checkpoint loaded from {self.checkpoint_file}, currently at the start of turn {turn_idx} (awaiting user response)"
            )

    def _restore_full_state(
        self,
        *,
        conversation_history: List[dict],
        action_history: Dict[int, List[dict]],
        hook_history: Dict[int, Dict[str, Any]],
        agent_executor_state: str,
        specification_state: Dict[str, Any],
        wants_to_end_conversation: bool,
    ) -> None:
        """
        Restore the state of the policy based on the checkpoint data.
        """
        # Otherwise, restore the entire state
        # Restore conversation history - convert dictionaries back to dataclasses
        self.conversation_history = [
            PolicyConversationTurn(**turn_dict) for turn_dict in conversation_history
        ]

        # Restore action history - convert dictionaries back to dataclasses
        self.action_history = defaultdict(list)
        for turn_num, actions_list in action_history.items():
            for action_dict in actions_list:
                # The action_dict is already a dictionary from get_action_history()
                # We can store it as-is since it's already serializable
                self.action_history[int(turn_num)].append(PolicyAction(**action_dict))

        # Restore hook history
        self.hook_history = hook_history

        # Restore agent executor state
        self.agent_executor.load_state(agent_executor_state)

        # Restore spec state
        if self.spec is not None:
            self.spec.load_state(specification_state)

        # Restore other state
        self.wants_to_end_conversation = wants_to_end_conversation

    ######## HELPER FUNCTIONS ##########

    def _fmt_tools(self) -> str:
        """
        Format the tools available as a string.

        Returns:
            str: A formatted string of the tools available
        """
        lines = []
        for action in self.actions:
            lines.append(f"- {action.name}: {action.description}")
        return "\n".join(lines)

    def _parse_langchain_response_to_actions(
        self, raw: List[BaseMessage]
    ) -> List[PolicyAction]:
        """
        Parse the raw response from a LangChain model into a list of PolicyAction objects.
        """
        from langchain_core.messages import AIMessage, ToolMessage
        from utils.model import get_token_usage

        # Look at the new messages and extract the tool calls for saving
        action_history = []
        for i, msg in enumerate(raw):
            if not isinstance(msg, AIMessage):
                continue

            token_cost = get_token_usage(msg.response_metadata, default_value=0)

            # collect all associated tool call info
            tool_call_kwargs, tool_call_names, tool_call_responses, statuses = (
                [],
                [],
                [],
                [],
            )
            for tool_call in getattr(msg, "additional_kwargs", {}).get(
                "tool_calls", []
            ):
                id = tool_call["id"]
                kwargs = tool_call["function"].get("arguments")
                if kwargs is None:
                    kwargs = tool_call["function"].get("args")
                name = tool_call["function"]["name"]

                # look ahead for response
                tool_response = None
                status = "error"
                for j in range(i + 1, len(raw)):
                    if isinstance(raw[j], ToolMessage) and raw[j].tool_call_id == id:
                        tool_response = raw[j].content
                        status = getattr(raw[j], "status", "success")
                        break

                tool_call_kwargs.append(kwargs)
                tool_call_names.append(name)
                tool_call_responses.append(tool_response)
                statuses.append(status)

            tool_calls = [
                {
                    "name": name,
                    "kwargs": kwargs,
                    "response": response,
                    "status": status,
                }
                for kwargs, name, response, status in zip(
                    tool_call_kwargs, tool_call_names, tool_call_responses, statuses
                )
            ]

            # append result to action history
            action_history.append(
                PolicyAction(
                    content=msg.content,
                    goal=None,
                    prompt=None,
                    token_cost=token_cost,
                    runtime_cost=None,
                    tool_calls=tool_calls,
                    status=(
                        "success"
                        if any(status == "success" for status in statuses)
                        or len(tool_calls) == 0
                        else "error"
                    ),
                )
            )

        return action_history

    def _run_hook(self, hook: Union[str, Callable]) -> None:
        """
        Run a hook.
        """
        hook_state = {
            "conversation_history": self.conversation_history,
            "action_history": self.action_history,
            "turn_count": self.turn_count,
            "specification": self.spec,
        }
        if isinstance(hook, Callable):
            return hook(hook_state)

        if hook == "get_state":
            return {
                "conversation_history": self.get_conversation_history(),
                "action_history": self.get_action_history(),
                "agent_executor_state": self.agent_executor.get_state(),
                "specification_state": self.spec.get_state(),
                "wants_to_end_conversation": self.wants_to_end_conversation,
            }
        elif hook == "get_test_prediction":
            return {"test_prediction": self._get_test_prediction()}
        elif hook == "save_checkpoint":
            self.save_checkpoint()
            return {}
        else:
            raise ValueError(f"Hook {hook} not recognized")
