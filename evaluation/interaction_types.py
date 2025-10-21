from dataclasses import dataclass, field, asdict, is_dataclass, fields
from typing import List, Dict, Any, Optional, Literal, Union
import json
import os
from user_simulator.user import UserSimulator
from new_baselines.policy import InteractionPolicy
from utils.misc import _clean_for_json

from evaluation.save_hooks import DEFAULT_SAVE_HOOKS, run_hook
from data.dataset import Specification

END_REASONS = Literal["budget_exhausted", "policy_end", "user_end", "unknown"]


def _check_config(config: dict):
    """
    Check that the config is valid.
    """
    required_keys = {
        "dataset",
        "dataset_kwargs",
        "spec_index",
        "policy",
        "policy_model",
        "policy_kwargs",
        "simulator",
        "simulator_model",
        "simulator_kwargs",
        "seed",
        "user_first",
        "interaction_budget",
        "include_initial_specification",
        "include_fmt_instructions",
    }
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config must include {key}")


@dataclass
class Grade:
    """Represents grading results for a set of predictions"""

    prediction: str
    score: float
    correct: bool
    eval_metadata: Dict[str, Any]
    prompt: dict = None


@dataclass
class Turn:
    """Represents a single turn in an interaction"""

    assistant_msg: str
    assistant_actions: List[Dict[str, Any]]
    user_msg: str
    user_actions: List[Dict[str, Any]]
    user_rationale: Optional[str] = None
    intermediate_grade: Optional[Grade] = None
    user_token_cost: Optional[float] = None
    user_runtime_cost: Optional[float] = None
    assistant_token_cost: Optional[float] = None
    assistant_runtime_cost: Optional[float] = None
    remaining_budget: Optional[float] = None


@dataclass
class ComparisonTurn(Turn):
    """Represents a single turn in an interaction with two different user responses for comparison"""

    user2_msg: str = None
    user2_actions: List[Dict[str, Any]] = None
    user2_rationale: Optional[str] = None
    user2_intermediate_grade: Optional[Grade] = None
    user2_token_cost: Optional[float] = None
    user2_runtime_cost: Optional[float] = None


@dataclass
class Interaction:
    """Represents a complete interaction between a simulator and policy"""

    turns: List[Turn]
    config: Dict[str, Any]
    remaining_budget: float
    final_turn: int
    final_grade: Grade
    end_reason: END_REASONS
    filename: str
    spec_information: Dict[str, Any] = field(default_factory=dict)
    hook_results: Dict[str, Any] = field(default_factory=dict)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


###############################################


def save_interaction(
    *,
    simulator: UserSimulator,
    policy: InteractionPolicy,
    output_path: str,
    config: dict,
    end_reason: END_REASONS,
    final_prediction: str = None,
    final_grade: Grade = None,
    grader: UserSimulator = None,
    save_hooks: List[str] = DEFAULT_SAVE_HOOKS,
    skip_grading: bool = False,
    spec: Specification = None,
    connection=None,
    **kwargs,
) -> Interaction:
    """
    Given a policy, simulator, predictions, and grader, save the results of an interaction evaluation to a file as an Interaction object.

    Args:
        simulator: The simulator instance
        policy: The policy instance
        output_path: Path to save results
        config: Configuration
        end_reason: The reason for the end of the interaction
        final_prediction: The final prediction from the policy
        grader: The grader instance (a simulator)
        **kwargs: Additional arguments
    Returns:
        An Interaction instance containing the results of the interaction evaluation
    """

    # Check that the config is valid
    _check_config(config)

    # Build information about the spec
    spec_information = {}
    if spec is not None:
        spec_information = {
            "initial_specification": spec.initial_specification,
            "prediction_fmt_instructions": spec.prediction_fmt_instructions,
            "msg_fmt_instructions": spec.msg_fmt_instructions,
            "dataset_name": spec.dataset_name,
            "index": spec.index,
            "dataset_kwargs": config.get("dataset_kwargs", {}),
        }
        action = next((a for a in spec.actions if a.fn.name == "ls_files"), None)
        if action is not None:
            spec_information["ls_output"] = str(action.fn.func())

    # Check that the grader is provided
    def _grade(prediction: str) -> Grade:
        assert grader is not None, "Grader must be provided"
        is_valid, score, eval_metadata = grader.grade(prediction)
        return Grade(
            prediction=prediction,
            score=score,
            correct=is_valid,
            eval_metadata=eval_metadata,
        )

    # No final prediction -> rerun prediction
    if final_prediction is None and not skip_grading:
        final_prediction = policy.get_test_prediction()

    # Run the grading function
    if final_grade is None and not skip_grading:
        final_grade = _grade(final_prediction)
        print(f"Grader's final score: {final_grade.score}")

    # Extract intermediate predictions from hook_history if not provided
    if policy is not None:
        intermediate_predictions = []
        for turn_num in sorted(policy.hook_history.keys()):
            if "get_test_prediction" in policy.hook_history[turn_num]:
                intermediate_predictions.append(
                    policy.hook_history[turn_num]["get_test_prediction"]
                )
    else:
        intermediate_predictions = None

    # Otherwise, start building the turns
    # The policy is the ground truth for the number of turns & interaction budget
    turns = []
    user_conversation_history = (
        simulator.get_conversation_history() if simulator is not None else []
    )
    user_action_history = (
        simulator.get_action_history() if simulator is not None else {}
    )
    policy_conversation_history = (
        policy.get_conversation_history() if policy is not None else []
    )
    policy_action_history = policy.get_action_history() if policy is not None else {}

    num_turns = max(
        len(user_conversation_history),
        len(policy_conversation_history),
        len(policy_action_history),
    )

    for i in range(num_turns):
        # Get the policy's turn
        assistant_msg = (
            policy_conversation_history[i]["assistant_msg"]
            if i < len(policy_conversation_history)
            else None
        )
        assistant_actions = policy_action_history.get(i, [])

        # Get the simulator's turn
        (
            user_msg,
            user_rationale,
            user_token_cost,
            user_runtime_cost,
            remaining_budget,
        ) = (
            (
                user_conversation_history[i]["user_msg"],
                user_conversation_history[i]["user_rationale"],
                user_conversation_history[i]["token_cost"],
                user_conversation_history[i]["runtime_cost"],
                user_conversation_history[i]["remaining_budget"],
            )
            if i < len(user_conversation_history)
            else (None, None, None, None, None)
        )
        user_actions = user_action_history.get(i, [])

        # Get the assistant costs form the PolicyConversationTurn
        if i < len(policy_conversation_history):
            assistant_runtime_cost = policy_conversation_history[i]["assistant_cost"]
        else:
            assistant_runtime_cost = None

        # Get the intermediate grade if it exists
        if (
            intermediate_predictions is not None
            and i < len(intermediate_predictions)
            and not skip_grading
        ):
            predictions = intermediate_predictions[i]
            intermediate_grade = _grade(predictions)
        else:
            predictions = None
            intermediate_grade = None

        # Build the turn object
        turn = Turn(
            assistant_msg=assistant_msg,
            assistant_actions=assistant_actions,
            user_msg=user_msg,
            user_actions=user_actions,
            user_rationale=user_rationale,
            user_token_cost=user_token_cost,
            user_runtime_cost=user_runtime_cost,
            assistant_runtime_cost=assistant_runtime_cost,
            remaining_budget=remaining_budget,
            intermediate_grade=intermediate_grade,
        )
        turns.append(turn)

    # Create the final interaction object
    interaction = Interaction(
        turns=turns,
        config=config,
        remaining_budget=(
            policy.remaining_budget if policy is not None else None
        ),  # BEFORE any end
        final_grade=final_grade,
        final_turn=(policy.turn_count if policy is not None else len(turns)),
        end_reason=end_reason,
        spec_information=spec_information,
        filename=os.path.basename(output_path),
    )

    # Save additional metadata
    kwargs["policy_checkpoint_file"] = (
        policy.checkpoint_file if policy is not None else None
    )

    # Save hooks
    for hook in save_hooks:
        try:
            interaction.hook_results[hook] = run_hook(hook, interaction)
        except Exception as e:
            print(f"Error running hook {hook}: {e}")
            pass

    out = {**asdict(interaction), **kwargs}
    out = _clean_for_json(out)

    # Save to file
    try:
        # Use connection if provided, otherwise fall back to direct file operations
        if connection is not None:
            connection.write(output_path, json.dumps(out, indent=2))
        else:
            with open(output_path, "w") as f:
                json.dump(out, f, indent=2)
        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"Error saving interaction to {output_path}: {e}")

    # Save policy checkpoint if checkpointing is enabled
    try:
        policy.save_checkpoint(connection=connection)
    except Exception as e:
        print(f"Error saving policy checkpoint: {e}")

    return interaction


def _convert_to_dataclass(data: Dict[str, Any], target_class: type) -> Any:
    """
    Recursively convert a dictionary to a dataclass instance.
    Uses dataclasses.is_dataclass() to detect and convert nested dataclasses.

    Args:
        data: Dictionary to convert
        target_class: The dataclass type to convert to
    Returns:
        An instance of the target dataclass with all nested dataclasses properly converted
    """
    if data is None:
        return None

    # Handle lists - recursively convert each item if it's a dict
    if isinstance(data, list):
        # For lists, we need to determine the type of items in the list
        if hasattr(target_class, "__origin__") and target_class.__origin__ is list:
            item_type = target_class.__args__[0]
            return [
                (
                    _convert_to_dataclass(item, item_type)
                    if isinstance(item, dict)
                    else item
                )
                for item in data
            ]
        return data

    # Handle dictionaries - recursively convert nested dicts to their appropriate dataclass types
    if isinstance(data, dict):
        # Get the field names and types from the dataclass
        field_types = target_class.__annotations__
        field_names = {field.name for field in fields(target_class)}

        # Filter out unknown keys
        filtered_data = {k: v for k, v in data.items() if k in field_names}

        # For Interaction dataclass, capture unknown keys in extra_metadata
        if target_class == Interaction:
            unknown_keys = {k: v for k, v in data.items() if k not in field_names}
            if unknown_keys:
                filtered_data["extra_metadata"] = unknown_keys

        # Convert each field according to its type
        for field_name, field_value in filtered_data.items():
            field_type = field_types[field_name]

            # Handle Optional types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                # Get the first non-None type from Optional[T]
                field_type = next(t for t in field_type.__args__ if t is not type(None))

            # Convert based on field type
            if isinstance(field_value, dict):
                if is_dataclass(field_type):
                    filtered_data[field_name] = _convert_to_dataclass(
                        field_value, field_type
                    )
            elif isinstance(field_value, list):
                if (
                    hasattr(field_type, "__origin__")
                    and field_type.__origin__ is list
                    and is_dataclass(field_type.__args__[0])
                ):
                    filtered_data[field_name] = [
                        (
                            _convert_to_dataclass(item, field_type.__args__[0])
                            if isinstance(item, dict)
                            else item
                        )
                        for item in field_value
                    ]

        # Create the dataclass instance
        return target_class(**filtered_data)

    return data


def load_interaction(
    path: str,
    connection = None,
) -> Interaction:
    """
    Load the results of an interaction evaluation from a file as an Interaction object.
    Properly reconstructs all dataclass objects from JSON data.
    """
    try:
        # use connection if provided, otherwise fall back to direct file operations
        if connection is not None:
            data = connection.read(path)
        else:
            with open(path, "r") as f:
                data = json.load(f)
        if "filename" not in data:
            data["filename"] = os.path.basename(path)
        return _convert_to_dataclass(data, Interaction)
    except Exception as e:
        print(f"Error loading interaction from {path}: {e}")
        raise e


def conversation_history_to_messages(
    conversation_history: List[Turn],
) -> List[Dict[str, Any]]:
    """
    Convert a list of Turn objects to a list of messages in
    {"role": "assistant" | "user", "content": str, "response_time": float}
    format.
    """
    messages = []
    for turn in conversation_history:
        if turn.assistant_msg is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.assistant_msg,
                    "response_time": turn.assistant_runtime_cost,
                }
            )
        if turn.user_msg is not None:
            messages.append(
                {
                    "role": "user",
                    "content": turn.user_msg,
                    "response_time": turn.user_runtime_cost,
                }
            )
    return messages
