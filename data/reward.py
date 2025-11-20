"""
Utility functions for defining reward functions.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Callable, Dict, Optional, List, Tuple, Any
from collections import defaultdict
import numpy as np
from utils.misc import parse_for_answer_tags
import lotus
from lotus.models import LM
import pandas as pd
from collections import Counter


def get_avg_score(
    prompt: str,
    test_dataset: Dict[str, str],
    models: List[str],
    score_fn: Callable[[Any, str], float],
    validate_fn: Callable[[str], bool] = lambda x: True,
    final_answer_desc: str = "answer",
) -> Tuple[Dict[str, float], Dict[str, Tuple[List[str], List[float]]]]:
    """
    R(y) where y is a prompt and R is the average score of models using that prompt.

    Gets the accuracy of each model on a test dataset.
    We use LOTUS to quickly run the models on the test dataset.
    Models are allowed one retry if they return an invalid output.
    Note: LOTUS cannot handle any prompt containing {*}, so we need to escape the { and } in the prompt.

    Args:
        prompt (str): The prompt to use for the model.
        test_dataset (Dict[str, str]): The test dataset.
        models (List[str]): The models to use.
        validate_fn (Callable[[str], bool]): The function to use to validate the model's output.
        score_fn (Callable[[Any, str], float]): The function to use to score the model's output.
            The function should take in z and zhat and return a float

    Returns:
        Dict[str, float]: The average score of each model on the test dataset.
        Dict[str, Tuple[List[str], List[float]]]: The outputs and scores of each model on each test case.
    """
    prompt = prompt.replace("{", "{{").replace("}", "}}")
    prompt += f"\n\nWrap your final {final_answer_desc} in <answer></answer> tags. Place all other reasoning outside these tags."

    test_covariates = pd.DataFrame(test_dataset.items(), columns=["input", "output"])
    zhats = defaultdict(list)
    scores = defaultdict(list)

    for model_name in models:
        lotus.settings.configure(
            lm=LM(model=model_name, temperature=0.0, max_tokens=10000)
        )
        outputs = test_covariates.sem_map(prompt + "\n\n{input}")
        outputs["yhat"] = (
            outputs["_map"].apply(lambda x: parse_for_answer_tags(x)).tolist()
        )
        outputs["_first_round_valid"] = outputs["yhat"].apply(validate_fn)
        revalidate = outputs[~outputs["_first_round_valid"]].sem_map(
            prompt + "\n\n{input}"
        )
        outputs["yhat"] = outputs["yhat"].where(
            outputs["_first_round_valid"], revalidate["_map"]
        )
        zhats[model_name] = outputs["yhat"].tolist()
        scores[model_name] = []
        for z, zhat in zip(test_dataset.values(), zhats[model_name]):
            try:
                scores[model_name].append(score_fn(z, zhat))
            except Exception as e:
                print(f"Error scoring {model_name} on {z}: {e}")
                scores[model_name].append(float("nan"))

    return {model_name: np.mean(scores[model_name]) for model_name in models}, {
        model_name: (zhats[model_name], scores[model_name]) for model_name in models
    }


def _multiset_jaccard_reward(d: dict):
    """
    The reward function is the Jaccard similarity between the true and predicted sets.
    """
    true_set = d.get("true_set", [])
    true_counts = Counter(true_set)

    def ms_jaccard(z):
        if len(z) > 0 and len(z) < len(true_set):
            # behave like any_in
            return sum(1 for v in z if v in true_set) / len(z)
        pred_counts = Counter(z)
        intersection = sum((true_counts & pred_counts).values())
        union = sum((true_counts | pred_counts).values())
        if union > 0:
            return intersection / union
        else:
            # Both sets are empty - perfect match
            return 1.0

    return lambda z: ms_jaccard(z)


def _radial_band_reward(d: dict):
    """
    The reward function is 1 if z is in the band [lower, upper], -1 if z is outside the band, and a smooth transition in between.
    """
    lower = d.get("lower", 0)
    upper = d.get("upper", 1)
    sigma = d.get("sigma", 1)
    none_val = d.get("none_val", 0)
    if lower == upper:
        return lambda z: none_val if z is None else (1 if z == lower else -1)
    else:

        def reward(z):
            if z is None:
                return none_val
            if lower <= z <= upper:
                return 1.0
            dist = max(0, abs(z - (lower + upper) / 2) - (upper - lower) / 2)
            return 2 * np.exp(-(dist**2) / (2 * sigma**2)) - 1

        return reward


def _hinge_reward(d: dict) -> Callable[[float], float]:
    """
    The reward function is a line that passes through (lower, lower_val) and (upper, upper_val).
    The line is clipped to be bounded by min_val and max_val.
    """
    lower = d.get("lower", 0)
    upper = d.get("upper", 1)
    lower_val = d.get("lower_val", -1)
    upper_val = d.get("upper_val", 1)
    min_val = d.get("min_val", min(lower_val, upper_val))
    max_val = d.get("max_val", max(lower_val, upper_val))
    none_val = d.get("none_val", 0)
    if lower == upper:
        return lambda z: (
            none_val if z is None else (upper_val if z == lower else lower_val)
        )
    else:
        return lambda z: (
            none_val
            if z is None
            else min(
                max_val,
                max(
                    min_val,
                    lower_val + (upper_val - lower_val) * (z - lower) / (upper - lower),
                ),
            )
        )


@dataclass
class Constraint:
    """
    Represents a hard or soft constraint that is part of a reward function.
    When called on an output y, the constraint c(y) returns a value in [-1, 1].

    We assume an extractor f(y) -> Any and define c(y) = g(f(y)) with the following types:

    - BOOLEAN_PENALIZE_FALSE (penalize; hard-suitable):
        z is a boolean. g(z) = 1 if z is False (violation active), else 0.
    - BOOLEAN_REWARD_TRUE (reward):
        z is a boolean. g(z) = 1 if z is True, else 0.
    - PENALIZE_ANY_NOT_IN_SET (penalize; set-fraction):
        z is an iterable. g(z) = fraction of elements not in required_set (in [0,1]).
    - PENALIZE_ANY_IN_SET (penalize; set-fraction):
        z is an iterable. g(z) = fraction of elements in bad_set (in [0,1]).
    - REWARD_ANY_IN_SET (reward; set-fraction):
        z is an iterable. g(z) = fraction of elements in good_set (in [0,1]).
    - REWARD_ANY_NOT_IN_SET (reward; set-fraction):
        z is an iterable. g(z) = fraction of elements not in required_set (in [0,1]).
    - GOOD_SET_BAD_SET (signed set membership):
        z is a value. g(z) = 1 if z in good_set; -1 if z in bad_set; 0 otherwise.
    - MULTISET_JACCARD (multiset Jaccard similarity):
        z is an iterable. g(z) = Jaccard similarity between z and true_set (in [0,1]).
    - RADIAL_BAND (continuous reward band):
        z is numeric. g(z) = 1 inside [lower, upper], decays smoothly toward -1 outside (sigma controls decay).
    - HINGE (clipped linear mapping):
        z is numeric in [lower, upper] mapped to [lower_val, upper_val], then clipped to [min_val, max_val].

    Attributes:
        description (str): Human-readable description of the constraint.
        func (Callable[[Any], float]): Mapping from extractor output to [-1, 1].
        is_hard (bool): Whether the constraint is hard or soft.
            A hard constraint will set the final reward to -1 as soon as c(y) = -1.
    """

    description: Optional[str]
    extractor: Callable[[Any], Any]  # f in the description above
    func: Callable[[Any], float]  # g in the description above
    is_hard: bool
    type: str
    _kwargs: Dict[str, Any]
    is_discoverable_by_questions: bool = True
    is_discoverable: bool = True
    is_minimal: bool = False
    extractor_kwargs: Dict[str, Any] = None
    id: str = field(init=False)

    def __call__(self, y: Any) -> float:
        if self.extractor_kwargs is None:
            self.extractor_kwargs = {}
        extractor_result = self.extractor(y, **self.extractor_kwargs)

        # Handle extractors that return tuples (value, message)
        if isinstance(extractor_result, tuple) and len(extractor_result) == 2:
            value, message = extractor_result
            # Store the message for potential debugging (could be added to metadata later)
            self._last_message = message
        else:
            value = extractor_result
            self._last_message = None

        return self.func(value)

    def __post_init__(self):
        # Create a stable ID from a normalized description
        normalized_desc = (self.description or "").strip().lower()
        h = hashlib.sha1(normalized_desc.encode("utf-8")).hexdigest()[:12]
        self.id = f"c_{h}"

    @property
    def oracle_value(self) -> float:
        """
        Best-case value for unconstrained optimization (given our phi semantics).
        """
        if self.type in [
            "boolean_penalize_false",
            "penalize_any_not_in_set",
            "penalize_any_in_set",
        ]:
            return 0.0
        if self.type in [
            "boolean_reward_true",
            "good_set_bad_set",
            "radial_band",
            "reward_any_in_set",
            "reward_any_not_in_set",
            "multiset_jaccard",
        ]:
            return 1.0
        if self.type == "hinge":
            return float(self._kwargs.get("max_val", 1))

    @property
    def worst_case_value(self) -> float:
        """
        Worst-case value for unconstrained optimization (given our phi semantics).
        """
        if self.type in [
            "boolean_penalize_false",
            "penalize_any_not_in_set",
            "penalize_any_in_set",
        ]:
            return 1.0
        if self.type == "good_set_bad_set":
            return -1.0
        if self.type == "radial_band":
            return -1.0
        if self.type in [
            "boolean_reward_true",
            "reward_any_in_set",
            "reward_any_not_in_set",
            "multiset_jaccard",
        ]:
            return 0.0
        if self.type == "hinge":
            return float(self._kwargs.get("min_val", -1))

    def __str__(self):
        out = []
        if self.description is not None:
            out.append(f"description={self.description}")
        out.append(f"is_hard={self.is_hard}")
        out.append(f"is_discoverable={self.is_discoverable}")
        out.append(f"is_minimal={self.is_minimal}")
        return f"Constraint(type={self.type}, {', '.join(out)})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "description": self.description,
            "is_hard": self.is_hard,
            "is_discoverable": self.is_discoverable,
            "is_discoverable_by_questions": self.is_discoverable_by_questions,
            "is_minimal": self.is_minimal,
            "extractor": self.extractor,
            "extractor_kwargs": self.extractor_kwargs,
            **self._kwargs,
        }

    @property
    def default_weight_sign(self) -> int:
        """
        Returns +1 for reward-like constraints and -1 for penalize-like constraints.
        This allows initializing signed weights while leaving scoring unchanged.
        """
        if self.type in [
            "boolean_penalize_false",
            "penalize_any_not_in_set",
            "penalize_any_in_set",
        ]:
            return -1
        # Treat others as reward-oriented by default
        return 1

    @classmethod
    def from_dict(cls, d: dict, extractor_lookup=None) -> "Constraint":
        """
        Create a constraint from a dictionary representation.

        Args:
            d: Dictionary with constraint specification. Must contain:
                - "type": Type of constraint (see below for supported types)
                - "description": Human-readable description
                - "is_hard": Boolean indicating if this is a hard constraint
                - Additional fields depending on the constraint type
            extractor_lookup: dict mapping extractor function names to callables
        Returns:
            Constraint: The constructed constraint
        """
        if extractor_lookup is None:
            raise ValueError(
                "extractor_lookup must be provided to from_dict to resolve extractor functions."
            )
        constraint_type = d["type"]
        extractor_name = d["extractor"]
        if extractor_name is None:

            def extractor(y):
                return y
        else:
            extractor = extractor_lookup[extractor_name]
        extractor_kwargs = d.get("extractor_kwargs", {})
        if constraint_type == "boolean_penalize_false":
            assert "none_val" in d
            return cls(
                description=d["description"],
                func=lambda z: d["none_val"]
                if z is None
                else (1.0 if (z is False) else 0.0),
                is_hard=d["is_hard"],
                is_discoverable=d.get("is_discoverable", True),
                is_discoverable_by_questions=d.get("is_discoverable_by_questions", True),
                is_minimal=d.get("is_minimal", False),
                type="boolean_penalize_false",
                extractor=extractor,
                _kwargs={"none_val": d["none_val"]},
                extractor_kwargs=extractor_kwargs,
            )
        elif constraint_type == "boolean_reward_true":
            assert "none_val" in d
            return cls(
                description=d["description"],
                func=lambda z: d["none_val"] if z is None else (1 if z else 0),
                is_hard=d["is_hard"],
                is_discoverable=d.get("is_discoverable", True),
                is_discoverable_by_questions=d.get("is_discoverable_by_questions", True),
                is_minimal=d.get("is_minimal", False),
                type="boolean_reward_true",
                extractor=extractor,
                _kwargs={"none_val": d["none_val"]},
                extractor_kwargs=extractor_kwargs,
            )
        elif constraint_type == "penalize_any_not_in_set":
            assert "required_set" in d and len(d["required_set"]) > 0
            assert "none_val" in d
            return cls(
                description=d["description"],
                func=lambda z: (
                    d["none_val"]
                    if z is None
                    else (
                        (
                            lambda vals: (
                                sum(1 for v in vals if v not in d["required_set"])
                                / max(len(vals), 1)
                            )
                        )(list(z) if hasattr(z, "__iter__") else [])
                    )
                ),
                is_hard=d["is_hard"],
                is_discoverable=d.get("is_discoverable", True),
                is_discoverable_by_questions=d.get("is_discoverable_by_questions", True),
                is_minimal=d.get("is_minimal", False),
                type="penalize_any_not_in_set",
                extractor=extractor,
                _kwargs={
                    "required_set": d["required_set"],
                    "none_val": d["none_val"],
                },
                extractor_kwargs=extractor_kwargs,
            )
        elif constraint_type == "penalize_any_in_set":
            assert "none_val" in d
            assert "bad_set" in d and len(d["bad_set"]) > 0
            return cls(
                description=d["description"],
                func=lambda z: (
                    d["none_val"]
                    if z is None
                    else (
                        (
                            lambda vals: (
                                sum(1 for v in vals if v in d["bad_set"])
                                / max(len(vals), 1)
                            )
                        )(list(z) if hasattr(z, "__iter__") else [])
                    )
                ),
                is_hard=d["is_hard"],
                is_discoverable=d.get("is_discoverable", True),
                is_discoverable_by_questions=d.get("is_discoverable_by_questions", True),
                is_minimal=d.get("is_minimal", False),
                type="penalize_any_in_set",
                extractor=extractor,
                _kwargs={"bad_set": d["bad_set"], "none_val": d["none_val"]},
                extractor_kwargs=extractor_kwargs,
            )
        elif constraint_type == "good_set_bad_set":
            assert "none_val" in d
            assert "good_set" in d and len(d["good_set"]) > 0
            assert "bad_set" in d and len(d["bad_set"]) > 0
            return cls(
                description=d["description"],
                func=lambda z: (
                    d["none_val"]
                    if z is None
                    else (1 if z in d["good_set"] else -1 if z in d["bad_set"] else 0)
                ),
                is_hard=d["is_hard"],
                is_discoverable=d.get("is_discoverable", True),
                is_discoverable_by_questions=d.get("is_discoverable_by_questions", True),
                is_minimal=d.get("is_minimal", False),
                type="good_set_bad_set",
                extractor=extractor,
                _kwargs={
                    "good_set": d["good_set"],
                    "bad_set": d["bad_set"],
                    "none_val": d["none_val"],
                },
                extractor_kwargs=extractor_kwargs,
            )
        elif constraint_type == "reward_any_in_set":
            assert "none_val" in d
            assert "good_set" in d and len(d["good_set"]) > 0
            return cls(
                description=d["description"],
                func=lambda z: (
                    d["none_val"]
                    if z is None
                    else (
                        (
                            lambda vals: (
                                sum(1 for v in vals if v in d["good_set"])
                                / max(len(vals), 1)
                            )
                        )(list(z) if hasattr(z, "__iter__") else [])
                    )
                ),
                is_hard=d["is_hard"],
                is_discoverable=d.get("is_discoverable", True),
                is_discoverable_by_questions=d.get("is_discoverable_by_questions", True),
                is_minimal=d.get("is_minimal", False),
                type="reward_any_in_set",
                extractor=extractor,
                _kwargs={"good_set": d["good_set"], "none_val": d["none_val"]},
                extractor_kwargs=extractor_kwargs,
            )
        elif constraint_type == "reward_any_not_in_set":
            assert "none_val" in d
            assert "required_set" in d and len(d["required_set"]) > 0
            return cls(
                description=d["description"],
                func=lambda z: (
                    d["none_val"]
                    if z is None
                    else (
                        (
                            lambda vals: (
                                sum(1 for v in vals if v not in d["required_set"])
                                / max(len(vals), 1)
                            )
                        )(list(z) if hasattr(z, "__iter__") else [])
                    )
                ),
                is_hard=d["is_hard"],
                is_discoverable=d.get("is_discoverable", True),
                is_discoverable_by_questions=d.get("is_discoverable_by_questions", True),
                is_minimal=d.get("is_minimal", False),
                type="reward_any_not_in_set",
                extractor=extractor,
                _kwargs={"required_set": d["required_set"], "none_val": d["none_val"]},
                extractor_kwargs=extractor_kwargs,
            )
        elif constraint_type == "multiset_jaccard":
            assert "true_set" in d
            return cls(
                description=d["description"],
                func=_multiset_jaccard_reward(d),
                is_hard=d["is_hard"],
                is_discoverable=d.get("is_discoverable", True),
                is_discoverable_by_questions=d.get("is_discoverable_by_questions", True),
                is_minimal=d.get("is_minimal", False),
                type="multiset_jaccard",
                extractor=extractor,
                _kwargs={"true_set": d["true_set"]},
                extractor_kwargs=extractor_kwargs,
            )
        elif constraint_type == "radial_band":
            assert "none_val" in d
            assert "lower" in d and "upper" in d
            assert "sigma" in d
            return cls(
                description=d["description"],
                func=_radial_band_reward(d),
                is_hard=d["is_hard"],
                is_discoverable=d.get("is_discoverable", True),
                is_discoverable_by_questions=d.get("is_discoverable_by_questions", True),
                is_minimal=d.get("is_minimal", False),
                type="radial_band",
                extractor=extractor,
                _kwargs={
                    "lower": d["lower"],
                    "upper": d["upper"],
                    "sigma": d["sigma"],
                    "none_val": d["none_val"],
                },
                extractor_kwargs=extractor_kwargs,
            )
        elif constraint_type == "hinge":
            assert "none_val" in d
            assert "lower" in d and "upper" in d
            assert "lower_val" in d and "upper_val" in d
            assert "min_val" in d and "max_val" in d
            return cls(
                description=d["description"],
                func=_hinge_reward(d),
                is_hard=d["is_hard"],
                is_discoverable=d.get("is_discoverable", True),
                is_discoverable_by_questions=d.get("is_discoverable_by_questions", True),
                is_minimal=d.get("is_minimal", False),
                type="hinge",
                extractor=extractor,
                _kwargs={
                    "lower": d["lower"],
                    "upper": d["upper"],
                    "lower_val": d["lower_val"],
                    "upper_val": d["upper_val"],
                    "min_val": d["min_val"],
                    "max_val": d["max_val"],
                    "none_val": d["none_val"],
                },
                extractor_kwargs=extractor_kwargs,
            )
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

    @classmethod
    def create_boolean_penalize_false_constraint(
        cls,
        description: Optional[str] = None,
        is_hard: bool = False,
        is_discoverable: bool = True,
        is_discoverable_by_questions: bool = True,
        is_minimal: bool = False,
        extractor: str = None,
        extractor_kwargs: Dict[str, Any] = None,
        none_val: float = 0,
    ) -> dict:
        return {
            "type": "boolean_penalize_false",
            "description": description,
            "is_hard": is_hard,
            "is_discoverable": is_discoverable,
            "is_discoverable_by_questions": is_discoverable_by_questions,
            "is_minimal": is_minimal,
            "extractor": extractor,
            "extractor_kwargs": extractor_kwargs,
            "none_val": none_val,
        }

    @classmethod
    def create_boolean_reward_true_constraint(
        cls,
        description: Optional[str] = None,
        is_hard: bool = False,
        is_discoverable: bool = True,
        is_discoverable_by_questions: bool = True,
        is_minimal: bool = False,
        extractor: str = None,
        extractor_kwargs: Dict[str, Any] = None,
        none_val: float = 0,
    ) -> dict:
        return {
            "type": "boolean_reward_true",
            "description": description,
            "is_hard": is_hard,
            "is_discoverable": is_discoverable,
            "is_discoverable_by_questions": is_discoverable_by_questions,
            "is_minimal": is_minimal,
            "extractor": extractor,
            "extractor_kwargs": extractor_kwargs,
            "none_val": none_val,
        }

    @classmethod
    def create_penalize_any_not_in_set_constraint(
        cls,
        required_set: List[Any],
        description: Optional[str] = None,
        is_hard: bool = False,
        is_discoverable: bool = True,
        is_discoverable_by_questions: bool = True,
        is_minimal: bool = False,
        extractor: str = None,
        extractor_kwargs: Dict[str, Any] = None,
        none_val: float = 0,
    ) -> dict:
        assert len(required_set) > 0
        return {
            "type": "penalize_any_not_in_set",
            "description": description,
            "is_hard": is_hard,
            "is_discoverable": is_discoverable,
            "is_discoverable_by_questions": is_discoverable_by_questions,
            "is_minimal": is_minimal,
            "extractor": extractor,
            "required_set": required_set,
            "extractor_kwargs": extractor_kwargs,
            "none_val": none_val,
        }

    @classmethod
    def create_penalize_any_in_set_constraint(
        cls,
        bad_set: List[Any],
        description: Optional[str] = None,
        is_hard: bool = False,
        is_discoverable: bool = True,
        is_discoverable_by_questions: bool = True,
        is_minimal: bool = False,
        extractor: str = None,
        extractor_kwargs: Dict[str, Any] = None,
        none_val: float = 0,
    ) -> dict:
        return {
            "type": "penalize_any_in_set",
            "description": description,
            "is_hard": is_hard,
            "is_discoverable": is_discoverable,
            "is_discoverable_by_questions": is_discoverable_by_questions,
            "is_minimal": is_minimal,
            "extractor": extractor,
            "bad_set": bad_set,
            "extractor_kwargs": extractor_kwargs,
            "none_val": none_val,
        }

    @classmethod
    def create_reward_any_in_set_constraint(
        cls,
        good_set: List[Any],
        description: Optional[str] = None,
        is_hard: bool = False,
        is_discoverable: bool = True,
        is_discoverable_by_questions: bool = True,
        is_minimal: bool = False,
        extractor: str = None,
        extractor_kwargs: Dict[str, Any] = None,
        none_val: float = 0,
    ) -> dict:
        assert len(good_set) > 0
        return {
            "type": "reward_any_in_set",
            "description": description,
            "is_hard": is_hard,
            "is_discoverable": is_discoverable,
            "is_discoverable_by_questions": is_discoverable_by_questions,
            "is_minimal": is_minimal,
            "extractor": extractor,
            "good_set": good_set,
            "extractor_kwargs": extractor_kwargs,
            "none_val": none_val,
        }

    @classmethod
    def create_reward_any_not_in_set_constraint(
        cls,
        required_set: List[Any],
        description: Optional[str] = None,
        is_hard: bool = False,
        is_discoverable: bool = True,
        is_discoverable_by_questions: bool = True,
        is_minimal: bool = False,
        extractor: str = None,
        extractor_kwargs: Dict[str, Any] = None,
        none_val: float = 0,
    ) -> dict:
        assert len(required_set) > 0
        return {
            "type": "reward_any_not_in_set",
            "description": description,
            "is_hard": is_hard,
            "is_discoverable": is_discoverable,
            "is_discoverable_by_questions": is_discoverable_by_questions,
            "is_minimal": is_minimal,
            "extractor": extractor,
            "required_set": required_set,
            "extractor_kwargs": extractor_kwargs,
            "none_val": none_val,
        }

    @classmethod
    def create_good_set_bad_set_constraint(
        cls,
        good_set: List[Any],
        bad_set: List[Any],
        description: Optional[str] = None,
        is_hard: bool = False,
        is_discoverable: bool = True,
        is_discoverable_by_questions: bool = True,
        is_minimal: bool = False,
        extractor: str = None,
        extractor_kwargs: Dict[str, Any] = None,
        none_val: float = 0,
    ) -> dict:
        return {
            "type": "good_set_bad_set",
            "description": description,
            "is_hard": is_hard,
            "is_discoverable": is_discoverable,
            "is_discoverable_by_questions": is_discoverable_by_questions,
            "is_minimal": is_minimal,
            "extractor": extractor,
            "good_set": good_set,
            "bad_set": bad_set,
            "extractor_kwargs": extractor_kwargs,
            "none_val": none_val,
        }

    @classmethod
    def create_multiset_jaccard_constraint(
        cls,
        true_set: List[Any],
        description: Optional[str] = None,
        is_hard: bool = False,
        is_discoverable: bool = True,
        is_discoverable_by_questions: bool = True,
        is_minimal: bool = False,
        extractor: str = None,
        extractor_kwargs: Dict[str, Any] = None,
    ) -> dict:
        return {
            "type": "multiset_jaccard",
            "description": description,
            "is_hard": is_hard,
            "is_discoverable": is_discoverable,
            "is_discoverable_by_questions": is_discoverable_by_questions,
            "is_minimal": is_minimal,
            "extractor": extractor,
            "true_set": true_set,
            "extractor_kwargs": extractor_kwargs,
        }

    @classmethod
    def create_radial_band_constraint(
        cls,
        lower: float,
        upper: float,
        sigma: float,
        description: Optional[str] = None,
        is_hard: bool = False,
        is_discoverable: bool = True,
        is_discoverable_by_questions: bool = True,
        is_minimal: bool = False,
        extractor: str = None,
        extractor_kwargs: Dict[str, Any] = None,
        none_val: float = 0,
    ) -> dict:
        assert lower <= upper
        assert sigma > 0
        return {
            "type": "radial_band",
            "lower": lower,
            "upper": upper,
            "sigma": sigma,
            "description": description,
            "is_hard": is_hard,
            "is_discoverable": is_discoverable,
            "is_discoverable_by_questions": is_discoverable_by_questions,
            "is_minimal": is_minimal,
            "extractor": extractor,
            "extractor_kwargs": extractor_kwargs,
            "none_val": none_val,
        }

    @classmethod
    def create_hinge_constraint(
        cls,
        lower: float,
        upper: float,
        lower_val: float,
        upper_val: float,
        min_val: float = None,
        max_val: float = None,
        description: Optional[str] = None,
        is_hard: bool = False,
        is_discoverable: bool = True,
        is_discoverable_by_questions: bool = True,
        is_minimal: bool = False,
        extractor: str = None,
        extractor_kwargs: Dict[str, Any] = None,
        none_val: float = 0,
    ) -> dict:
        assert lower <= upper
        assert -1 <= lower_val <= 1
        assert -1 <= upper_val <= 1
        if min_val is not None and max_val is not None:
            assert min_val <= max_val
            assert -1 <= min_val <= 1
            assert -1 <= max_val <= 1
        return {
            "type": "hinge",
            "lower": lower,
            "upper": upper,
            "lower_val": lower_val,
            "upper_val": upper_val,
            "min_val": min_val if min_val is not None else min(lower_val, upper_val),
            "max_val": max_val if max_val is not None else max(lower_val, upper_val),
            "description": description,
            "is_hard": is_hard,
            "is_discoverable": is_discoverable,
            "is_discoverable_by_questions": is_discoverable_by_questions,
            "is_minimal": is_minimal,
            "extractor": extractor,
            "extractor_kwargs": extractor_kwargs,
            "none_val": none_val,
        }


def linear_reward(
    y: Any,
    constraints: List[Constraint],
    weights: np.array,
    enforce_hard: bool = True,
    raise_errors: bool = False,
) -> Tuple[float, bool, float, float, Dict[str, float]]:
    """
    R(y) = sum_i w_i * f_i(y)

    If enforce_hard is True, then the reward function is -1 if any hard constraint is violated.
    Otherwise, the hard constraints are just treated as linear terms.

    Note: All weights should be positive when passed to this function. The function will
    automatically negate weights for penalty constraints (boolean_penalize_false,
    penalize_any_not_in_set, penalize_any_in_set).

    Args:
        constraints (List[Constraint]): The constraints to use for the reward function.
        weights (List[float]): The weights to use for the reward function (should all be positive).
        enforce_hard (bool): Whether to enforce hard constraints.
        raise_errors (bool): Whether to raise errors if the solution is invalid.

    Returns:
        Tuple[bool, float, float, Dict[str, float]]
            bool: False if any hard constraints are violated and enforce_hard is True
            float: the final score
            float: the minimum possible score (unconstrained)
            float: the maximum possible score (unconstrained)
            Dict[str, float]: the score for each constraint
    """
    # If no constraints are provided, default to fully valid and full score
    if constraints is None or len(constraints) == 0:
        metadata = {
            "constraint_info": {},
            "weights": weights,
            "violated_constraints": [],
        }
        return True, 1.0, 1.0, 1.0, metadata

    # Apply sign correction: negate weights for penalty constraints
    # Penalty constraint types: boolean_penalize_false, penalize_any_not_in_set, penalize_any_in_set
    penalty_types = {
        "boolean_penalize_false",
        "penalize_any_not_in_set",
        "penalize_any_in_set",
    }
    if weights is not None:
        adjusted_weights = np.array(
            [
                -w if constraints[i].type in penalty_types else w
                for i, w in enumerate(weights)
            ]
        )
    else:
        adjusted_weights = None

    results = {}
    for i, constraint in enumerate(constraints):
        try:
            results[i] = constraint(y)
        except Exception as e:
            if raise_errors:
                raise Exception(f"Error when evaluating constraint {constraint.description}: {e}")
            results[i] = constraint.worst_case_value

    # Collect detailed messages from extractors
    detailed_messages = []
    for i, constraint in enumerate(constraints):
        if (
            hasattr(constraint, "_last_message")
            and constraint._last_message is not None
        ):
            detailed_messages.append(constraint._last_message)
        else:
            detailed_messages.append(None)

    metadata = {
        "constraint_info": {
            constraints[i].description: (results[i], detailed_messages[i], constraints[i].oracle_value, constraints[i].worst_case_value)
            for i in range(len(constraints))
        },
        "weights": weights,
        "violated_constraints": [],
    }

    def _is_violation(c: Constraint, val: float) -> bool:
        if c.type in [
            "boolean_penalize_false",
            "penalize_any_not_in_set",
            "penalize_any_in_set",
        ]:
            return val > 0
        # treat negative values as violations for reward-style
        return val < 0

    if enforce_hard:
        soft_constraint_indices = [
            i for i, constraint in enumerate(constraints) if not constraint.is_hard
        ]
        if weights is not None:
            assert len(weights) == len(soft_constraint_indices), (
                "Weights and constraints must have the same length"
            )
            score = sum(
                weight * results[i]
                for weight, i in zip(adjusted_weights, soft_constraint_indices)
            )
            # Since all weights are now positive (after adjustment), we can use oracle/worst_case directly
            max_unconstrained_score = sum(
                weight * constraints[i].oracle_value
                for weight, i in zip(adjusted_weights, soft_constraint_indices)
            )
            min_unconstrained_score = sum(
                weight * constraints[i].worst_case_value
                for weight, i in zip(adjusted_weights, soft_constraint_indices)
            )
        else:
            assert len(soft_constraint_indices) == 0
            score = None
            max_unconstrained_score = None
            min_unconstrained_score = None

        for i, constraint in enumerate(constraints):
            if constraint.is_hard and _is_violation(constraint, results[i]):
                if raise_errors:
                    detailed_msg = (
                        detailed_messages[i] or "No detailed message available"
                    )
                    raise Exception(
                        f"Failed hard constraint: {constraint.description}.\nDetails: {detailed_msg}"
                    )
                score = float("-inf")
                metadata["violated_constraints"].append(constraint.description)
    else:
        if weights is not None:
            assert len(weights) == len(constraints), (
                "Weights and constraints must have the same length"
            )
            score = sum(
                weight * results[i]
                for weight, i in zip(adjusted_weights, range(len(constraints)))
            )
            # Since all weights are now positive (after adjustment), we can use oracle/worst_case directly
            max_unconstrained_score = sum(
                weight * constraints[i].oracle_value
                for weight, i in zip(adjusted_weights, range(len(constraints)))
            )
            min_unconstrained_score = sum(
                weight * constraints[i].worst_case_value
                for weight, i in zip(adjusted_weights, range(len(constraints)))
            )
        else:
            assert len(constraints) == 0
            score = None
            max_unconstrained_score = None
            min_unconstrained_score = None

    return (
        (score != float("-inf")),
        score,
        min_unconstrained_score,
        max_unconstrained_score,
        metadata,
    )


def pairwise_win_rate(A: List[List[int]], B: List[List[int]]) -> float:
    """
    Compute the aggregated pairwise win rate of A over B across multiple levels.

    Suppose A and B are objects that contain "kinds" of items. If we sample two items of the same "kind" from both objects, this function computes the probability that A's item beats B's item.

    Parameters
    ----------
    A[i] : list of ranks (ints) for items in y's set at level i
    B[i] : list of ranks (ints) for items in y0's set at level i
        Lower rank = better.
        Ranks within each level i must be unique.

    Returns
    -------
    float
        Overall score in [0, 1]:
        - 1   : A completely dominates B
        - 0   : B completely dominates A
        - 0.5 : no information (both empty at that level or balanced overall)
    """

    total_wins = 0
    total_pairs = 0

    for Ai, Bi in zip(A, B):
        # Handle empty sets explicitly
        if not Ai and not Bi:
            level_score = 0.5  # no information
            level_pairs = 0
        elif not Ai:  # A empty, B non-empty
            level_score = 0.0  # B wins by default
            level_pairs = 1  # count as one "virtual" comparison
        elif not Bi:  # B empty, A non-empty
            level_score = 1.0  # A wins by default
            level_pairs = 1
        else:
            wins = 0
            for a in Ai:
                for b in Bi:
                    if a < b:  # lower rank = better
                        wins += 1
            level_pairs = len(Ai) * len(Bi)
            level_score = wins / level_pairs

        # accumulate weighted by number of comparable pairs
        total_wins += level_score * level_pairs
        total_pairs += level_pairs

    # If all levels empty (no pairs at all), return neutral 0.5
    if total_pairs == 0:
        return 0.5

    return total_wins / total_pairs


def likert_to_win_rate(
    comparison_likert_scores: List[str], return_total: bool = False
) -> float:
    """
    Convert a list of comparison likert scores ("A much more", "A slightly more", "Neutral", "B slightly more", "B much more") to P(A wins).

    Converts the likert score to a "pseudo-win":
    * A much more -> A wins 2x
    * A slightly more -> A wins 1x
    * Neutral -> A wins 1x and B wins 1x
    * B slightly more -> B wins 1x
    * B much more -> B wins 2x

    Parameters
    ----------
    comparison_likert_scores: List[str]
        List of comparison likert scores ("A much more", "A slightly more", "Neutral", "B slightly more", "B much more")

    Returns
    -------
    float
    """
    mapping = {
        "A much more": (2, 0),
        "A slightly more": (1, 0),
        "Neutral": (1, 1),
        "B slightly more": (0, 1),
        "B much more": (0, 2),
    }

    a_wins = 0
    b_wins = 0

    for score in comparison_likert_scores:
        if score not in mapping:
            raise ValueError(f"Invalid score: {score}")
        a_add, b_add = mapping[score]
        a_wins += a_add
        b_wins += b_add

    total = a_wins + b_wins
    if total == 0:
        return float("nan")  # undefined if no comparisons
    if return_total:
        return a_wins, total
    return a_wins / total
