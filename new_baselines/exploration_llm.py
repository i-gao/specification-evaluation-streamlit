from typing import Dict, Tuple, Optional, List
import json

from new_baselines.single_llm import SingleLLM
from utils.misc import (
    add_section,
    print_debug,
    parse_json,
)


class FeatureEnumerationLLM(SingleLLM):
    """
    Handles:
    - Hierarchical task decomposition into subtasks
    - Feature space brainstorming for each subtask
    - Subtask transitions

    Does NOT handle:
    - User state tracking (known/unknown features)
    - Exploration phase management
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Hierarchical planning state
        self.subtasks: Optional[List[Dict]] = None  # List of subtask dicts
        self.current_subtask_idx: int = 0
        self.current_subtask: Optional[Dict] = None

        # Feature tracking (scoped to current subtask)
        self.feature_space: Optional[List[Dict]] = (
            None  # [{name, importance, variance, priority}, ...]
        )

        # Register hooks
        self._pre_conversation_hooks = list(
            getattr(self, "_pre_conversation_hooks", [])
        ) + [
            self._hook_task_decomposition,
        ]

        self._post_user_response_hooks = list(
            getattr(self, "_post_user_response_hooks", [])
        ) + [
            self._insert_system_message,
            self._insert_user_message,
            self._hook_check_subtask_completion,
            self._hook_brainstorm_features,
        ]

    ############ HOOKS ############

    def _insert_system_message(self, hook_state: Dict) -> None:
        """Insert the system message on first turn."""
        if not self.has_seen_system_prompt:
            system_msg = self._get_generate_prompt()
            self.agent_executor.insert_message("system", system_msg)
            self.has_seen_system_prompt = True

    def _insert_user_message(self, hook_state: Dict) -> None:
        """Insert the most recent user message."""
        conversation_history = hook_state.get("conversation_history", []) or []
        last_user_msg = None
        if len(conversation_history) > 0 and getattr(
            conversation_history[-1], "user_msg", None
        ):
            last_user_msg = conversation_history[-1].user_msg
        print(last_user_msg)
        if last_user_msg is None:
            last_user_msg = ""
        self.agent_executor.insert_message("user", last_user_msg)

    def _hook_task_decomposition(self, hook_state: Dict) -> Dict:
        """
        Decompose the task into subtasks. Each subtask will be handled independently
        with its own feature space and exploration strategy.
        """
        initial_user_msg = None
        conversation_history = hook_state.get("conversation_history", []) or []
        if len(conversation_history) > 0 and getattr(
            conversation_history[-1], "user_msg", None
        ):
            initial_user_msg = conversation_history[-1].user_msg
        if initial_user_msg is None:
            initial_user_msg = ""

        prompt = (
            "system",
            f"""Consider the following task:

User request: {initial_user_msg}
{self.task_context}

**Analyze whether the user's request can be decomposed into independent subtasks (slots).**

A "slot" represents a distinct, independently-designable component of the solution. Create separate slots only when:
- Each component can be selected/designed independently
- Each component serves a different purpose or category
- The user would naturally think about these as separate decisions

## Example 1: "Plan a trip to Paris"

- **Decomposable:** Yes - flights, hotels, and restaurants are independent choices
- **Output:**

```json
[
  {{"name": "flight", "description": "Select round-trip flights to/from Paris"}},
  {{"name": "hotel", "description": "Select accommodation in Paris"}},
  {{"name": "restaurants", "description": "Select dining options in Paris"}}
]
```

## Example 2: "Find a hoodie from the catalog"

- **Decomposable:** No - this is a single product selection
- **Output:**

```json
[
  {{"name": "product", "description": "Select a hoodie from the catalog"}}
]
```

## Your Task

1. Use available tools to understand the request and domain
2. Determine the minimal set of slots needed
3. Respond with a JSON array following this format:

```json
[
  {{"name": "slot_name", "description": "Clear description of this slot's purpose"}},
  ...
]
```

**Note:** You'll be able to ask clarifying questions for each slot later. Keep slots minimal - only create separate slots when components are truly independent.
""",
        )

        raw, _, _ = self._call_agent_executor(
            prompt,
            persist_state=False,
            min_react_steps=3 if len(self.actions) > 0 else 1,
        )

        content = raw.strip() if raw is not None else "[]"
        subtasks = parse_json(content) or []
        if not isinstance(subtasks, list) or len(subtasks) == 0:
            # Fallback: single solution
            subtasks = [{"name": "solution", "description": initial_user_msg}]

        self.subtasks = subtasks
        self.current_subtask_idx = 0
        self.current_subtask = subtasks[0] if subtasks else None

        if self.verbosity:
            print_debug(
                f"Hierarchical plan: {len(subtasks)} subtasks",
                "task_decomposition",
                color="orange",
            )
        return {"task_decomposition": subtasks}

    def _hook_brainstorm_features(self, hook_state: Dict) -> Dict:
        """
        Combined step: Analyze solution space variance and brainstorm prioritized features in one pass.

        Uses a single prompt that:
        1. Analyzes what features exist in the solution space and classifies their variance
        2. Immediately scores importance for each feature

        SCOPED TO CURRENT SUBTASK.
        Only runs if feature_space is None (not yet initialized for current subtask).
        """
        # Only run if we don't have features for the current subtask yet
        if self.feature_space is not None:
            return {"feature_space": self.feature_space}

        # Need a subtask to brainstorm for
        if not self.current_subtask:
            return {"feature_space": []}

        initial_user_msg = None
        conversation_history = hook_state.get("conversation_history", []) or []
        if len(conversation_history) > 0 and getattr(
            conversation_history[-1], "user_msg", None
        ):
            initial_user_msg = conversation_history[-1].user_msg
        if initial_user_msg is None:
            initial_user_msg = ""

        prompt = (
            "system",
            f"""Consider the following task:

User request: {initial_user_msg}
{self.task_context}
Current subtask: {self.current_subtask.get("name")} - {self.current_subtask.get("description")}

**Brainstorm a large list of features that the user might secretly care about when evaluating this subtask.** These features should range from general to quite specific. 

Include features that are:
* Computable: Can be determined or inferred using available tools
* Variable: Differ meaningfully across available options
* Relevant: Actually matter to users making this decision

For each feature, provide:

### `name` (string)
A clear, descriptive name for the feature
- Use natural language that users would understand
- Example: "Price", "Distance from city center", "Has free cancellation"

### `description` (string)
A detailed explanation including:
- What the feature measures or represents
- How to extract or infer it from available tools
- Any important nuances or edge cases

**Important:** If you cannot determine this feature from available tools, omit it entirely.

### `importance` (number: 0-100)
How much typical users care about this feature
- 80-100: Critical decision factor (e.g., price, safety)
- 50-79: Important consideration (e.g., reviews, amenities)
- 20-49: Nice-to-have preference (e.g., aesthetic details)
- 0-19: Minor detail (e.g., brand color scheme)

### `discriminative_value` (string: "high" | "medium" | "low")
How much identifying the desired value for this feature reduces the feasible set
- **High:** Identifying the desired value for this feature reduces the search set by a large amounts
- **Medium:** Identifying the desired value for this feature reduces the search set by a moderate amount
- **Low:** Identifying the desired value for this feature does not reduce the search set by a large amount

### `desired_value` (string or null)
The user's preference for this feature, if known
- Use when preferences are clear from context or previous statements
- Use `null` when the user's preference is unknown

## Your Task

1. **Brainstorm broadly:** Generate a comprehensive initial list ranging from obvious to subtle features
2. **Verify computability:** Use available tools to confirm each feature can be determined
3. **Filter rigorously:** Remove features that cannot be computed or have no variance
4. **Return JSON:** Provide a JSON array of feature objects

## Output Format

```json
[
  {{
    "name": "feature name",
    "description": "detailed description including how to compute it",
    "importance": 75,
    "variance": "high",
    "variance_justification": "specific evidence from solution space",
    "desired_value": "user preference or null"
  }},
  ...
]
```
""",
        )

        raw, _, _ = self._call_agent_executor(
            prompt,
            persist_state=False,
            min_react_steps=3 if len(self.actions) > 0 else 1,
        )

        content = raw.strip() if raw is not None else "[]"
        features_list = parse_json(content) or []
        if not isinstance(features_list, list):
            features_list = []

        # Process features: compute priority from variance and importance
        for feature in features_list:
            if not isinstance(feature, dict):
                continue

            # Convert variance to numeric score (0-10) for priority calculation
            variance = feature.get("variance", "medium")
            if variance == "high":
                variance_score = (
                    9  # high variance = more useful for distinguishing solutions
                )
            elif variance == "medium":
                variance_score = 5.5  # medium variance = moderately useful
            else:  # low
                variance_score = (
                    1.5  # low variance = less useful for distinguishing solutions
                )

            # Normalize importance from 0-100 to 0-10 scale
            importance_raw = feature.get("importance", 50)
            importance_normalized = importance_raw / 10.0

            # Compute priority = importance × variance_score (both on 0-10 scale)
            feature["priority"] = importance_normalized * variance_score

        # Sort by priority (highest first)
        features_list.sort(key=lambda f: f.get("priority", 0), reverse=True)
        self.feature_space = features_list

        if self.verbosity:
            print_debug(
                f"Features for subtask {self.current_subtask}: {self.feature_space}",
                "brainstorm_features",
                color="orange",
            )

        return {
            f"subtask_{self.current_subtask_idx}_feature_space": features_list,
        }

    def _hook_check_subtask_completion(self, hook_state: Dict) -> Dict:
        """
        Check if current subtask is complete and transition to next if so.
        Resets feature_space to None so that brainstorm hook will run for the new subtask.
        """
        if not self.subtasks or len(self.subtasks) <= 1:
            return {"subtask_transitioned": False}

        if self.current_subtask_idx >= len(self.subtasks) - 1:
            return {"subtask_transitioned": False}

        # Ask model if subtask is complete
        prompt = (
            "system",
            f"""Determine if the current subtask is reasonably complete based on the user's most recent response.

Current subtask: {self.current_subtask.get("name")} - {self.current_subtask.get("description")}

Respond with JSON:
{{"complete": true/false, "reason": "explanation"}}
""",
        )

        raw, _, _ = self._call_agent_executor(
            prompt, persist_state=False, min_react_steps=1
        )
        result = parse_json(raw.strip() if raw else "{}") or {}

        try:
            if result.get("complete", False):
                # Move to next subtask
                self.current_subtask_idx += 1
                self.current_subtask = self.subtasks[self.current_subtask_idx]

                # Reset feature space for new subtask
                # Setting feature_space to None will trigger brainstorm hook to run
                self.feature_space = None

                # Reset state for new subtask (override in derived class if needed)
                self._reset_subtask_state()

                if self.verbosity:
                    print_debug(
                        f"Transitioning to subtask {self.current_subtask_idx + 1}/{len(self.subtasks)}: {self.current_subtask.get('name')}",
                        "subtask_transition",
                        color="yellow",
                    )
                return {"subtask_transitioned": True}

        except Exception as e:
            print_debug(
                f"Error checking subtask completion: {e}",
                "subtask_transition",
                color="red",
            )
        return {"subtask_transitioned": False}

    def _reset_subtask_state(self) -> None:
        """Override in derived classes to reset subtask-specific state."""
        pass

    def generate_message(self, user_response: Optional[str] = None) -> Tuple[str, bool]:
        """Generate next message following exploration strategy."""
        if self.verbosity == 2:
            print_debug(
                f"Generating message with user_response:\n{user_response}",
                "generate_message",
                color="blue",
            )

        raw, _, _ = self._call_agent_executor()
        if raw is None:
            return None, False

        wants_to_end_conversation = "<END_CONVERSATION>" in raw
        assistant_msg = raw.replace("<END_CONVERSATION>", "")

        if self.verbosity:
            print_debug(
                f"Generated message: {raw}",
                "generate_message",
                color="orange",
            )

        return assistant_msg, wants_to_end_conversation


class ClarifyWithFeatureTrackingLLM(FeatureEnumerationLLM):
    def _get_generate_prompt(self) -> str:
        """System prompt for clarifying with feature tracking."""
        feature_summary = ""
        if self.feature_space:
            top_features = self.feature_space[:10]
            feature_summary = add_section(
                "Prioritized Features for Current Subtask (importance × variance)",
                json.dumps(top_features, indent=2),
            )

        # Add current subtask context
        current_subtask_section = ""
        if self.current_subtask and self.subtasks and len(self.subtasks) > 1:
            subtask_list = "\n".join(
                [
                    f"{i + 1}. {st.get('name')}: {st.get('description')}"
                    for i, st in enumerate(self.subtasks)
                ]
            )
            current_subtask_section = add_section(
                f"Plan - Currently on Subtask {self.current_subtask_idx + 1}/{len(self.subtasks)}",
                f"Full plan:\n{subtask_list}\n\nCurrent subtask: {self.current_subtask.get('name')} - {self.current_subtask.get('description')}\n\nFocus on completing THIS subtask before moving to the next one.",
            )

        if self._show_prediction_fmt_instructions_in_msg:
            fmt_instructions = f"\n\n{self.prediction_fmt_instructions}"
        else:
            fmt_instructions = ""

        return f"""You are a helpful assistant working with a user to complete a task. Often, users are unclear about their intent or context. Not knowing this information can make it difficult to provide a maximally helpful answer. Therefore, before executing the task (and possibly throughout the task), you should ask questions to clarify any ambiguities about the task with the user. However, avoid asking questions that are repetitive.

You know the following basic information about the task: 
{self.commonsense_instructions}

To complete this task, we've broken it down into subtasks:
{current_subtask_section}
{feature_summary}

Use the tools available to you to ground your work in the actual features of the task space. If there is a CSV of options, your work must use that CSV.

There are two kinds of messages you can send to the user: 1) a clarifying question to better specify the user's intent, or 2) a complete output for the subtask. You may not send the user intermediate options or explanations, unless they directly ask for these.
{fmt_instructions}
Work with the user. {self.msg_fmt_instructions} When you have finished the entire task AND received user confirmation of its completeness, generate the string <END_CONVERSATION>. To show a user a message, do not make tool calls in that message.

Remember to ask questions! You MUST ask clarifying questions on your first turn, BEFORE showing any results. To show a user a message, do not make tool calls in that message.
"""


class ExplorationLLM(FeatureEnumerationLLM):
    """
    Agent that explicitly manages exploration vs exploitation using a cognitive budget.
    Extends FeatureEnumerationLLM with user state tracking.

    Key principles:
    1. User has latent reward function <theta, phi(y)> but only knows subset of features
    2. Agent must help user discover important, discriminative features
    3. Exploration is front-loaded: show diverse options early, focused questions mid, exploit late
    4. Cognitive cost decreases over time as we transition from exploration to exploitation
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # User state tracking (scoped to current subtask)
        self.known_features = set()  # Features user has explicitly mentioned/seen
        self.unknown_features: List[str] = []  # Features not yet surfaced to user

        # Exploration state (scoped to current subtask)
        self.exploration_phase: str = "high"  # high -> medium -> low -> exploit
        self.remaining_exploration_budget: float = None  # Set dynamically

        # Register additional hooks for user state tracking
        self._post_user_response_hooks = list(
            getattr(self, "_post_user_response_hooks", [])
        ) + [
            self._hook_parse_user_response,
        ]

        self._pre_generation_hooks = list(
            getattr(self, "_pre_generation_hooks", [])
        ) + [
            self._hook_update_exploration_phase,
            self._hook_select_exploration_strategy,
        ]

    ############ HOOKS ############

    def _hook_brainstorm_features(self, hook_state: Dict) -> Dict:
        """Override to update unknown_features after brainstorming."""
        result = super()._hook_brainstorm_features(hook_state)

        # Update unknown_features based on feature_space
        if self.feature_space:
            self.unknown_features = [
                f.get("name")
                for f in self.feature_space
                if f.get("name") and f.get("name") not in self.known_features
            ]

        return result

    def _hook_parse_user_response(self, hook_state: Dict) -> Dict:
        """
        Extract which features the user mentioned/revealed in their response.
        Update known_features and unknown_features accordingly.
        """
        prompt = (
            "system",
            f"""Identify newly discovered features from the user's most recent message.

**Context:**
- Current subtask: {self.current_subtask.get("name")} - {self.current_subtask.get("description")}
- Features previously unknown to user:
{json.dumps(self.unknown_features, indent=2)}

**Task:**
Analyze the user's most recent message to determine if they have mentioned, asked about, or expressed interest in any features from the "previously unknown" list above.

**What counts as "newly discovered":**
- User explicitly mentions the feature by name or concept
- User asks questions about the feature
- User expresses preferences related to the feature
- User reacts to information about the feature

**What does NOT count:**
- Features merely implied by context but not actually discussed
- Features the user might care about but hasn't referenced

**Output:**
Return a JSON array of feature names (strings) that the user newly discovered.
- If the user mentioned or engaged with features: ["feature_name_1", "feature_name_2"]
- If no new features were discovered: []
""",
        )

        raw, _, _ = self._call_agent_executor(
            prompt, persist_state=False, min_react_steps=1
        )

        content = raw.strip() if raw is not None else "{}"
        discovered = parse_json(content) or []

        # Update known features
        for feature_name in discovered:
            if feature_name not in self.known_features:
                self.known_features.add(feature_name)
            if feature_name in self.unknown_features:
                self.unknown_features.remove(feature_name)

        if self.verbosity:
            print_debug(
                f"Discovered features: {discovered}\nKnown: {self.known_features}\nUnknown: {self.unknown_features[:5]}",
                "parse_user_response",
                color="orange",
            )
        return {"discovered_features": discovered}

    def _hook_update_exploration_phase(self, hook_state: Dict) -> Dict:
        """
        Update exploration phase based on cognitive budget and feature discovery progress.

        Phases:
        - high (>70% budget): Show diverse options, maximize feature discovery
        - medium (40-70% budget): Mix of options and targeted questions
        - low (20-40% budget): Focused questions on remaining important features
        - exploit (<20% budget): Generate best solution, refine
        """
        if self.remaining_exploration_budget is None:
            self.remaining_exploration_budget = self.interaction_budget

        budget_ratio = (
            self.interaction_budget - self.total_cost
        ) / self.interaction_budget

        # Also consider feature discovery progress
        total_features = len(self.feature_space) if self.feature_space else 1
        known_feature_ratio = len(self.known_features) / max(total_features, 1)

        # Phase transition logic: move to exploitation if budget low OR most features known
        if budget_ratio < 0.2 or known_feature_ratio > 0.8:
            new_phase = "exploit"
        elif budget_ratio < 0.4 or known_feature_ratio > 0.6:
            new_phase = "low"
        elif budget_ratio < 0.7 or known_feature_ratio > 0.4:
            new_phase = "medium"
        else:
            new_phase = "high"

        if new_phase != self.exploration_phase:
            if self.verbosity:
                print_debug(
                    f"Phase transition: {self.exploration_phase} -> {new_phase} "
                    f"(budget: {budget_ratio:.1%}, features: {known_feature_ratio:.1%})",
                    "update_phase",
                    color="yellow",
                )
            self.exploration_phase = new_phase

        return {"exploration_phase": new_phase, "budget_ratio": budget_ratio}

    def _hook_select_exploration_strategy(self, hook_state: Dict) -> Dict:
        """
        Provide exploration strategy guidance as internal context.
        """
        # Get top unknown features by priority
        unknown_prioritized = [
            f
            for f in (self.feature_space or [])
            if f.get("name") in self.unknown_features
        ][:3]  # Top 3

        phase = self.exploration_phase
        if phase == "exploit":
            phase_description = "EXPLOIT: Show single best recommendation or refine based on known preferences."
        elif phase == "low":
            phase_description = "LOW exploration: Show a best recommendation with a single targeted question."
        elif phase == "medium":
            phase_description = "MEDIUM exploration: 1-2 focused questions on remaining important features to minimize cognitive load."
        elif phase == "high":
            phase_description = "HIGH exploration: Maximize feature discovery with 1-2 questions (explaining features when asking about them) and small option sets."
        budget_ratio = (
            self.interaction_budget - self.total_cost
        ) / self.interaction_budget

        # Insert as internal guidance for generation
        strategy_guidance = f"""<think>Let me reflect on what message I should send next.

Current subtask: {self.current_subtask.get("name")} - {self.current_subtask.get("description")}
Current exploration phase: {phase_description}
Remaining budget: {budget_ratio:.1%} of total
Features the user has thought about: {self.known_features}
Most important features the user has not thought about (I should bring these up): {[f.get("name") for f in unknown_prioritized]}
</think>"""

        self.agent_executor.insert_message("assistant", strategy_guidance)

        if self.verbosity:
            print_debug(
                f"Exploration strategy guidance: phase={phase}, budget={budget_ratio:.1%}, "
                f"known={len(self.known_features)}, unknown_top3={[f.get('name') for f in unknown_prioritized]}",
                "select_exploration_strategy",
                color="orange",
            )

        return {"exploration_phase": phase, "budget_ratio": budget_ratio}

    def _reset_subtask_state(self) -> None:
        """Reset user state when transitioning to a new subtask."""
        self.known_features = set()
        self.unknown_features = []
        self.exploration_phase = "high"

    ############ PROMPTS ############

    def _get_generate_prompt(self) -> str:
        if self._show_prediction_fmt_instructions_in_msg:
            fmt_instructions = f"\n\n{self.prediction_fmt_instructions}"
        else:
            fmt_instructions = ""
        """System prompt emphasizing exploration-exploitation tradeoff."""
        return f"""You are a helpful assistant working with a user to complete a task.

Here is the plan for the conversation:
{self.subtasks}

Work with the user, one subtask at a time. Carefully manage the user's cognitive load: try not to ask more than 2 questions in a single message. Do not send > than 2 messages in a row which only ask questions. {self.msg_fmt_instructions} When you have finished the entire task AND received user confirmation of its completeness, generate the string <END_CONVERSATION>. To show a user a message, do not make tool calls in that message.
{fmt_instructions}
"""
