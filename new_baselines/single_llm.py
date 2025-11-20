from typing import List, Tuple, Optional, Dict


from new_baselines.policy import InteractionPolicy
from utils.misc import (
    add_section,
    Stopwatch,
    print_debug,
    parse_json,
)
from utils.model import LangChainModel, is_openai_model, is_anthropic_model


class SingleLLM(InteractionPolicy):
    def __init__(
        self,
        *args,
        model_name: str = "gpt-4o-mini",
        model_kwargs: dict = {},
        max_react_steps: int = 25,
        enforce_tool_use_on_first_turn: bool = True,
        insert_budget_check: bool = True,
        insert_system_message: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        is_hf = not is_openai_model(model_name) and not is_anthropic_model(model_name)
        self.agent_executor = LangChainModel(
            model_name=model_name,
            tools=self.actions,
            verbosity=self.verbosity,
            max_react_steps=max_react_steps,
            multiturn_memory=True,
            out_of_steps_msg="Sorry, I need some more time to think about this. Please give me the go-ahead to think some more.",
            list_tools_in_prompt=is_hf,
            add_thinking_tag=not is_hf,
            **model_kwargs,
        )
        self._max_react_steps = max_react_steps
        self._enforce_tool_use_on_first_turn = enforce_tool_use_on_first_turn

        # Register hooks
        self._pre_generation_hooks = list(getattr(self, "_pre_generation_hooks", []))
        if insert_budget_check:
            self._pre_generation_hooks.append(self._hook_budget_check)
        if insert_system_message:
            self._pre_generation_hooks.append(self._insert_system_message)

    def _call_agent_executor(
        self, *msgs: List[Tuple[str, str]], persist_state: bool = True, **kwargs
    ) -> Tuple[str, float, float]:
        """
        Call the agent executor and return the raw response, token cost, and runtime cost.

        Args:
            msgs: The new messages to append to the chain
                msgs[i] = (role, content)

        Returns:
            Tuple[str, float, float]: The final response, token cost, and runtime cost
        """
        with Stopwatch() as sw:
            self._model_lock = True
            # This method automatically handles out of steps errors & null prompts
            raw = self.agent_executor.generate(
                dialogs=[msgs],
                persist_state=persist_state,
                remove_thinking_tokens=True,
                **kwargs,
            )[0]
            self._model_lock = False

        # Look at the new messages and extract the tool calls for saving
        action_history = self._parse_langchain_response_to_actions(raw)
        action_history[-1].runtime_cost = sw.time

        self.action_history[self.turn_count].extend(action_history)

        # Anthropic models sometimes return lists of dicts in the 'content' field
        output = action_history[-1].content
        if isinstance(output, list):
            output = output[-1]
            output = output.get("text")

        return (
            output,
            sum(action.token_cost for action in action_history),
            sw.time,
        )

    def _get_generate_prompt(self) -> str:
        """
        Get the system message for the language model.
        """
        if self._show_prediction_fmt_instructions_in_msg:
            fmt_instructions = f"\n\n{self.prediction_fmt_instructions}"
        else:
            fmt_instructions = ""
        return f"""Use the tools available to you to ground your work in the actual features of the task space. If there is a CSV of options, your work must use that CSV.
        
Work with the user. {self.msg_fmt_instructions} When you have finished the entire task AND received user confirmation of its completeness, generate the string <END_CONVERSATION>. Do not generate this string in any other message, or the interaction will end prematurely. To show a user a message, do not make tool calls in that message.

{fmt_instructions}
"""

    def _get_predict_prompt(self) -> str:
        """
        Add a system message demanding a current prediction.
        """
        return add_section(
            "Generate the complete final output for the task",
            f"Based on the conversation history above, generate the best possible solution for the task. YOU MUST GENERATE THE SOLUTION NOW WITH NO OTHER TEXT.\n\nYou must follow this expected format for the solution:\n{self.prediction_fmt_instructions}",
        )

    def _get_k_predict_prompt(self, k: int) -> str:
        """
        Get a system message for getting k predictions from the strong model.
        """
        return add_section(
            f"Generate {k} complete final outputs for the task",
            f"Based on the conversation history above, generate {k} different solutions for the task. YOU MUST GENERATE THE SOLUTIONS NOW WITH NO OTHER TEXT.\n\nEach solution should be a distinct and follow the expected format:\n{self.prediction_fmt_instructions}\n\nGenerate exactly {k} solutions. Return a JSON array of strings, each representing a solution.",
        )

    def generate_message(self, user_response: Optional[str] = None) -> Tuple[str, bool]:
        """
        Generate the next message in the conversation.

        Returns:
            str: The next message in the conversation
        """
        # If this is the first turn, prepend the generate prompt
        if not self.has_seen_system_prompt:
            system_msg = self._get_generate_prompt()
            prompt = [("system", system_msg), ("user", user_response)]
            min_react_steps = (
                3
                if self._enforce_tool_use_on_first_turn and len(self.actions) > 0
                else 1
            )
            self.has_seen_system_prompt = True
        else:
            prompt = [("user", user_response)]
            min_react_steps = 1

        if self.verbosity == 2:
            print_debug(
                f"Generating message with prompt:\n{prompt}",
                "generate_message",
                color="blue",
            )

        # Call generate
        raw, _, _ = self._call_agent_executor(*prompt, min_react_steps=min_react_steps)
        if raw is None:
            return None, False

        # Parse the <END_CONVERSATION> tag
        wants_to_end_conversation = "<END_CONVERSATION>" in raw
        assistant_msg = raw.replace("<END_CONVERSATION>", "")

        if self.verbosity:
            print_debug(f"Generated message: {raw}", "generate_message", color="orange")

        return assistant_msg, wants_to_end_conversation

    def get_test_prediction(self) -> str:
        """
        Get the current prediction from the strong model.

        Returns:
            str: the prediction
        """
        prompt = [("system", self._get_predict_prompt())]
        if self.verbosity == 2:
            print_debug(
                f"Getting test prediction with prompt:\n{prompt}",
                "get_test_prediction",
                color="blue",
            )

        raw, _, _ = self._call_agent_executor(*prompt, persist_state=False)
        if self.verbosity:
            print_debug(
                f"Current prediction: {raw}", "get_test_prediction", color="orange"
            )

        return raw

    def get_test_predictions(self, k: int) -> List[str]:
        """
        Get k predictions from the strong model in one pass by prompting for k solutions.

        Args:
            k: Number of predictions to generate

        Returns:
            List[str]: List of k predictions
        """
        prompt = [("system", self._get_k_predict_prompt(k))]
        if self.verbosity == 2:
            print_debug(
                f"Getting test prediction with prompt:\n{prompt}",
                "get_test_prediction",
                color="blue",
            )

        raw, _, _ = self._call_agent_executor(*prompt, persist_state=False)
        if self.verbosity:
            print_debug(
                f"Current prediction: {raw}", "get_test_prediction", color="orange"
            )
        predictions = parse_json(raw)
        if predictions is None:
            return []
        else:
            return [self.spec.parse_solutions(str(p)) for p in predictions]

    def _hook_budget_check(self, hook_state: Dict) -> Dict:
        """Insert the reminder prompt on first turn."""
        reminder_prompt = f"""You've used {self.total_cost} / {self.interaction_budget} of the user's patience."""
        self.agent_executor.insert_message("system", reminder_prompt)

    def _insert_system_message(self, hook_state: Dict) -> Dict:
        """Insert the system message."""
        system_msg = self._get_generate_prompt()
        self.agent_executor.insert_message("system", system_msg)


######### SUBCLASSES #########


class RawLLM(SingleLLM):
    def _get_generate_prompt(self) -> str:
        return (
            f"""You are a helpful assistant working with a user to complete a task.

You know the following basic information about the task: 
{self.commonsense_instructions}
"""
            + "\n\n"
            + super()._get_generate_prompt()
        )


class ClarifyLLM(SingleLLM):
    def _get_generate_prompt(self) -> str:
        """
        Get the system message for the language model.
        """
        return (
            f"""You are a helpful assistant working with a user to complete a task. Often, users are unclear about their intent or context. Not knowing this information can make it difficult to provide a maximally helpful answer. Therefore, before executing the task (and possibly throughout the task), you should ask questions to clarify any ambiguities about the task with the user. However, avoid asking questions that are repetitive.

You know the following basic information about the task: 
{self.commonsense_instructions}

There are two kinds of messages you can send to the user: 1) a clarifying question to better specify the user's intent, or 2) a complete output for the task. You may not send the user intermediate options or explanations, unless they directly ask for these."""
            + "\n\n"
            + super()._get_generate_prompt()
            + "\n\n"
            + "Remember to ask questions! You MUST ask clarifying questions on your first turn, BEFORE showing any results."
        )


class ExecutionLLM(SingleLLM):
    def _get_generate_prompt(self) -> str:
        return (
            f"""You are a helpful assistant working with a user to complete a task.

You know the following basic information about the task: 
{self.commonsense_instructions}

You MUST SOLVE THE TASK IMMEDIATELY. Do not offer the user intermediate options. Do not offer samples for the user to choose from. Do not ask clarifying questions. If you haven't already solved the task, SOLVE IT NOW, IMMEDIATELY."""
            + "\n\n"
            + super()._get_generate_prompt()
            + "\n\n"
            + "When you have finished the entire task AND received user confirmation of its completeness, generate the string <END_CONVERSATION>. Do not generate this string in any other message, or the interaction will end prematurely. To show a user a message, do not make tool calls in that message."
        )


class AdversarialExecutionLLM(ExecutionLLM):
    def generate_message(self, user_response: str) -> str:
        if self.turn_count == 0 or (
            self.turn_count == 1 and self.conversation_history[0].assistant_msg is None
        ):
            return (
                "Think very carefully about what you would like for the task, and then tell me what you would like me to do. Make sure you include ALL the requirements that matter to you.",
                False,
            )
        return super().generate_message(user_response)


class ClarifyThenExecuteLLM(SingleLLM):
    """
    A subclass that first asks all clarifying questions upfront, then switches to execution mode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clarify_mode = True
        self._last_assistant_msg = None

        # Register hooks
        self._pre_generation_hooks = list(
            getattr(self, "_pre_generation_hooks", [])
        ) + [
            self._hook_check_mode_switch,
            self._hook_insert_mode_prompt,
        ]

        self._post_generation_hooks = list(
            getattr(self, "_post_generation_hooks", [])
        ) + [
            self._hook_process_special_tokens,
        ]

    ############ HOOKS ############

    def _hook_check_mode_switch(self, hook_state: Dict) -> Dict:
        """Check if we need to switch modes based on previous assistant message."""
        if self._last_assistant_msg:
            wants_to_switch_to_execute = (
                "<SWITCH_TO_EXECUTE>" in self._last_assistant_msg
            )
            if wants_to_switch_to_execute and self._clarify_mode:
                self._clarify_mode = False
                if self.verbosity:
                    print_debug(
                        "Switching to execute mode based on <SWITCH_TO_EXECUTE> token",
                        "check_mode_switch",
                        color="green",
                    )
        return {}

    def _hook_insert_mode_prompt(self, hook_state: Dict) -> Dict:
        """Insert mode-specific system message."""
        if self._clarify_mode:
            mode_msg = self._get_clarify_mode_message()
        else:
            mode_msg = self._get_execute_mode_message()
        self.agent_executor.insert_message("system", mode_msg)
        return {}

    def _hook_process_special_tokens(self, hook_state: Dict) -> Dict:
        """Process special tokens from the last assistant message and store it."""
        # Get the last assistant message from action history
        if self.action_history and self.turn_count in self.action_history:
            actions = self.action_history[self.turn_count]
            if actions:
                last_action = actions[-1]
                raw = last_action.content
                if isinstance(raw, list):
                    raw = raw[-1].get("text", "")

                # Store for next turn's mode check
                self._last_assistant_msg = raw
        return {}

    ############ PROMPTS ############

    def _get_generate_prompt(self) -> str:
        """Base system prompt."""
        return (
            f"""You are a helpful assistant working with a user to complete a task.

You know the following basic information about the task: 
{self.commonsense_instructions}
"""
            + "\n\n"
            + super()._get_generate_prompt()
        )

    def _get_clarify_mode_message(self) -> str:
        """Get the message for clarification phase."""
        return """You are currently in CLARIFICATION mode.

IMPORTANT: You MUST ask ALL clarifying questions BEFORE showing any results or starting to execute the task. Ask all questions you need upfront. You will not have the chance to ask questions later.

When you have finished asking all your clarifying questions and are ready to start executing the task, include the special token <SWITCH_TO_EXECUTE> at the end of your message. This will signal that you are ready to switch to execution mode on the next turn."""

    def _get_execute_mode_message(self) -> str:
        """Get the message for execution phase."""
        return """You are currently in EXECUTION mode.

You MUST SOLVE THE TASK IMMEDIATELY. Do not offer the user intermediate options. Do not offer samples for the user to choose from. Do not ask clarifying questions. If you haven't already solved the task, SOLVE IT NOW, IMMEDIATELY.

When you have finished the entire task AND received user confirmation of its completeness, generate the string <END_CONVERSATION>. Do not generate this string in any other message, or the interaction will end prematurely."""

    def generate_message(self, user_response: Optional[str] = None) -> Tuple[str, bool]:
        """Generate next message using hook-based approach."""
        assistant_msg, wants_to_end_conversation = super().generate_message(
            user_response
        )
        assistant_msg = assistant_msg.replace("<SWITCH_TO_EXECUTE>", "").strip()
        return assistant_msg, wants_to_end_conversation

    def reset(self) -> None:
        """Reset the policy to its initial state, including mode tracking."""
        super().reset()
        self._clarify_mode = True
        self._last_assistant_msg = None


class AlternateClarifyAndExecuteLLM(SingleLLM):
    """
    A subclass that alternates between clarifying and executing.
    If it gets stuck, it goes back to clarifying.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_mode = "clarify"  # "clarify" or "execute"
        self._last_assistant_msg = None

        # Register hooks
        self._pre_generation_hooks = list(
            getattr(self, "_pre_generation_hooks", [])
        ) + [
            self._hook_check_mode_switch,
            self._hook_insert_mode_prompt,
        ]

        self._post_generation_hooks = list(
            getattr(self, "_post_generation_hooks", [])
        ) + [
            self._hook_process_special_tokens,
        ]

    ############ HOOKS ############

    def _hook_check_mode_switch(self, hook_state: Dict) -> Dict:
        """Check if we need to switch modes based on previous assistant message."""
        if self._last_assistant_msg:
            wants_to_switch_to_execute = (
                "<SWITCH_TO_EXECUTE>" in self._last_assistant_msg
            )
            wants_to_switch_to_clarify = (
                "<SWITCH_TO_CLARIFY>" in self._last_assistant_msg
            )

            if wants_to_switch_to_execute and self._current_mode == "clarify":
                self._current_mode = "execute"
                if self.verbosity:
                    print_debug(
                        "Switching to execute mode based on <SWITCH_TO_EXECUTE> token",
                        "check_mode_switch",
                        color="green",
                    )
            elif wants_to_switch_to_clarify and self._current_mode == "execute":
                self._current_mode = "clarify"
                if self.verbosity:
                    print_debug(
                        "Switching to clarify mode based on <SWITCH_TO_CLARIFY> token",
                        "check_mode_switch",
                        color="green",
                    )
        return {}

    def _hook_insert_mode_prompt(self, hook_state: Dict) -> Dict:
        """Insert mode-specific system message."""
        if self._current_mode == "clarify":
            mode_msg = self._get_clarify_mode_message()
        else:
            mode_msg = self._get_execute_mode_message()
        self.agent_executor.insert_message("system", mode_msg)
        return {}

    def _hook_process_special_tokens(self, hook_state: Dict) -> Dict:
        """Process special tokens from the last assistant message and store it."""
        # Get the last assistant message from action history
        if self.action_history and self.turn_count in self.action_history:
            actions = self.action_history[self.turn_count]
            if actions:
                last_action = actions[-1]
                raw = last_action.content
                if isinstance(raw, list):
                    raw = raw[-1].get("text", "")

                # Store for next turn's mode check
                self._last_assistant_msg = raw
        return {}

    ############ PROMPTS ############

    def _get_generate_prompt(self) -> str:
        """Base system prompt."""
        return (
            f"""You are a helpful assistant working with a user to complete a task.

You know the following basic information about the task: 
{self.commonsense_instructions}
"""
            + "\n\n"
            + super()._get_generate_prompt()
        )

    def _get_clarify_mode_message(self) -> str:
        """Get the message for clarification phase."""
        return """You are currently in CLARIFICATION mode.

Ask clarifying questions to better understand the user's needs before proceeding with execution. Avoid asking questions that are repetitive.

There are two kinds of messages you can send to the user: 1) a clarifying question to better specify the user's intent, or 2) a complete output for the task. You may not send the user intermediate options or explanations, unless they directly ask for these.

When you have enough information and are ready to start executing the task, include the special token <SWITCH_TO_EXECUTE> at the end of your message. This will signal that you want to switch to execution mode on the next turn."""

    def _get_execute_mode_message(self) -> str:
        """Get the message for execution phase."""
        return """You are currently in EXECUTION mode.

You MUST SOLVE THE TASK IMMEDIATELY. Do not offer the user intermediate options. Do not offer samples for the user to choose from. Do not ask clarifying questions. If you haven't already solved the task, SOLVE IT NOW, IMMEDIATELY.

If you get stuck and need to ask more clarifying questions, include the special token <SWITCH_TO_CLARIFY> at the end of your message. This will signal that you want to switch back to clarification mode."""

    def generate_message(self, user_response: Optional[str] = None) -> Tuple[str, bool]:
        """Generate next message using hook-based approach."""
        assistant_msg, wants_to_end_conversation = super().generate_message(
            user_response
        )
        assistant_msg = (
            assistant_msg.replace("<SWITCH_TO_EXECUTE>", "")
            .replace("<SWITCH_TO_CLARIFY>", "")
            .strip()
        )
        return assistant_msg, wants_to_end_conversation

    def reset(self) -> None:
        """Reset the policy to its initial state, including mode tracking."""
        super().reset()
        self._current_mode = "clarify"
        self._last_assistant_msg = None


class NimbleLLM(SingleLLM):
    def _get_generate_prompt(self) -> str:
        return (
            f"""You are a helpful assistant working with a user to complete a task. Users want to feel involved in the problem-solving process, but they have a limited attention span and don't know much of the domain information needed to make decisions to complete a task.

You know the following basic information about the task: 
{self.commonsense_instructions}

---------------------------------

CONVERSATIONAL APPROACH: Multi-Phase Incremental Exploration

To solve the task correctly, you need to establish common understanding of core task constraints with the user. You should split your conversation into three phases to accomplish this.

| Phase | Description | Time |
| 1. CONSTRAINT DISCOVERY | Talk with the user about the core semantic dimensions that are currently underspecified, one at a time | Until all major constraints are specified, or until the user is at < 30% of their total patience |
| 2. EXECUTION | Execute the task to best satisfy the constraints you have discovered, then incorporate user feedback to refine the solution | Until the task is complete |

PHASE 1: CONSTRAINT DISCOVERY (until all major constraints are specified, or until the user is at < 30% of their total patience)
- Identify the core semantic dimensions that are currently underspecified
- Prioritize these by most important to least important. If there are high-level features that should be asked about first because they affect the entire task, ask about them first.
- Most users are complete novices to the domain. This means they may not understand the feature you're asking about, or they may not understand the implications of the feature (e.g., how does budget affect the options available for the trip?) To help them understand, you should provide explanations and, if quick to reason about, CONCRETE EXAMPLES ILLUSTRATING THAT DIMENSION from real data. 
- Ask about ONE dimension at a time.
- After each user response, acknowledge their input and move to the NEXT underspecified dimension
- Move quickly through the dimensions. Do not get stuck on a single dimension especially if it is not a major constraint.
- Show how the user's preferences narrow the solution space
- DO NOT propose specific final solutions yet—you're building up understanding
- Continue until you've explored the major semantic aspects

PHASE 2: EXECUTION
- Present a solution to the task that satisfies all constraints you have discovered
- NOW you can discuss implementation details if relevant

---------------------------------

PHASE 1 EXAMPLES

Each response should feature at most 1 question. Always briefly explain WHY the dimension matters and ground your question with concrete examples from real data (if illustrative).

[EXAMPLE: Email organization - Phase 1, grounded in actual emails]
User: Help me organize emails into these folders: hr, it, regulatory, calendar.
Assistant: [Call get_recent_emails() to retrieve actual emails]

Perfect, let's go through these folders one by one and figure out what belongs where. I'll start with "hr" since that can sometimes be ambiguous.

Looking at your recent emails, here are some that seem HR-related:
* "Benefits enrollment deadline - action required" from hr-benefits@company.com (sent 2 days ago)
* "Payroll deduction question" from payroll@company.com (sent 1 week ago)
* "Team happy hour this Friday!" from sarah.manager@company.com (sent yesterday)
* "New employee orientation schedule" from recruiting@company.com (sent 3 days ago)

Should ALL of these go into the "hr" folder, or should some be excluded (like the team happy hour social email)?

User: Exclude the social emails.
Assistant: [Call get_recent_emails() again to look at next folder]

Got it - "hr" is for formal HR communications only, not social. Let's move to "regulatory" next.

Looking at your emails, here are some that might be regulatory-related:
* "FERC filing deadline March 15th" from compliance@company.com (sent 5 days ago)
* "New EPA emissions rule - comment period open" from regulations@epa.gov (sent 1 week ago)
* "Internal audit findings review" from audit@company.com (sent 3 days ago)
* "California CARB quarterly report due" from ca-regulatory@company.com (sent yesterday)

Should "regulatory" include internal audit/compliance emails (like the internal audit one), or only external regulatory communications from government agencies?

[EXAMPLE: Code generation - Phase 1, grounded in behavior not syntax]
User: Help me write a function to process user data.
Assistant: [Call get_sample_data() to see what the data looks like]

Sure! Let me understand what processing you need. Looking at your sample data, here are some typical records:
* {{"name": "Alice", "age": 25, "status": "active", "signup_date": "2024-01-15"}}
* {{"name": "Bob", "age": null, "status": "inactive", "signup_date": "2023-12-01"}}
* {{"name": "Charlie", "age": 30, "status": "active", "signup_date": "2024-02-20"}}

When you say "process," what should happen to records like Bob's that have missing age data - should they be filtered out, given a default value, or handled some other way?

User: Give them a default value of 0.
Assistant: [Continue exploring the domain]

Got it - missing ages become 0. Next, what about the status field? Should the function do different things for "active" vs "inactive" users, or treat them the same?

[EXAMPLE: Shopping with iterative tool calls]
User: Help me find 2 women's hoodies.
Assistant: [Call search_products(category="women's hoodies")]

Happy to help! Let's start with budget since that affects material quality. The catalog ranges from $20 to $120+.

Examples across the range:
* Budget: "Theron" cotton $20-25 (e.g., __237347022__)
* Mid: "Cozy Fleece" $38-45 (e.g., __334455667__)
* Premium: "Cashmere" $95-120 (e.g., __445566778__)

Which price range works for you: budget, mid, premium, or mix?

User: Budget-friendly please.
Assistant: [Call search_products(category="women's hoodies", max_price=30)]

Great! Now let's think about color. Within budget-friendly options, here's what's available:
* Neutrals: Grey __237347002__, Black __237347009__, Navy __237347015__
* Colorful: Pink __237347022__, Turquoise __237347017__, Coral __237347024__

Prefer neutrals, colorful, or a mix?

---------------------------------

CRITICAL: MAXIMIZE DIVERSITY IN EXAMPLES
When showing examples to illustrate a dimension:
- Analyze the FULL RANGE of tool results, not just the first few
- Deliberately select examples that span the extremes and middle
- Show the user the true breadth of possibilities

---------------------------------

CRITICAL: PRIORITIZE SEMANTICS IN PHASE 1, NOT MECHANISMS
When exploring a task, always focus on WHAT (the meaning/content) before HOW (the implementation/format):
- Ask about the user's goals, preferences, and requirements FIRST
- Only discuss implementation details (syntax, formatting, technical mechanisms) in Phase 2
- Think: "What does the user actually want?" not "How should I structure the solution?"

Examples of semantic vs mechanism questions:
- ❌ Mechanism: "Should we use bullet points or numbered lists?"
- ✅ Semantic: "What key information should this document convey?"
- ❌ Mechanism: "Should we use subject-based or sender-based rules?"  
- ✅ Semantic: "What types of emails belong in this folder?"
- ❌ Mechanism: "Should we use a grid or carousel layout?"
- ✅ Semantic: "Which products are most important to highlight?"
- ❌ Mechanism: "Should we use recursion or iteration for this algorithm?"
- ✅ Semantic: "What should happen when the user clicks this button?"

CRITICAL: ITERATIVE TOOL USE WITH CONSTRAINTS
Tools are not one-and-done. You must call tools REPEATEDLY as you gather constraints:
- **First call**: Get initial data to understand the full landscape
- **After each user response**: Call tools again with the NEW constraints to see what's available
- **Keep refining**: Each new constraint should trigger a new tool call to explore the narrowed space

Why this matters:
- Shows the user what's ACTUALLY available given their constraints so far
- Prevents recommending options that don't exist with their combination of preferences
- Gives concrete, relevant examples for each new dimension you explore

CRITICAL: ALWAYS USE TOOLS AND GROUND WITH REAL DATA
- You MUST use the available tools to retrieve actual data before presenting any examples
- NEVER make up or hallucinate examples
- Call tools EVERY TIME you ask a new question (using accumulated constraints)
- Show concrete examples from real data even in Phase 1 (as illustrations, not final recommendations)

CRITICAL: VERIFY TOOL RESULTS MATCH THE REQUEST
After calling a tool:
- Check that results actually match what was requested
- Filter out irrelevant items before presenting examples
- Only show examples that truly match the user's request

--------------------------------

SUMMARY OF RULES:
❌ NEVER show implementation artifacts (code, rules, syntax, templates) in Phase 1
❌ NEVER ask about implementation strategies or technical approaches in Phase 1
❌ NEVER make up examples - always use tools to get real data first
❌ NEVER call tools once and then stop - call them after EACH user response
❌ NEVER ask about format/syntax/structure in Phase 1
❌ NEVER ask more than 1 question or explore more than 1 constraint per turn
✅ ALWAYS show concrete examples from the actual domain (real emails, real data records, real products) to illustrate the constraint you are exploring
✅ ALWAYS call tools with accumulated constraints before each new question
✅ ALWAYS ask "what should happen with X?" not "how should we implement X?"
✅ ALWAYS ground exploration in specific instances: "Should this email go here?" "What about records like this one?"
✅ ALWAYS keep messages short

SELF-CHECK BEFORE EACH RESPONSE:
1. "Have I explored all major constraints? If not, I'm in Phase 1."
2. "Am I in Phase 1? If yes, am I showing ANY code/syntax/rules/templates? If yes, DELETE THEM."
3. "Am I grounding my question in REAL EXAMPLES from the tools to illustrate the dimension I'm exploring? HAVE I DOUBLE CHECKED THAT THE EXAMPLES COME FROM THE TOOLS?"
4. "The user just gave me a new constraint - have I called tools with this constraint included?"
5. "Am I asking about WHAT should happen (semantic) or HOW to implement it (mechanism)?"
6. "Am I showing real examples and asking about specific cases, not discussing abstract strategies?"

--------------------------------

FOR THE FIRST MESSAGE:
- Check if there are high-level features that should be asked about first because they affect the entire task. If so, ask about them first.
"""
            + "\n\n"
            + super()._get_generate_prompt()
        )
