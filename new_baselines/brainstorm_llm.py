from typing import Dict, List, Tuple, Union, Optional
import sys


from new_baselines.single_llm import SingleLLM
from utils.misc import (
    add_section,
    Stopwatch,
    print_debug,
)
from utils.model import LangChainModel, is_openai_model, is_anthropic_model


class BrainstormLLM(SingleLLM):
    """
    On the first call to this model, we generate a plan for the conversation, which is injected
    into the system prompt.
    """

    def generate_plan(self, initial_user_msg: str) -> str:
        """
        Generate a plan for the conversation.
        """
        return self._call_agent_executor(
            *[
                (
                    "system",
                    f"""You are a helpful assistant working with a user to complete a task.

You know the following basic information about the task: 
{self.initial_specification}
User request: {initial_user_msg}

Note that the request is underspecified: you should work with the user to better understand their context, preferences, and constraints. You can do this through questions and getting user feedback (e.g. by asking them to react to a proposed option). 

To facilitate this interaction, we are going to break the task down into (mostly independent) subtasks, and then work with the user one subtask at a time. This will help us understand user needs in bite-sized pieces, and it prevents the user from being cognitively overloaded by too many choices at once. We want independence between subtasks to help the user avoid context switching.

Each subtask should include: a list of what still requires additional specification from the user, a plan for how to solve the subtask, a plan for how to observe user behavior to better understand their preferences, and a stop condition for when the subtask is complete.

(Example)
<plan>
Goal: plan a trip to Paris, specifying the flight and attractions.
Context from available tools: we have a flights database with columns describing departure times and prices, and an attractions database with prices and types.

Subgoal 1: Find flight
- Needs additional specification: travel dates, flight budget
- Plan: ask questions, filter flight database. If < 3 options, show all options, otherwise, show top 3 cheapest options.
- Revealed preferences: user selects flight from options
- Stop condition: user confirms flight

Subgoal 2: Find attractions
- Needs additional specification: attraction types, budget
- Plan: ask questions, filter attraction database. If < 3 options, show all options, otherwise, show top 3 cheapest options.
- Revealed preferences: user selects attractions from options
- Stop condition: user confirms attractions

The subgoals should be mostly independent, but may need to go back to older subgoals if, e.g., there is one overall budget for both flights and attractions.
</plan>

Notice that by separating flights from attractions, we can avoid the user needing to answer questions about both at once. We want to separate the specification steps as much as possible to help the user avoid context switching. Here is an example of a bad plan:
<plan>
Subgoal 1: Get user specifications
Subgoal 2: Find flight and attractions
</plan>
This is a bad plan because the user needs to answer questions about both flights and attractions at once. The subgoals are also not independent (2 depends on 1 directly); dependent subgoals should be merged.

You MUST use the tools before generating the plan to understand the task space better.

This summary will not be read by the user: it is only for you to reference. Write in the third person. Surround your response with <plan> tags.
""",
                )
            ],
            persist_state=False,
            min_react_steps=3,
        )[0]

    def generate_message(self, user_response: Optional[str] = None) -> Tuple[str, bool]:
        """
        Generate the next message in the conversation.

        Returns:
            str: The next message in the conversation
        """
        # If this is the first turn, generate the plan, and then prepend the generate prompt with the plan
        if not self.has_seen_system_prompt:
            plan = self.generate_plan(user_response)

            if self.verbosity:
                print_debug(f"Generated plan: {plan}", "generate_plan", color="orange")

            system_msg = self._get_generate_prompt(plan)
            prompt = [("system", system_msg), ("user", user_response)]
            self.has_seen_system_prompt = True
        else:
            prompt = [("user", user_response)]

        if self.verbosity == 2:
            print_debug(
                f"Generating message with prompt:\n{prompt}",
                "generate_message",
                color="blue",
            )

        # Call generate
        raw, _, _ = self._call_agent_executor(*prompt)
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


######### SUBCLASSES #########


class BreakItDownLLM(BrainstormLLM):
    def _get_generate_prompt(self, plan: str) -> str:
        return f"""You are a helpful assistant working with a user to complete a task.

Here is the plan for the conversation:
{plan}

Work with the user, one subtask at a time. Carefully manage the user's cognitive load: try not to ask more than 2 questions in a single message. Do not send > than 2 messages in a row which only ask questions. {self.msg_fmt_instructions} When you have finished the entire task and received user confirmation, generate the string <END_CONVERSATION>. To show a user a message, do not make tool calls in that message.

If this is the first message, you should provide a brief explanation of the plan. (Do not repeat anything else in this message.) 
"""
