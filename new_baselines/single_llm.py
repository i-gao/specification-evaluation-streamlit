from typing import Dict, List, Tuple, Union, Optional
import sys


from new_baselines.policy import InteractionPolicy
from utils.misc import (
    add_section,
    Stopwatch,
    print_debug,
)
from utils.model import LangChainModel, is_openai_model, is_anthropic_model

import time
class SingleLLM(InteractionPolicy):
    def __init__(
        self,
        *args,
        model_name: str = "gpt-4o-mini",
        model_kwargs: dict = {},
        max_react_steps: int = 25,
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
        self._model_lock = False

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
        if self._model_lock:
            print("Model is locked; waiting for it to unlock")
            time.sleep(10) # this is a hack
        
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
        raise NotImplementedError("Subclass must implement this method")

    def _get_predict_prompt(self) -> str:
        """
        Add a system message demanding a current prediction.
        """
        return add_section(
            "Generate the complete final output for the task",
            f"Based on the conversation history above, generate the best possible solution for the task. YOU MUST GENERATE THE SOLUTION NOW WITH NO OTHER TEXT.\n\nYou must follow this expected format for the solution:\n{self.prediction_fmt_instructions}",
        )

    def generate_message(self, user_response: Optional[str] = None) -> Tuple[str, bool]:
        """
        Generate the next message in the conversation.

        Returns:
            str: The next message in the conversation
        """
        print("Calling generate_message; user_response", user_response)

        # If this is the first turn, prepend the generate prompt
        if not self.has_seen_system_prompt:
            system_msg = self._get_generate_prompt()
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


class RawLLM(SingleLLM):
    def _get_generate_prompt(self) -> str:
        return f"""You are a helpful assistant working with a user to complete a task.

You know the following basic information about the task: 
{self.initial_specification}

Work with the user. {self.msg_fmt_instructions} When you have finished the entire task and received user confirmation, generate the string <END_CONVERSATION>. To show a user a message, do not make tool calls in that message.
"""

class ClarifyLLM(SingleLLM):
    def _get_generate_prompt(self) -> str:
        """
        Get the system message for the language model.
        """
        return f"""You are a helpful assistant working with a user to complete a task. Often, users are unclear about their intent or context. Not knowing this information can make it difficult to provide a maximally helpful answer. Therefore, before executing the task (and possibly throughout the task), you should ask questions to clarify any ambiguities about the task with the user. However, avoid asking questions that are repetitive.

You know the following basic information about the task: 
{self.initial_specification}
Use the tools available to you to ground your work in the actual features of the task space. If there is a CSV of options, your work must use that CSV.

There are two kinds of messages you can send to the user: 1) a clarifying question to better specify the user's intent, or 2) a complete output for the task. You may not send the user intermediate options or explanations, unless they directly ask for these.

Work with the user. {self.msg_fmt_instructions} When you have finished the entire task and received user confirmation, generate the string <END_CONVERSATION>. 

Remember to ask questions! You MUST ask clarifying questions on your first turn, BEFORE showing any results. To show a user a message, do not make tool calls in that message.
"""


class ExecutionLLM(SingleLLM):
    def _get_generate_prompt(self) -> str:
        return f"""You are a helpful assistant working with a user to complete a task.

You know the following basic information about the task: 
{self.initial_specification}
Use the tools available to you to ground your work in the actual features of the task space. If there is a CSV of options, your work must use that CSV.

You MUST SOLVE THE TASK IMMEDIATELY. Do not offer the user intermediate options. Do not offer samples for the user to choose from. Do not ask clarifying questions. If you haven't already solved the task, SOLVE IT NOW, IMMEDIATELY.

Output the solution to the task and include it in your final message. {self.msg_fmt_instructions} DO NOT, UNDER ANY CIRCUMSTANCES, ASK QUESTIONS OR SHOW MULTIPLE OPTIONS TO CHOOSE FROM. YOU MAY ONLY SHOW A SINGLE FINAL RECOMMENDATION IN ANY GIVEN MESSAGE. When you have finished the entire task and received user confirmation, generate the string <END_CONVERSATION>. To show a user a message, do not make tool calls in that message.
"""
