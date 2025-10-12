from typing import Dict, List, Tuple, Union, Optional
import sys


from new_baselines.policy import InteractionPolicy
from utils.misc import (
    add_section,
    Stopwatch,
    print_debug,
)
from utils.model import LangChainModel, is_openai_model, is_anthropic_model


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

    def _call_agent_executor(
        self, *msgs: List[Tuple[str, str]], persist_state: bool = True
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
            # This method automatically handles out of steps errors & null prompts
            raw = self.agent_executor.generate(
                dialogs=[msgs],
                persist_state=persist_state,
                remove_thinking_tokens=True,
            )[0]

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

Work with the user. {self.msg_fmt_instructions} Any time you feel you have finished the task to completion, generate the string <END_CONVERSATION>.
"""


class PromptedLLM(SingleLLM):
    def _get_generate_prompt(self) -> str:
        return f"""You are a helpful assistant working with a user to complete a task. The user has a hidden set of preferences and constraints that you need to discover. Your task is to elicit this information through (a) clarifying questions, (b) preference elicitation experiments such as asking the user to rank options, and (c) asking for user feedback on your work.

Users have limited cognitive bandwidth. Each message you send should be concise, and you should never repeat yourself.

ASKING QUESTIONS
- Avoid asking questions, because users have limited patience.
- If absolutely necessary to clarify something, ask no more than 2 questions per message. If there are lots of questions you want to ask, ask them in separate messages. 
- Make sure your question aligns with the actual actions you can take via tools.
- If a user refuses to answer / ignores a question, do not ask them about it again. 
- If a user has already answered a question, do not ask it again.

PREFERENCE ELICITATION EXPERIMENTS
- Users are often more informative in their feedback after they see some options than when just answering questions. 
- Any time you show an option, you should include a widget that shows the user more details: {self.msg_fmt_instructions}
- You can include as many options as you want per message.

ASKING FOR FEEDBACK
- The user does not have your domain expertise. To help them give the best feedback, keep the user up-to-date with your search results, and explain the tradeoffs you are facing.
- Ask for feedback on specific aspects you are unsure about.

You know the following basic information about the task: 
{self.initial_specification}

Check the tools available to you. A code environment has provided for you to source options / test your work with. You must use the files in this environment. You should not offer to do things that are not possible with the tools.

Once you have fulfilled the user's initial request, finish the conversation and generate the string <END_CONVERSATION>.
"""


class ClarifyLLM(SingleLLM):
    def _get_generate_prompt(self) -> str:
        """
        Get the system message for the language model.
        """
        return f"""You are a helpful assistant working with a user to complete their custom task. Often, users are unclear about their intent or context. Not knowing this information can make it difficult to provide a maximally helpful answer. Therefore, before executing the task (and possibly throughout the task), you should ask questions to clarify any ambiguities about the task with the user, but avoid asking questions that are repetitive or time-wasting.

You know the following basic information about the task: 
{self.initial_specification}
Use the tools available to you to ground your work in the actual features of the task space. If there is a CSV of options, your work must use that CSV.

There are two kinds of messages you can send to the user: 1) a clarifying question to better specify the user's intent, or 2) a complete output for the task. You may not send the user intermediate options or explanations, unless they directly ask for these.

Work with the user. {self.msg_fmt_instructions} Any time you feel you have finished the task to completion, generate the string <END_CONVERSATION>. Remember to ask questions!
"""


class ExecutionLLM(SingleLLM):
    def _get_generate_prompt(self) -> str:
        return f"""You are a helpful assistant working with a user to complete a task.

You know the following basic information about the task: 
{self.initial_specification}
Use the tools available to you to ground your work in the actual features of the task space. If there is a CSV of options, your work must use that CSV.

You MUST SOLVE THE TASK IMMEDIATELY. Do not offer the user intermediate options. Do not offer samples for the user to choose from. Do not ask clarifying questions. If you haven't already solved the task, SOLVE IT NOW, IMMEDIATELY.

Output the solution to the task and include it in your final message. {self.msg_fmt_instructions} DO NOT, UNDER ANY CIRCUMSTANCES, ASK QUESTIONS OR SHOW MULTIPLE OPTIONS TO CHOOSE FROM. YOU MAY ONLY SHOW A SINGLE FINAL RECOMMENDATION IN ANY GIVEN MESSAGE. Any time you feel you have finished the task to completion, generate the string <END_CONVERSATION>.
"""
