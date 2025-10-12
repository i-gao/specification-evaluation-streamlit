from typing import List, Tuple, Union, Dict, Optional
import os
import json
import re
import hashlib
import uuid
from PIL import Image
import base64
from langchain_core.messages import BaseMessage
from functools import lru_cache


def encode_image_as_user_msg(
    image: Image.Image = None,
    image_path: str = None,
    extension: str = "png",
    caption: str = None,
    model_name: str = "gpt-4o-mini",
) -> str:
    """
    Encode an image to a base64 string.
    """
    if image_path is None:
        assert image is not None
        need_to_remove = True
        image.save(f"temp.{extension}")
        image_path = f"temp.{extension}"
    else:
        need_to_remove = False

    with open(image_path, "rb") as image_file:
        bytes = base64.b64encode(image_file.read()).decode("utf-8")
        content = [] if caption is None else [{"type": "text", "text": caption}]
        msg = {
            "role": "user",
            "content": content
            + [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{extension};base64,{bytes}"},
                },
            ],
        }

    if need_to_remove:
        os.remove(image_path)
    return msg


class Model:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def fmt_as_dialog(
        self,
        prompts: List[Union[str, Image.Image]] = None,
        dialogs: List[List[Tuple[str, Union[str, Image.Image]]]] = None,
    ):
        """
        Args:
            prompts: A list of prompts. The list dimension is over (B,)
                e.g. ["What is the capital of France?"] -> [[{"role": "user", "content": "What is the capital of France?"}]]
            dialogs: A list of [(role, content)] pairs. The list dimension is over (B, D)
                e.g. [[("user", "What is the capital of France?"), ("assistant", "Paris.")]] -> [[{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}]]
        Returns:
            Formatted dialogs: list of list of dictionaries (B, D) where each dictionary has keys "role" and "content"
        """
        assert (prompts is None) ^ (dialogs is None), (
            "Exactly one of prompts or dialogs must be provided."
        )
        out = []
        if prompts is not None:
            for prompt in prompts:
                if prompt is None:
                    continue
                if isinstance(prompt, Image.Image):
                    out.append(encode_image_as_user_msg(image=prompt))
                else:
                    out.append([{"role": "user", "content": prompt}])
        else:
            for dialog in dialogs:
                o = []
                for role, content in dialog:
                    if content is None:
                        continue
                    if isinstance(content, Image.Image):
                        o.append(encode_image_as_user_msg(image=content))
                    else:
                        o.append({"role": role, "content": content})
                out.append(o)
        return out

    def generate(
        self,
        *,
        prompts: List[str] = None,
        dialogs: List[List[Tuple[str, str]]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Args:
            prompts: A list of prompts. The list dimension is over (B,)
                e.g. ["What is the capital of France?"]
            dialogs: A list of [(role, content)] pairs. The list dimension is over (B, D)
                e.g. [[("user", "What is the capital of France?"), ("assistant", "Paris.")]]
        Returns:
            A list of completions. The list dimension is over (B,)
            e.g. ["Paris."]
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


def get_reasoning_effort_kwargs(
    model_name: str, reasoning_effort: str, max_tokens=64000
) -> dict:
    """
    Get the kwargs for the reasoning effort for a model.
    """
    if any(model_name.startswith(m) for m in ["gpt-5", "o3", "o4", "o1"]):
        return {"reasoning_effort": reasoning_effort}
    elif is_anthropic_model(model_name):
        if reasoning_effort == "minimal":
            return {"thinking": {"type": "disabled"}}
        effort_to_tokens = {
            "low": 0.2 * max_tokens,
            "medium": 0.4 * max_tokens,
            "high": 0.8 * max_tokens,
        }
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": max(
                    1024, int(effort_to_tokens.get(reasoning_effort, max_tokens))
                ),
            }
        }
    print("Warning: model not supported for reasoning effort: ", model_name)
    return {}


@lru_cache(maxsize=10)
def is_openai_model(model_name: str) -> bool:
    """
    Check if a model is an OpenAI model.
    Args:
        model_name: The name of the model to check.
    Returns:
        True if the model is an OpenAI model, False otherwise.
    """
    from openai import OpenAI

    client = OpenAI()
    return model_name in [m.id for m in client.models.list()]


@lru_cache(maxsize=10)
def is_anthropic_model(model_name: str) -> bool:
    """
    Check if a model is an Anthropic model.
    Args:
        model_name: The name of the model to check.
    Returns:
        True if the model is an Anthropic model, False otherwise.
    """
    from anthropic import Anthropic

    client = Anthropic()
    return model_name in [m.id for m in client.models.list()]


def init_model(model_name: str, **kwargs):
    """
    Initialize a model.
    Args:
        model_name: The name of the model to initialize.
        **kwargs: Additional keyword arguments to pass to the model.
    Returns:
        A model.
    """
    try:
        if is_openai_model(model_name):
            return OpenAIModel(model_name, **kwargs)

        elif is_anthropic_model(model_name):
            return AnthropicModel(model_name, **kwargs)

        else:
            return TransformersModel(model_name, **kwargs)
    except Exception as e:
        raise ValueError(f"Unknown model: {model_name}")


def init_langchain_model(model_name: str, **kwargs):
    """
    Initialize a LangChain chat model.
    Args:
        model_name: The name of the model to initialize.
        **kwargs: Additional keyword arguments to pass to the model.
    Returns:
        A LangChain model.
    """
    from langchain.chat_models import init_chat_model

    provider = (
        "openai"
        if is_openai_model(model_name)
        else "anthropic"
        if is_anthropic_model(model_name)
        else "huggingface"
    )
    if "reasoning_effort" in kwargs:
        kwargs.update(
            get_reasoning_effort_kwargs(
                model_name,
                kwargs.pop("reasoning_effort"),
                max_tokens=kwargs.get("max_tokens", 64000),
            )
        )
    if provider == "huggingface":
        # init_chat_model doesn't support HF correctly: https://github.com/langchain-ai/langchain/issues/28226
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            pipeline,
            BitsAndBytesConfig,
        )
        from langchain_huggingface import HuggingFacePipeline
        from utils.tools import ChatHuggingFaceTools

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens", 8000)
        if "quantization" in kwargs:
            q = kwargs.pop("quantization")
            kwargs["model_kwargs"] = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=(q == 8),
                    load_in_4bit=(q == 4),
                )
            }

        print("Initializing pipeline with kwargs: ", kwargs)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            **kwargs,
        )
        hf = HuggingFacePipeline(pipeline=pipe)
        return ChatHuggingFaceTools(llm=hf, verbose=True)
    else:
        return init_chat_model(
            model_name,
            model_provider=provider,
            max_tokens=(
                None if provider != "anthropic" else 64000
            ),  # anthropic needs max tokens
            **kwargs,
        )

def view_messages_hook(state: dict) -> dict:
    """
    View the messages in the state.
    """
    print("==================")
    print(state["messages"])
    print("==================")
    return state


class LangChainModel(Model):
    from langchain_core.tools import StructuredTool

    def __init__(
        self,
        model_name: str,
        tools: List[StructuredTool] = None,
        verbosity: int = 0,
        max_react_steps: int = 25,
        min_react_steps: int = 1,
        prompt_cache: bool = True,
        multiturn_memory: bool = False,
        summarize_state_after: int = None,
        out_of_steps_msg: str = None,
        list_tools_in_prompt: bool = False,
        thinking_tokens: Tuple[str, str] = ("<think>", "</think>"),
        add_thinking_tag: bool = True,
        **kwargs,
    ):
        super().__init__(model_name)
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
        from langchain.tools.render import render_text_description
        from langchain_core.messages import SystemMessage

        assert max_react_steps > 0, "max_react_steps must be positive"
        assert min_react_steps <= max_react_steps, (
            "min_react_steps must be less than or equal to max_react_steps"
        )

        self.model_no_tools = init_langchain_model(model_name, **kwargs)

        if is_anthropic_model(model_name):
            bind_tools_kwargs = {
                "tool_choice": {
                    "type": "auto",
                    "disable_parallel_tool_use": True,
                }
            }
        elif is_openai_model(model_name):
            bind_tools_kwargs = {
                "parallel_tool_calls": False,
            }
        else:
            # hf model
            if not list_tools_in_prompt:
                print("Warning: list_tools_in_prompt is False for huggingface model")
            bind_tools_kwargs = {}

            # multiturn memory must be true if prompt_cache is true
            if prompt_cache:
                assert multiturn_memory, "For HF models, multiturn_memory must be true if prompt_cache is true"

        # setup ReAct style prompt
        if list_tools_in_prompt:
            from utils.tools import post_model_parse_tools_hook

            system_tools_msg = """You have access to the following tools:\n{tools}\n\nTo call a tool, wrap a JSON with the tool name and arguments in <tool_call> tags. For example, <tool_call>{{"name": "add", "arguments": {{"a": 1, "b": 2}}}}</tool_call>. Only choose from the following tools: {tool_names}""".format(
                tools=render_text_description(tools),
                tool_names=", ".join([t.name for t in tools]),
            )
            extra_kwargs = {
                "post_model_hook": post_model_parse_tools_hook,
                "prompt": SystemMessage(content=system_tools_msg),
            }
        else:
            extra_kwargs = {}

        self.graph = create_react_agent(
            (
                self.model_no_tools.bind_tools(tools, **bind_tools_kwargs)
                if tools
                else self.model_no_tools
            ),
            tools=tools if tools is not None else [],
            debug=(verbosity == 2),
            checkpointer=MemorySaver(),  # need to keep this for state fetching
            # pre_model_hook=view_messages_hook,
            **extra_kwargs,
        )
        self._raw_state = []
        self._max_react_steps = max_react_steps
        self._min_react_steps = min_react_steps
        self._multiturn_memory = multiturn_memory
        self._summarize_state_after = summarize_state_after
        self._out_of_steps_msg = out_of_steps_msg
        self._thinking_tokens = thinking_tokens
        self._add_thinking_tag = add_thinking_tag

        # whether to change system-only prompts to user-only prompts
        self._is_anthropic = is_anthropic_model(model_name)
        self._is_openai = is_openai_model(model_name)
        self._prompt_cache = prompt_cache
        self.thread_id = str(uuid.uuid4())

    def fmt_as_dialog(
        self, prompts: List[str] = None, dialogs: List[List[Tuple[str, str]]] = None
    ):
        dialogs = super().fmt_as_dialog(prompts=prompts, dialogs=dialogs)
        for dialog in dialogs:
            if (
                not self._is_openai
            ):  # both hf and anthropic require that the last msg is a human message
                if len(dialog) == 1 and dialog[0]["role"] == "system":
                    dialog[0]["role"] = "user"
            if self._prompt_cache:
                for msg in dialog:
                    if self._is_anthropic:
                        msg["cache_control"] = {"type": "ephemeral"}
                    elif self._is_openai:
                        msg["prompt_cache_key"] = self.thread_id
        return dialogs

    def generate(
        self,
        *,
        prompts: List[str] = None,
        dialogs: List[List[Tuple[str, str]]] = None,
        raw_messages: List[List[BaseMessage]] = None,
        remove_thinking_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        if prompts is not None:
            dialogs = self.fmt_as_dialog(prompts=prompts)
        elif dialogs is not None:
            dialogs = self.fmt_as_dialog(dialogs=dialogs)
        else:
            dialogs = raw_messages

        assert len(dialogs) == 1, (
            "Only one prompt / conversation at a time is supported"
        )

        if self._summarize_state_after is not None:
            if len(self.state) >= self._summarize_state_after:
                self.compress_state()

        outs = []
        for dialog in dialogs:
            outs.append(
                self._call_graph(
                    dialog,
                    **kwargs,
                )
            )

        if remove_thinking_tokens:
            pattern = (
                re.escape(self._thinking_tokens[0])
                + r".*?"
                + re.escape(self._thinking_tokens[1])
            )
            for out in outs:
                for msg in out:
                    msg.content = re.sub(pattern, "", msg.content, flags=re.DOTALL)

        return outs

    @property
    def state(self):
        """Return the state of the chain"""
        return self.graph.get_state(
            {"configurable": {"thread_id": self.thread_id}}
        ).values.get("messages", [])

    @property
    def raw_state(self):
        """
        Return the raw state of the chain (read-only)
        The raw-state only appends: no summarization, no deletion
        The exception is if multiturn_memory is False: then the raw_state
        will be [].
        """
        return self._raw_state

    def get_state(self):
        """Dump the state of the chain"""
        from langchain_core.load import dumps

        return dumps(self.state)

    def load_state(self, state: str):
        """Load the state of the chain"""
        from langchain_core.load import loads

        self.clear_state()
        self.graph.update_state(
            {"configurable": {"thread_id": self.thread_id}},
            {"messages": loads(state)},
        )

    def __len__(self):
        """Return the length of the chain of messages"""
        return len(self.state)

    def clear_state(self):
        """Clear the state of the chain"""
        self.graph.checkpointer.delete_thread(self.thread_id)

    def _call_model_no_tools(
        self, messages: List[Union[dict, BaseMessage]], persist_state: bool = True
    ) -> BaseMessage:
        """Call the model without tools. This is guaranteed to return a single message."""
        print("==================")
        print(messages)
        print("==================")
        msg = self.model_no_tools.invoke(messages)
        if persist_state:
            self.graph.update_state(
                {"configurable": {"thread_id": self.thread_id}},
                {"messages": [msg]},
            )
        return msg

    def _summarize_messages(
        self,
        messages_to_summarize: List[Union[dict, BaseMessage]],
        context_messages: List[Union[dict, BaseMessage]] = [],
        add_tag_to_summary: bool = True,
    ) -> BaseMessage:
        """Summarize the messages"""
        from langchain_core.messages import SystemMessage

        if context_messages:
            context_messages = [
                SystemMessage(
                    content="Below is a conversation between a user and an assistant."
                )
            ] + context_messages

        summary_msg = self._call_model_no_tools(
            context_messages
            + [
                SystemMessage(
                    content="Now, summarize the content of the FOLLOWING messages into a single paragraph. ONLY summarize the following messages."
                )
            ]
            + messages_to_summarize,
            persist_state=False,
        )
        if add_tag_to_summary:
            summary_msg.content = f"[SUMMARY OF {len(messages_to_summarize)} MESSAGES] {summary_msg.content}"
        return summary_msg

    def compress_state(self, compress_types: List[str] = ["Tool", "AI"]):
        """
        Compresses the chain by summarizing non-System message
        """
        for t in compress_types:
            assert t in ["Tool", "AI", "Human", "System"], "Invalid compress type"

        from langchain_core.messages import (
            ToolMessage,
            AIMessage,
            HumanMessage,
            SystemMessage,
        )

        def _is_compress_type(msg: BaseMessage) -> bool:
            if "Tool" in compress_types and isinstance(msg, ToolMessage):
                return True
            elif "AI" in compress_types and isinstance(msg, AIMessage):
                return True
            elif "Human" in compress_types and isinstance(msg, HumanMessage):
                return True
            elif "System" in compress_types and isinstance(msg, SystemMessage):
                return True
            return False

        og_len = len(self.state)

        # Scan through the state, compressing chunks of contiguous messages that match a compress type
        chunk_start = None
        new_state = []
        for i, msg in enumerate(self.state):
            if _is_compress_type(msg):
                if chunk_start is None:
                    chunk_start = i
            else:
                if chunk_start is not None:
                    # time to end the current chunk
                    new_state.append(
                        self._summarize_messages(
                            messages_to_summarize=self.state[chunk_start:i],
                            context_messages=new_state,
                        )
                    )
                    chunk_start = None
                new_state.append(msg)
        if chunk_start is not None:
            # for the last chunk
            new_state.append(
                self._summarize_messages(
                    messages_to_summarize=self.state[chunk_start:],
                    context_messages=new_state,
                )
            )

        # To reset state: clear and then set the new state
        self.clear_state()
        self.graph.update_state(
            {"configurable": {"thread_id": self.thread_id}},
            {"messages": new_state},
        )

        print(f"Compressed state from {og_len} messages to {len(self.state)} messages")

    def _call_graph(
        self,
        messages: List[Union[dict, BaseMessage]],
        persist_state: bool = True,
        end_tokens: List[str] = [],  # special tokens which terminate the chain
        num_restarts: int = 0,  # 0: no retry, 1: retry once, 2: retry twice, etc.
        max_react_steps: int = None,
        min_react_steps: int = None,
        tag_on_final_msg: bool = False,
    ):
        """
        Call the graph and return only the new messages generated via this call, EXCLUDING the passed in
                prompts (messages arg), and excluding the previous history up to this point
        This may generate multiple messages until a stop condition is met (by default: no tool calls).

        Args:
            messages: the initial messages (prompt)
            end_tokens: special tokens which terminate the chain
            num_restarts: the number of times to force the chain to continue (until the recursion limit is hit)
            max_react_steps: the maximum number of steps to take (after the initial prompt)
            min_react_steps: the minimum number of steps to take (after the initial prompt)
            tag_on_final_msg: whether to tag the final message
                there should be an automatic tag regardless of this flag in the case of a GraphRecursionError

        Note for chain length:
        We run the chain. When it stops (no tool calls), we check if we should force it to continue:
        - If an end_token is found, we stop the chain.
        - Otherwise, we check if the chain is longer than min_react_steps.
            - If it is, we check if we should force the chain to continue.
                - If max_react_steps is hit, we stop the chain.
                - If num_restarts > 0, we force the chain to continue until max_react_steps is hit.
                - If num_restarts has been exceeded, we stop the chain.
        - If the chain is shorter than min_react_steps, we force the chain to continue.

        Additionally, if tag_on_final_msg=True, we add an extra step to the chain.
        """
        from langgraph.errors import GraphRecursionError
        from langchain_core.messages import HumanMessage, AIMessage

        if max_react_steps is None:
            max_react_steps = self._max_react_steps
        if min_react_steps is None:
            min_react_steps = self._min_react_steps
        assert min_react_steps >= 1, "min_react_steps must be >= 1"
        assert max_react_steps >= min_react_steps, (
            "max_react_steps must be > min_react_steps"
        )

        cfg = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": max_react_steps
            + len(
                messages
            ),  # the recursion_limit considers the entire length of the chain, including the prompt. So if we want to allow the model to take 4 steps after a len-1 prompt, we need to set the recursion limit to 1 + 4 = 5
        }
        og_len = len(self.state)

        def _retry(remaining_retries):
            """Logic for forcing the chain to continue"""
            if any(end_token in self.state[-1].content for end_token in end_tokens):
                return

            num_react_steps = len(self.state) - og_len - len(messages)
            if num_react_steps < min_react_steps:
                # If we're under min_react_steps, force the chain to continue
                continue_chain = True
            elif remaining_retries <= 0:
                # otherwise, if we're over the min number and have exhausted all the retries, finish
                continue_chain = False
            elif cfg["recursion_limit"] - num_react_steps < 1:
                # otherwise, if we're over the min number and have hit the recursion limit, finish
                print("Manual recursion limit hit condition")
                continue_chain = False
            else:
                # otherwise, we're over the min number and have not hit the recursion limit, continue
                continue_chain = False

            print(
                f"Current chain length: {num_react_steps}. Forcing chain to continue: {continue_chain}"
            )

            if not continue_chain:
                return

            # force the chain to continue
            cfg["recursion_limit"] = cfg["recursion_limit"] - num_react_steps
            _len_before_call = len(self.state)
            out = self.graph.invoke(
                {
                    "messages": self.state
                    + [HumanMessage(content="Use the tools to improve your response.")]
                },
                cfg,
            )
            self._raw_state += out["messages"][_len_before_call:]
            self._delete_messages(
                ids=self._get_msg_id_by_content(
                    "Use the tools to improve your response."
                )
            )
            return _retry(remaining_retries - 1)

        def _final_msg(cfg):
            msg = self._call_model_no_tools(
                self.state
                + [
                    HumanMessage(
                        content="In the history above, the user sent a message, and then you had some internal thoughts. You now need to generate a final, user-facing response to the original user message. Note that the user cannot see any of your intermediate thoughts, only this one, so you may need to repeat information above."
                    )
                ],
            )
            self._raw_state += [
                HumanMessage(
                    content="In the history above, the user sent a message, and then you had some internal thoughts. You now need to generate a final, user-facing response to the original user message. Note that the user cannot see any of your intermediate thoughts, only this one, so you may need to repeat information above."
                ),
                msg,
            ]

        ######

        try:
            _len_before_call = len(self.state)
            out = self.graph.invoke({"messages": messages}, cfg)
            self._raw_state += out["messages"][_len_before_call:]
            _retry(num_restarts)
        except GraphRecursionError:
            print("Recursion limit hit")
            self._delete_messages(
                ids=self._get_msg_id_by_content(
                    "Sorry, need more steps to process this request."
                )
            )
            _final_msg(cfg)

        if tag_on_final_msg:
            _final_msg(cfg)

        stub = self.state[og_len:]

        # Modify the state (but not the returned stub)
        if not self._multiturn_memory:
            self.clear_state()
            self._raw_state = []

        elif not persist_state:
            self._delete_messages(ids=[m.id for m in stub])

        elif self._add_thinking_tag:
            # This only affects the state, not the returned stub
            self._update_messages(
                ids=[
                    m.id
                    for m in stub[:-1]
                    if isinstance(m, AIMessage) and m.content.strip() != ""
                ],
                content=[
                    f"{self._thinking_tokens[0]}{m.content}{self._thinking_tokens[1]}"
                    for m in stub[:-1]
                    if isinstance(m, AIMessage) and m.content.strip() != ""
                ],
            )

        elif self._out_of_steps_msg is not None:
            _ids = [
                m.id
                for m in stub[:-1]
                if isinstance(m, AIMessage)
                and m.content == "Sorry, need more steps to process this request."
            ]
            self._update_messages(
                ids=_ids,
                content=[self._out_of_steps_msg for _ in _ids],
            )

        # Cut out prompt messages before returning
        return stub[len(messages) :]

    def _update_messages(self, ids: List[str], content: List[str]):
        """Change the content of the messages with the given ids"""
        assert len(ids) == len(content), "There should be one content per id"
        new_state = []
        for m in self.state:
            if m.id not in ids:
                new_state.append(m)
            else:
                m.content = content[ids.index(m.id)]
                new_state.append(m)
        self.clear_state()
        self.graph.update_state(
            {"configurable": {"thread_id": self.thread_id}},
            {"messages": new_state},
        )

    def _delete_messages(self, ids: List[str]):
        """Remove messages from the state"""
        from langchain_core.messages import RemoveMessage

        # special edge case: langchain code doesn't handle this well
        # if the ids = the entire state, clear_state() instead
        if ids == [m.id for m in self.state]:
            self.clear_state()
        else:
            self.graph.update_state(
                {"configurable": {"thread_id": self.thread_id}},
                {"messages": [RemoveMessage(id=id) for id in ids]},
            )

    def _get_msg_id_by_content(self, content: str) -> List[str]:
        """Get the ids of the messages with the given content"""
        return [m.id for m in self.state if m.content == content]


def get_token_usage(
    response_metadata: dict,
    default_value: int = 0,
    return_reasoning_tokens: bool = False,
) -> Union[int, Tuple[int, int]]:
    """
    Get the token usage from the response metadata.
    If return_reasoning_tokens is True, return a tuple of (completion_tokens, reasoning_tokens)
    Otherwise, return the completion tokens.
    """
    try:
        # openai
        if return_reasoning_tokens:
            return response_metadata["token_usage"][
                "completion_tokens"
            ], response_metadata["token_usage"]["completion_tokens_details"][
                "reasoning_tokens"
            ]
        else:
            return response_metadata["token_usage"]["completion_tokens"]
    except:
        pass

    try:
        # anthropic
        if return_reasoning_tokens:
            return response_metadata["usage"]["output_tokens"], response_metadata[
                "usage"
            ]["output_tokens_details"]["reasoning_tokens"]
        else:
            return response_metadata["usage"]["output_tokens"]
    except:
        pass

    return default_value


######################
# OpenAI
######################


class OpenAIModel(Model):
    def __init__(self, name, api_key=None):
        super().__init__(name)
        from openai import OpenAI

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        *,
        prompts: List[str] = None,
        dialogs: List[List[Tuple[str, str]]] = None,
        **kwargs,
    ) -> List[str]:
        assert (prompts is None) ^ (dialogs is None), (
            "Exactly one of prompts or dialogs must be provided."
        )
        if prompts is not None:
            dialogs = self.fmt_as_dialog(prompts=prompts)
        else:
            dialogs = self.fmt_as_dialog(dialogs=dialogs)
        if self._multiturn_memory:
            assert len(dialogs) == 1, "Only one conversation at a time is supported"

        return [
            self.client.chat.completions.create(
                model=self.name, messages=dialog, **kwargs
            )
            .choices[0]
            .message.content
            for dialog in dialogs
        ]

    def __repr__(self):
        return f"OpenAIModel(name={self.name})"

    def __str__(self):
        return f"OpenAIModel(name={self.name})"


def submit_openai_batch_job(
    custom_ids: List[str],
    messages: List[dict],
    model: str,
    filename: str = None,
    output_dir: str = None,
    job_desc: str = None,
    do_not_launch: bool = False,
    dump_batch_job_id_to_txt: bool = True,
    hash_custom_ids_if_too_long: bool = False,
    **decoding_kwargs,
) -> dict:
    """
    Submits an OpenAI batch job
    Args:
        custom_ids: custom ids for each message
        messages: list of {role:, content:} dicts, representing a chat
            completion conversation prompt
        filename: root for all filenames of batch job files, batch job ids, etc.
        output_dir: location to write all batch job files, batch job ids, etc.
        job_desc: batch job description; shows up in the Batches API page
        do_not_launch: if True, writes the batch job file but stops short of actually
            submitting the batch job request
        dump_batch_job_id_to_txt: whether to dump the batch job id to a text file
        decoding_kwargs: other kwargs that get in the 'body' field of the request
    """
    assert len(custom_ids) == len(messages), "There should be one custom ID per request"

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{filename}_batch_job_file.jsonl", "w") as f:
        for id, msgs in zip(custom_ids, messages):
            f.write(
                json.dumps(
                    {
                        "custom_id": id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": msgs,
                            **decoding_kwargs,
                        },
                    }
                )
                + "\n"
            )
    print(f"Batch job file saved to {output_dir}/{filename}_batch_job_file.jsonl")

    out = {"batch_job_file": f"{output_dir}/{filename}_batch_job_file.jsonl"}
    if do_not_launch:
        return out

    from openai import OpenAI

    client = OpenAI()
    batch_input_file = client.files.create(
        file=open(f"{output_dir}/{filename}_batch_job_file.jsonl", "rb"),
        purpose="batch",
    )
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": job_desc,
        },
    )
    print(f"Batch job launched with ID {batch_job.id}")
    if dump_batch_job_id_to_txt:
        with open(f"{output_dir}/{filename}_batch_job_id.txt", "w") as f:
            f.write(batch_job.id)

    out["batch_job_id"] = batch_job.id
    return out


def read_openai_batch_job(
    batch_job_id: str,
    include_raw: bool = False,
) -> List[dict]:
    """
    Args:
        remove_unknown_custom_ids: if True, remove any custom ids that are not in the batch job file
            5/1/25: OpenAI sometimes returns responses for custom ids that are not in the batch job file ??
        include_raw: if True, include the raw response in the output
    """
    from openai import OpenAI

    client = OpenAI()

    # Check if the batch job is done
    batch_job = client.batches.retrieve(batch_job_id)
    if batch_job.status not in ["completed", "failed", "cancelled"]:
        raise RuntimeError(
            f"Error: Batch job {batch_job_id} is still in progress with status {batch_job.status}"
        )
    elif batch_job.status != "completed":
        raise ValueError(f"Error: Batch job {batch_job_id} is {batch_job.status}")
    if batch_job.error_file_id is not None and batch_job.output_file_id is None:
        raise ValueError(
            f"Error: Batch job {batch_job_id} failed and outputted an error file but no output file"
        )

    # Get the results
    output_file_id = batch_job.output_file_id
    output_file = client.files.content(output_file_id)
    lines = output_file.text.strip().split("\n")
    jsons = []
    for line in lines:
        try:
            data = json.loads(line)
            jsons.append(
                {
                    "custom_id": data["custom_id"],
                    "output": data["response"]["body"]["choices"][0]["message"][
                        "content"
                    ],
                    "error": False,
                    "raw": line if include_raw else None,
                }
            )
        except Exception as e:
            jsons.append(
                {
                    "custom_id": None,
                    "output": None,
                    "error": True,
                    "raw": line if include_raw else None,
                }
            )
    return jsons


######################
# HF Transformers
######################


class TransformersModel(Model):
    def __init__(self, name, quantize: int = None, **kwargs):
        super().__init__(name)

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # model
        if quantize is not None:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(quantize == 4),
                load_in_8bit=(quantize == 8),
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            name, device_map="auto", token=os.environ.get("HF_TOKEN"), **kwargs
        )
        import torch

        print("GPU memory allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")

    def generate(
        self,
        *,
        prompts: List[str] = None,
        dialogs: List[List[Tuple[str, str]]] = None,
        **kwargs,
    ) -> List[str]:
        assert (prompts is None) ^ (dialogs is None), (
            "Exactly one of prompts or dialogs must be provided."
        )
        if prompts is not None:
            dialogs = self.fmt_as_dialog(prompts=prompts)
        else:
            dialogs = self.fmt_as_dialog(dialogs=dialogs)
        inputs = self.tokenizer.apply_chat_template(
            dialogs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        outputs = self.model.generate(inputs, **kwargs)
        outputs = outputs[:, inputs.shape[1] :]
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def __repr__(self):
        return f"TransformersModel(name={self.name})"

    def __str__(self):
        return f"TransformersModel({self.name})"


######################
# Anthropic
######################


def _openai_dialog_to_anthropic_dialog(dialog: List[dict]):
    """
    Convert {'role': str, 'content': str, **kwargs} -> {'role': str, 'content': ['type': 'text', 'text': str], **kwargs}
    """
    out = []
    for d in dialog:
        out.append(
            {
                "role": d.pop("role"),
                "content": [{"type": "text", "text": d.pop("content")}],
                **d,
            }
        )
    return out


class AnthropicModel(Model):
    def __init__(self, name, api_key=None):
        super().__init__(name)
        from anthropic import Anthropic

        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key)

    def _get_max_tokens(self):
        maxes = {
            "claude-opus-4-20250514": 32000,
            "claude-sonnet-4-20250514": 64000,
            "claude-3-7-sonnet-20250219": 64000,
            "claude-3-5-sonnet-20241022": 64000,
            "claude-3-5-haiku-20241022": 64000,
            "claude-3-opus-20240229": 4096,
            "claude-3-haiku-20240307": 4096,
        }
        if self.name not in maxes:
            print(f"Warning: Unknown model max output tokens: {self.name}")
            return 4096  # best guess
        return maxes[self.name]

    def generate(
        self,
        *,
        prompts: List[str] = None,
        dialogs: List[List[Tuple[str, str]]] = None,
        **kwargs,
    ) -> List[str]:
        assert (prompts is None) ^ (dialogs is None), (
            "Exactly one of prompts or dialogs must be provided."
        )

        if prompts is not None:
            dialogs = self.fmt_as_dialog(prompts=prompts)
        else:
            dialogs = self.fmt_as_dialog(dialogs=dialogs)

        if "max_tokens" not in kwargs:
            print(
                "Anthropic models don't behave well without a max_tokens kwarg. Setting to 64000 for now."
            )
            kwargs["max_tokens"] = 64000
            # kwargs["max_tokens"] = self._get_max_tokens()

        if "reasoning_effort" in kwargs:
            kwargs.update(
                get_reasoning_effort_kwargs(
                    self.name,
                    kwargs["reasoning_effort"],
                    max_tokens=kwargs["max_tokens"],
                )
            )
            kwargs.pop("reasoning_effort")

        out = []
        for dialog in dialogs:
            system = [d for d in dialog if d["role"] == "system"]
            system = [
                l for d in system for l in d["content"]
            ]  # unwrap from the 'role', 'content' keys
            nonsystem = [d for d in dialog if d["role"] != "system"]
            res = self.client.messages.create(
                model=self.name, system=system, messages=nonsystem, **kwargs
            )
            out.append(res.content[0].text)
        return out

    def __repr__(self):
        return f"AnthropicModel(name={self.name})"

    def __str__(self):
        return f"AnthropicModel(name={self.name})"

    def fmt_as_dialog(
        self, prompts: List[str] = None, dialogs: List[List[Tuple[str, str]]] = None
    ):
        """
        Overrides super() to inject _openai_dialog_to_anthropic_dialog
        """
        openai_fmt = super().fmt_as_dialog(prompts=prompts, dialogs=dialogs)
        return [_openai_dialog_to_anthropic_dialog(d) for d in openai_fmt]


def submit_anthropic_batch_job(
    custom_ids: List[str],
    messages: List[dict],
    model: str,
    filename: str = None,
    output_dir: str = None,
    job_desc: str = None,
    do_not_launch: bool = False,
    dump_batch_job_id_to_txt: bool = True,
    hash_custom_ids_if_too_long: bool = False,
    **decoding_kwargs,
) -> dict:
    """
    Submits an OpenAI batch job
    Args:
        custom_ids: custom ids for each message
        messages: list of {role:, content:} dicts, representing a chat
            completion conversation prompt
        filename: root for all filenames of batch job files, batch job ids, etc.
        output_dir: location to write all batch job files, batch job ids, etc.
        job_desc: batch job description; shows up in the Batches API page
        do_not_launch: if True, writes the batch job file but stops short of actually
            submitting the batch job request
        dump_batch_job_id_to_txt: whether to dump the batch job id to a text file
        decoding_kwargs: other kwargs that get in the 'body' field of the request
    """
    assert len(custom_ids) <= 100000, (
        "The Anthropic API only accepts up to 100K requests"
    )
    assert len(custom_ids) == len(messages), "There should be one custom ID per request"

    if "max_tokens" not in decoding_kwargs:
        print(
            "Anthropic models don't behave well without a max_tokens kwarg. Setting to 64000 for now."
        )
        decoding_kwargs["max_tokens"] = (
            64000  # current model maxes for 3.5 Haiku & 3.7 Sonnet
        )

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{filename}_batch_job_file.jsonl", "w") as f:
        for id, msgs in zip(custom_ids, messages):
            msgs = _openai_dialog_to_anthropic_dialog(msgs)
            system = [d for d in msgs if d["role"] == "system"]
            system = [
                d["content"][0] for d in system
            ]  # unwrap from the 'role', 'content' keys
            nonsystem = [d for d in msgs if d["role"] != "system"]
            f.write(
                json.dumps(
                    {
                        "custom_id": id,
                        "params": {
                            "model": model,
                            "system": system,
                            "messages": nonsystem,
                            **decoding_kwargs,
                        },
                    }
                )
                + "\n"
            )
    print(f"Batch job file saved to {output_dir}/{filename}_batch_job_file.jsonl")

    out = {"batch_job_file": f"{output_dir}/{filename}_batch_job_file.jsonl"}
    if do_not_launch:
        return out

    from anthropic import Anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = Anthropic()

    requests = []
    with open(f"{output_dir}/{filename}_batch_job_file.jsonl", "r") as f:
        for line in f:
            js = json.loads(line)

            # custom id cleaning
            if len(js["custom_id"]) > 64:
                if hash_custom_ids_if_too_long:
                    js["custom_id"] = hashlib.sha256(
                        js["custom_id"].encode()
                    ).hexdigest()[:64]
                else:
                    raise ValueError(
                        "Warning: Anthropic restricts custom_id to be at most 64 characters."
                    )
            if not bool(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", js["custom_id"])):
                print(
                    "Warning: Anthropic expects custom_id to match pattern '^[a-zA-Z0-9_-]{1,64}$'. Making some heuristic substitutions"
                )
                js["custom_id"] = re.sub(r"[^a-zA-Z0-9_-]", "-", js["custom_id"])

            requests.append(
                Request(
                    custom_id=js["custom_id"],
                    params=MessageCreateParamsNonStreaming(**js["params"]),
                )
            )
    batch_job = client.messages.batches.create(
        requests=requests,
    )
    print(f"Batch job launched with ID {batch_job.id}")
    if dump_batch_job_id_to_txt:
        with open(f"{output_dir}/{filename}_batch_job_id.txt", "w") as f:
            f.write(batch_job.id)

    out["batch_job_id"] = batch_job.id
    return out


def read_anthropic_batch_job(
    batch_job_id: str,
    include_raw: bool = False,
) -> List[dict]:
    from anthropic import Anthropic

    client = Anthropic()

    # Check if the batch job is done
    batch_job = client.messages.batches.retrieve(batch_job_id)
    if batch_job.processing_status != "ended":
        raise ValueError(f"Batch job {batch_job_id} is {batch_job.processing_status}")

    # Stream results file in memory-efficient chunks, processing one at a time
    jsons = []
    for data in client.messages.batches.results(batch_job_id):
        if data.result.type != "succeeded":
            raise ValueError(
                f"Batch job {batch_job_id} failed with error: {data.result.error}"
            )

        jsons.append(
            {
                "custom_id": data.custom_id,
                "error": data.result.type != "succeeded",
                "output": (
                    data.result.message.content[0].text
                    if data.result.type == "succeeded"
                    else None
                ),
                "raw": vars(data) if include_raw else None,
            }
        )
    return jsons

@lru_cache(maxsize=1)
def load_clip(model_name: str = "ViT-B/32"):
    """
    Load a CLIP model.
    """
    import torch
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device