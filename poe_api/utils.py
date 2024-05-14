import uuid
from typing import Literal, List

import fastapi_poe as fp
import tiktoken

from poe_api.models import OpenAIMessageRequest, OpenAIChatCompletionResponseUsage
from poe_api.const import openai_model_map


def convert_role_to_poe(role: Literal["system", "user", "assistant", "tool"]) -> str:
    if role == "system":
        return "system"
    elif role == "user":
        return "user"
    elif role == "assistant":
        return "bot"
    elif role == "tool":
        return "bot"
    else:
        raise ValueError(f"Unsupported role: {role}")


def convert_role_to_openai(role: Literal["system", "user", "bot"]) -> str:
    if role == "system":
        return "system"
    elif role == "user":
        return "user"
    elif role == "bot":
        return "assistant"
    else:
        raise ValueError(f"Unsupported role: {role}")


def gen_id() -> str:
    return str(uuid.uuid4())


def get_bot_name(model: str) -> str:
    bot_name = openai_model_map.get(model, model)
    return bot_name


UsageTargetType = str | OpenAIMessageRequest | fp.PartialResponse | List[str] | List[OpenAIMessageRequest] | List[fp.PartialResponse]


def convert_usage_target_to_str(target: UsageTargetType) -> str:
    def convert_msg_to_str(msg: str | OpenAIMessageRequest | fp.PartialResponse) -> str:
        if isinstance(msg, str):
            return msg
        elif isinstance(msg, OpenAIMessageRequest):
            return msg.content
        elif isinstance(msg, fp.PartialResponse):
            return msg.text
        else:
            raise TypeError(f"Unsupported type: {type(msg)}")

    if isinstance(target, list):
        return "".join([convert_msg_to_str(msg) for msg in target])
    else:
        return convert_msg_to_str(target)


def calculate_usage(prompts: UsageTargetType, response: UsageTargetType) -> OpenAIChatCompletionResponseUsage:
    encoding = tiktoken.get_encoding("cl100k_base")
    prompt_tokens = len(encoding.encode(convert_usage_target_to_str(prompts)))
    completion_tokens = len(encoding.encode(convert_usage_target_to_str(response)))

    total_tokens = prompt_tokens + completion_tokens

    return OpenAIChatCompletionResponseUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
