import time
from typing import List, Literal
import uuid
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import fastapi_poe as fp
from models import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionResponseChoice,
    OpenAIChatCompletionResponseChunk,
    OpenAIChatCompletionResponseChunkChoice,
    OpenAIChatCompletionResponseUsage,
    OpenAIMessageRequest,
    OpenAIMessageResponse,
)
import tiktoken
from const import openai_model_map

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "your-api-key"


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
    bot_name = openai_model_map.get(model)
    if not bot_name:
        raise ValueError(f"Unsupported model: {model}")
    return bot_name


@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatCompletionRequest):
    bot_name = get_bot_name(request.model)

    messages = [fp.ProtocolMessage(role=convert_role_to_poe(msg.role), content=msg.content) for msg in request.messages]

    response_id = gen_id()

    if request.stream:

        async def generate():
            response_text = ""
            async for partial in fp.get_bot_response(messages=messages, bot_name=bot_name, api_key=API_KEY):
                chunk = OpenAIChatCompletionResponseChunk(
                    id=response_id,
                    object="chat.completion.chunk",
                    choices=[
                        OpenAIChatCompletionResponseChunkChoice(
                            index=0, finish_reason=None, delta=OpenAIMessageResponse(role="assistant", content=partial.text)
                        )
                    ],
                    created=int(time.time()),
                    model=request.model,
                )
                response_text += partial.text
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

            chunk = OpenAIChatCompletionResponseChunk(
                id=response_id,
                object="chat.completion.chunk",
                choices=[OpenAIChatCompletionResponseChunkChoice(index=0, finish_reason="stop", delta=OpenAIMessageResponse())],
                created=int(time.time()),
                model=request.model,
            )
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

            # Send the final chunk with usage
            usage = calculate_usage(request.messages, response_text)
            chunk = OpenAIChatCompletionResponseChunk(
                id=response_id,
                object="chat.completion.chunk",
                choices=[],
                created=int(time.time()),
                model=request.model,
                usage=usage,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    else:
        response_text = ""
        async for partial in fp.get_bot_response(messages=messages, bot_name=bot_name, api_key=API_KEY):
            response_text += partial.text

        usage = calculate_usage(request.messages, response_text)

        response = OpenAIChatCompletionResponse(
            id=gen_id(),
            object="chat.completion",
            choices=[
                OpenAIChatCompletionResponseChoice(
                    index=0, finish_reason="stop", message=OpenAIMessageResponse(role="assistant", content=response_text)
                )
            ],
            created=int(time.time()),
            model=request.model,
            usage=usage,
        )
        return response


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8800)
