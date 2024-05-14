import time
from typing import Annotated
from fastapi import APIRouter, Header, Response
from fastapi.responses import StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import fastapi_poe as fp
from poe_api.config import CONFIG
from poe_api.exceptions import AuthenticationFailedException
from poe_api.models import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionResponseChoice,
    OpenAIChatCompletionResponseChunk,
    OpenAIChatCompletionResponseChunkChoice,
    OpenAIMessageResponse,
)
from poe_api.utils import convert_role_to_poe, gen_id, get_bot_name, calculate_usage


api_router = APIRouter()


def authenticate(auth_key):
    if CONFIG.auth_secret:
        if auth_key is None:
            raise AuthenticationFailedException("Missing authentication header")
        if "Bearer " in auth_key:
            auth_key = auth_key.split("Bearer ")[1]
        if auth_key.strip() != CONFIG.auth_secret.strip():
            raise AuthenticationFailedException()


@api_router.get("/")
async def root():
    return Response("running")


@api_router.get("/status")
async def status(authorization: Annotated[str | None, Header()] = None):
    authenticate(auth_key=authorization)
    return {"status": "ok"}


@api_router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatCompletionRequest, authorization: Annotated[str | None, Header()] = None):
    bot_name = get_bot_name(request.model)

    API_KEY = CONFIG.poe_api_key

    authenticate(auth_key=authorization)

    messages = [fp.ProtocolMessage(role=convert_role_to_poe(msg.role), content=msg.content) for msg in request.messages]

    response_id = gen_id()

    extra_kwargs = {}
    if request.logit_bias:
        extra_kwargs["logit_bias"] = request.logit_bias
    if request.temperature:
        extra_kwargs["temperature"] = request.temperature

    if request.stream:

        async def generate():
            response_text = ""
            async for partial in fp.get_bot_response(messages=messages, bot_name=bot_name, api_key=API_KEY, **extra_kwargs):
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
        async for partial in fp.get_bot_response(messages=messages, bot_name=bot_name, api_key=API_KEY, **extra_kwargs):
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
