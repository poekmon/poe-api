from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel


class OpenAIMessageRequest(BaseModel):
    role: Literal["user", "system", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class OpenAIMessageResponse(BaseModel):
    role: Optional[Literal["user", "system", "assistant", "tool"]] = None
    content: Optional[str] = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIMessageRequest]
    stream: Optional[bool] = False
    logit_bias: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None


class OpenAIChatCompletionResponseChoice(BaseModel):
    index: int
    message: OpenAIMessageResponse
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls", "function_call"]] = None


class OpenAIChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    choices: List[OpenAIChatCompletionResponseChoice]
    created: int  # Unix timestamp (in seconds)
    model: str
    system_fingerprint: Optional[str] = None
    object: Literal["chat.completion"] = "chat.completion"
    usage: OpenAIChatCompletionResponseUsage


class OpenAIChatCompletionResponseChunkChoice(BaseModel):
    index: int
    delta: OpenAIMessageResponse
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls", "function_call"]] = None


class OpenAIChatCompletionResponseChunk(BaseModel):
    id: str
    choices: List[OpenAIChatCompletionResponseChunkChoice]
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int  # Unix timestamp (in seconds)
    model: str
    usage: Optional[OpenAIChatCompletionResponseUsage] = None
