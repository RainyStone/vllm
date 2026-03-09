# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeVar

from fastapi import Request
from fastapi.exceptions import RequestValidationError

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)

# Used internally
_ChatCompletionResponseChoiceT = TypeVar(
    "_ChatCompletionResponseChoiceT",
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)


def maybe_filter_parallel_tool_calls(
    choice: _ChatCompletionResponseChoiceT, request: ChatCompletionRequest
) -> _ChatCompletionResponseChoiceT:
    """Filter to first tool call only when parallel_tool_calls is False."""

    if request.parallel_tool_calls:
        return choice

    if isinstance(choice, ChatCompletionResponseChoice) and choice.message.tool_calls:
        choice.message.tool_calls = choice.message.tool_calls[:1]
    elif (
        isinstance(choice, ChatCompletionResponseStreamChoice)
        and choice.delta.tool_calls
    ):
        choice.delta.tool_calls = [
            tool_call for tool_call in choice.delta.tool_calls if tool_call.index == 0
        ]

    return choice


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=["Unsupported Media Type: Only 'application/json' is allowed"]
        )


def is_enable_thinking_request(request: ChatCompletionRequest) -> bool:
    chat_template_kwargs = request.chat_template_kwargs

    # if chat_template_kwargs is None, we assume thinking padding is needed
    if chat_template_kwargs is None:
        return True

    # if "enable_thinking" exists in chat_template_kwargs, we use its value as the padding setting
    for field in ("thinking", "enable_thinking"):
        if field in chat_template_kwargs:
            value = chat_template_kwargs[field]
            if not isinstance(value, bool):
                raise TypeError(
                    f"chat_template_kwargs['{field}'] must be a bool, "
                    f"got {type(value).__name__}"
                )
            # RequestPadding / NoRequestPadding
            return value

    # padding setting does not exist in request (DefaultPaddingSetting)
    return True
