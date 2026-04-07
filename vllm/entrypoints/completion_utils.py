from typing import Optional, Tuple, Awaitable
from vllm.config import ModelConfig
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.multimodal import MultiModalDataDict, MultiModalUUIDDict
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.entrypoints.chat_utils import AsyncMultiModalItemTracker, _parse_chat_message_content_part


async def preprocess_completion_with_multi_modal(
    engine_prompt: EngineTokensPrompt,
    request: CompletionRequest,
    model_config: ModelConfig,
) -> list[EngineTokensPrompt]:
    try:
        mm_data, mm_uuids = await parse_completion_multimodal_data(
            request,
            model_config
        )
    except Exception as e:
        raise ValueError(f"Failed to parse multimodal data: {e}") from e

    if mm_data is not None:
        engine_prompt["multi_modal_data"] = mm_data

    if mm_uuids is not None:
        engine_prompt["multi_modal_uuids"] = mm_uuids

    if hasattr(request, "mm_processor_kwargs") and request.mm_processor_kwargs is not None:
        engine_prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

    if hasattr(request, "cache_salt") and request.cache_salt is not None:
        engine_prompt["cache_salt"] = request.cache_salt

    return [engine_prompt]


async def parse_completion_multimodal_data(
    request: CompletionRequest,
    model_config: ModelConfig,
) -> tuple[
    MultiModalDataDict | None,
    MultiModalUUIDDict | None,
]:
    multi_modal_data = request.multi_modal_data
    mm_tracker = AsyncMultiModalItemTracker(model_config)

    mm_parser = mm_tracker.create_parser()
    for part in multi_modal_data:
        _parse_chat_message_content_part(
            part,
            mm_parser,
            wrap_dicts=True,
            interleave_strings=False,
        )
    mm_data, mm_uuids = await mm_tracker.resolve_items()
    return mm_data, mm_uuids
