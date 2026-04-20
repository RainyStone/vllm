from typing import Optional, Tuple, Awaitable
from vllm.config import ModelConfig
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.multimodal import MultiModalDataDict, MultiModalUUIDDict
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.entrypoints.chat_utils import AsyncMultiModalItemTracker, _parse_chat_message_content_part


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
