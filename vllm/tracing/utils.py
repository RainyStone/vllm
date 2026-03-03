# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping

from vllm.logger import init_logger
from vllm.utils.func_utils import run_once

logger = init_logger(__name__)

# Standard W3C headers used for context propagation
TRACE_HEADERS = ["traceparent", "tracestate"]


class SpanAttributes:
    """
    Standard attributes for spans.

    These are largely based on OpenTelemetry Semantic Conventions but are defined
    here as constants so they can be used by any backend or logger.
    """

    # Attribute names copied from OTel semantic conventions to avoid version conflicts
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"

    # Additional request parameters
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_MIN_P = "gen_ai.request.min_p"
    GEN_AI_REQUEST_REPETITION_PENALTY = "gen_ai.request.repetition_penalty"
    GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"

    # Attribute names added until they are added to the semantic conventions
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"

    # Latency attributes
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = "gen_ai.latency.time_in_model_inference"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = "gen_ai.latency.time_in_model_forward"
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = "gen_ai.latency.time_in_model_execute"

    # Trace level
    GEN_AI_REQUEST_TRACE_LEVEL = "gen_ai.request.trace_level"

    # Per-token trace attributes (enhanced tracing level 2)
    GEN_AI_LATENCY_PER_TOKEN_GENERATION_TIME = "gen_ai.latency.per_token_generation_time"
    GEN_AI_LATENCY_PER_TOKEN_SCHEDULED_TIME = "gen_ai.latency.per_token_scheduled_time"
    GEN_AI_ITERATION_PER_TOKEN_BATCH_SIZE = "gen_ai.iteration.per_token_batch_size"
    GEN_AI_ITERATION_PER_TOKEN_WAITING_SIZE = "gen_ai.iteration.per_token_waiting_size"
    GEN_AI_ITERATION_PER_TOKEN_TOTAL_TOKENS = "gen_ai.iteration.per_token_total_tokens"
    GEN_AI_ITERATION_PER_TOKEN_CACHED_TOKENS = "gen_ai.iteration.per_token_cached_tokens"
    GEN_AI_RESPONSE_PER_TOKEN_CANDIDATE_DECODED_TOKENS = "gen_ai.response.per_token_candidate_decoded_tokens"
    GEN_AI_RESPONSE_PER_TOKEN_CANDIDATE_TOKEN_IDS = "gen_ai.response.per_token_candidate_token_ids"
    GEN_AI_RESPONSE_PER_TOKEN_CANDIDATE_TOKENS_LOGPROBS = "gen_ai.response.per_token_candidate_tokens_logprobs"
    GEN_AI_REQUEST_PREFILL_CHUNKED_STATS = "gen_ai.request.prefill_chunked_stats"

    # SOFA trace context
    SOFA_TRACE_ID = "Parent-TraceId"
    SOFA_RPC_ID = "Parent-RpcId"

    # Alipay/AICloud specific attributes
    REQUEST_ID = "alipay.aicloud.request_id"
    API_KEY_ID = "alipay.aicloud.api_key_id"
    POD_IP = "alipay.base.ip"
    POD_NAME = "alipay.base.pod_name"
    HOSTNAME = "alipay.base.host"
    IDC = "alipay.base.idc"
    MODEL_SERVICE_ID = "alipay.aicloud.model_service_id"
    MODEL_INSTANCE_ID = "alipay.aicloud.model_instance_id"
    MODEL_INSTANCE_NAME = "alipay.aicloud.model_instance_name"
    APP_NAME = "alipay.aicloud.app_name"
    ALIPAY_LATENCY_TIME_IN_API_SERVER = "alipay.aicloud.time_in_api_server"
    ALIPAY_LATENCY_TIME_IN_INPUT_PROCESSING = "alipay.aicloud.time_in_input_processing"
    ALIPAY_LATENCY_TIME_IN_OUTPUT_QUEUE = "alipay.aicloud.time_in_output_queue"
    ALIPAY_LATENCY_TIME_IN_OUTPUT_PROCESSING = "alipay.aicloud.time_in_output_processing"
    ALIPAY_REQUEST_PARAMS = "alipay.aicloud.request_params"
    ALIPAY_REQ_METRIC = "alipay.aicloud.req_metric"


class LoadingSpanAttributes:
    """Custom attributes for code-level tracing (file, line number)."""

    CODE_NAMESPACE = "code.namespace"
    CODE_FUNCTION = "code.function"
    CODE_FILEPATH = "code.filepath"
    CODE_LINENO = "code.lineno"


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    """Check if the provided headers dictionary contains trace context."""
    return any(h in headers for h in TRACE_HEADERS)


def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    """
    Extract only trace-related headers from a larger header dictionary.
    Useful for logging or passing context to a non-OTel client.
    """
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning("Received a request with trace context but tracing is disabled")
