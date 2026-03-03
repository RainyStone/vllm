# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Trace processor for observability.

This module provides the core tracing logic for collecting and emitting
detailed trace information for LLM requests.
"""

import json
import time
from typing import TYPE_CHECKING, Any, Mapping, Optional

import msgspec
import orjson

from vllm.tracing import (
    APP_NAME,
    ENHANCED_TRACE_LEVEL,
    NORMAL_TRACE_LEVEL,
    SpanAttributes,
    SpanKind,
    Tracer,
    extract_trace_context,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.observability.context import ObservableContext
from vllm.observability.env_info import get_env_info, get_sofa_trace_info

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutput
    from vllm.v1.metrics.stats import IterationStats


def do_enhanced_tracing(
    tracer: Tracer,
    req_state: Any,  # RequestState
    engine_core_output: Optional["EngineCoreOutput"],
    iteration_stats: Optional["IterationStats"],
    trace_headers: Optional[Mapping[str, str]],
    attributes: Optional[dict[str, Any]] = None,
    error: Optional[BaseException] = None,
) -> None:
    """Perform enhanced tracing for a completed request.

    This function creates a detailed trace span with:
    - Timing metrics (TTFT, E2E latency, queue time, etc.)
    - Request parameters
    - Token-level profiling data (if enabled)
    - Environment and SOFA trace information
    - Engine events

    Args:
        tracer: OpenTelemetry tracer instance.
        req_state: Request state containing all request metadata and stats.
        engine_core_output: Output from the engine core.
        iteration_stats: Statistics from the current iteration.
        trace_headers: HTTP headers for trace context propagation.
        attributes: Pre-computed span attributes from the caller.
        error: Optional exception if the request failed.
    """
    assert req_state.stats is not None

    metrics = req_state.stats

    # Determine span start time
    span_start_time = (
        getattr(metrics, "api_server_arrival_time", None)
        or metrics.arrival_time
    )
    arrival_time_nano_seconds = int(span_start_time * 1e9)

    # Extract trace context
    trace_context = extract_trace_context(
        trace_headers or engine_core_output.trace_headers if engine_core_output else None
    )

    # Calculate prompt length
    prompt_length = length_from_prompt_token_ids_or_embeds(
        req_state.prompt_token_ids, req_state.prompt_embeds
    )

    with tracer.start_as_current_span(
        "llm_request",
        kind=SpanKind.SERVER,
        context=trace_context,
        start_time=arrival_time_nano_seconds,
    ) as span:
        # Set pre-computed attributes from caller if provided
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)

        _set_timing_attributes(span, metrics, iteration_stats)
        _set_request_attributes(span, req_state, prompt_length)
        _set_trace_info_attributes(span, trace_headers)
        _set_env_info_attributes(span)
        _set_observable_context_attributes(span, req_state.observable_context)
        _set_finish_status(span, engine_core_output, error)


def _set_timing_attributes(
    span: Any,
    metrics: Any,
    iteration_stats: Optional["IterationStats"],
) -> None:
    """Set timing-related span attributes.

    Args:
        span: OpenTelemetry span.
        metrics: Request state metrics.
        iteration_stats: Current iteration statistics.
    """
    e2e_time = (
        iteration_stats.iteration_timestamp - metrics.arrival_time
        if iteration_stats
        else time.time() - metrics.arrival_time
    )
    queued_time = metrics.scheduled_ts - metrics.queued_ts
    prefill_time = metrics.first_token_ts - metrics.scheduled_ts
    decode_time = metrics.last_token_ts - metrics.first_token_ts
    inference_time = metrics.last_token_ts - metrics.scheduled_ts

    span.set_attribute(SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN, metrics.first_token_latency)
    span.set_attribute(SpanAttributes.GEN_AI_LATENCY_E2E, e2e_time)
    span.set_attribute(SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE, queued_time)
    span.set_attribute(SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS, metrics.num_generation_tokens)
    span.set_attribute(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL, prefill_time)
    span.set_attribute(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE, decode_time)
    span.set_attribute(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE, inference_time)

    # Extended timing attributes
    if hasattr(metrics, "api_server_arrival_time"):
        api_server_time = metrics.arrival_time - metrics.api_server_arrival_time
        span.set_attribute(SpanAttributes.ALIPAY_LATENCY_TIME_IN_API_SERVER, api_server_time)
    if hasattr(metrics, "process_input_finish_time"):
        process_input_time = metrics.process_input_finish_time - metrics.arrival_time
        span.set_attribute(SpanAttributes.ALIPAY_LATENCY_TIME_IN_INPUT_PROCESSING, process_input_time)
    if hasattr(metrics, "output_token_queued_latency"):
        span.set_attribute(SpanAttributes.ALIPAY_LATENCY_TIME_IN_OUTPUT_QUEUE, metrics.output_token_queued_latency)
    if hasattr(metrics, "output_token_process_latency"):
        span.set_attribute(SpanAttributes.ALIPAY_LATENCY_TIME_IN_OUTPUT_PROCESSING, metrics.output_token_process_latency)


def _set_request_attributes(
    span: Any,
    req_state: Any,
    prompt_length: int,
) -> None:
    """Set request-related span attributes.

    Args:
        span: OpenTelemetry span.
        req_state: Request state with parameters.
        prompt_length: Length of the prompt in tokens.
    """
    span.set_attribute(SpanAttributes.GEN_AI_REQUEST_ID, req_state.external_req_id)
    span.set_attribute(SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS, prompt_length)

    # Set request params if available
    if hasattr(req_state, "request_params") and req_state.request_params:
        span.set_attribute(SpanAttributes.ALIPAY_REQUEST_PARAMS, json.dumps(req_state.request_params))
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TOP_P, req_state.request_params.get("top_p"))
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TOP_K, req_state.request_params.get("top_k"))
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS, req_state.request_params.get("max_tokens"))
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TEMPERATURE, req_state.request_params.get("temperature"))
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_N, req_state.request_params.get("n"))
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_MIN_P, req_state.request_params.get("min_p"))
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY, req_state.request_params.get("frequency_penalty"))
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_REPETITION_PENALTY, req_state.request_params.get("repetition_penalty"))
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY, req_state.request_params.get("presence_penalty"))

    span.set_attribute(SpanAttributes.APP_NAME, APP_NAME)

    # Set metrics if available
    if req_state.stats:
        span.set_attribute(SpanAttributes.ALIPAY_REQ_METRIC, msgspec.json.encode(req_state.stats).decode("utf-8"))


def _set_trace_info_attributes(
    span: Any,
    trace_headers: Optional[Mapping[str, str]],
) -> None:
    """Set SOFA trace information span attributes.

    Args:
        span: OpenTelemetry span.
        trace_headers: HTTP headers containing trace info.
    """
    if not trace_headers:
        return

    sofa_trace_info = get_sofa_trace_info(trace_headers)
    if sofa_trace_info.sofa_trace_id:
        span.set_attribute(SpanAttributes.SOFA_TRACE_ID, sofa_trace_info.sofa_trace_id)
    if sofa_trace_info.sofa_rpc_id:
        span.set_attribute(SpanAttributes.SOFA_RPC_ID, sofa_trace_info.sofa_rpc_id)
    if sofa_trace_info.request_id:
        span.set_attribute(SpanAttributes.REQUEST_ID, sofa_trace_info.request_id)
    if sofa_trace_info.aigw_app_key_id:
        span.set_attribute(SpanAttributes.API_KEY_ID, sofa_trace_info.aigw_app_key_id)


def _set_env_info_attributes(span: Any) -> None:
    """Set environment information span attributes.

    Args:
        span: OpenTelemetry span.
    """
    env_info = get_env_info()
    if env_info.pod_ip:
        span.set_attribute(SpanAttributes.POD_IP, env_info.pod_ip)
    if env_info.idc:
        span.set_attribute(SpanAttributes.IDC, env_info.idc)
    if env_info.model_instance_id:
        span.set_attribute(SpanAttributes.MODEL_INSTANCE_ID, env_info.model_instance_id)
    if env_info.model_service_id:
        span.set_attribute(SpanAttributes.MODEL_SERVICE_ID, env_info.model_service_id)
    if env_info.model_instance_name:
        span.set_attribute(SpanAttributes.MODEL_INSTANCE_NAME, env_info.model_instance_name)
    if env_info.pod_name:
        span.set_attribute(SpanAttributes.POD_NAME, env_info.pod_name)
    if env_info.hostname:
        span.set_attribute(SpanAttributes.HOSTNAME, env_info.hostname)


def _set_observable_context_attributes(
    span: Any,
    observable_context: Optional[ObservableContext],
) -> None:
    """Set observable context token-level attributes.

    Args:
        span: OpenTelemetry span.
        observable_context: Context with token-level profiling data.
    """
    if observable_context and observable_context.not_empty:
        ctx = observable_context
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TRACE_LEVEL, ENHANCED_TRACE_LEVEL)

        if ctx.candidate_token_ids:
            span.set_attribute(
                SpanAttributes.GEN_AI_RESPONSE_PER_TOKEN_CANDIDATE_TOKEN_IDS,
                orjson.dumps(ctx.candidate_token_ids)
            )
        if ctx.candidate_decoded_tokens:
            span.set_attribute(
                SpanAttributes.GEN_AI_RESPONSE_PER_TOKEN_CANDIDATE_DECODED_TOKENS,
                orjson.dumps(ctx.candidate_decoded_tokens)
            )
        if ctx.candidate_token_probs:
            rounded_probs = [
                [round(prob, 4) for prob in sublist]
                for sublist in ctx.candidate_token_probs
            ]
            span.set_attribute(
                SpanAttributes.GEN_AI_RESPONSE_PER_TOKEN_CANDIDATE_TOKENS_LOGPROBS,
                orjson.dumps(rounded_probs)
            )
        if ctx.iter_batch_size:
            span.set_attribute(
                SpanAttributes.GEN_AI_ITERATION_PER_TOKEN_BATCH_SIZE,
                orjson.dumps(ctx.iter_batch_size)
            )
        if ctx.iter_waiting_size:
            span.set_attribute(
                SpanAttributes.GEN_AI_ITERATION_PER_TOKEN_WAITING_SIZE,
                orjson.dumps(ctx.iter_waiting_size)
            )
        if ctx.iter_total_tokens_count:
            span.set_attribute(
                SpanAttributes.GEN_AI_ITERATION_PER_TOKEN_TOTAL_TOKENS,
                orjson.dumps(ctx.iter_total_tokens_count)
            )
        if ctx.token_time:
            span.set_attribute(
                SpanAttributes.GEN_AI_LATENCY_PER_TOKEN_GENERATION_TIME,
                orjson.dumps(ctx.token_time)
            )
        if ctx.scheduled_time:
            span.set_attribute(
                SpanAttributes.GEN_AI_LATENCY_PER_TOKEN_SCHEDULED_TIME,
                orjson.dumps(ctx.scheduled_time)
            )
        if ctx.num_cached_tokens is not None:
            span.set_attribute(
                SpanAttributes.GEN_AI_ITERATION_PER_TOKEN_CACHED_TOKENS,
                ctx.num_cached_tokens
            )

        # Add events
        if ctx.events:
            from vllm.v1.engine import get_event_name
            for event in ctx.events:
                span.add_event(
                    name=get_event_name(event.type),
                    timestamp=int(event.wall_clock_timestamp * 1e9),
                    attributes=event.attributes,
                )
    else:
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TRACE_LEVEL, NORMAL_TRACE_LEVEL)


def _set_finish_status(
    span: Any,
    engine_core_output: Optional["EngineCoreOutput"],
    error: Optional[BaseException],
) -> None:
    """Set finish reason and status on the span.

    Args:
        span: OpenTelemetry span.
        engine_core_output: Engine output with finish reason.
        error: Optional exception that occurred.
    """
    from opentelemetry.trace import Status, StatusCode
    from vllm.v1.engine import FinishReason

    # Set finish reason
    finish_reasons = [
        str(engine_core_output.finish_reason if engine_core_output else FinishReason.ABORT)
    ]
    span.set_attribute(SpanAttributes.GEN_AI_RESPONSE_FINISH_REASON, json.dumps(finish_reasons))

    # Set status
    if error:
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, str(error)))
    else:
        span.set_status(Status(StatusCode.OK))