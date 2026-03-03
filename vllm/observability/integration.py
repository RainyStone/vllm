# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration adapter for observability module.

This module provides the integration layer between the vLLM engine
and the observability module, enabling clean separation of concerns.
"""

from typing import TYPE_CHECKING, Any, Mapping, Optional

from vllm.observability.context import ObservableContext
from vllm.tracing import Tracer

if TYPE_CHECKING:
    from vllm.config import ObservabilityConfig
    from vllm.tokenizers import TokenizerLike
    from vllm.v1.engine import EngineCoreOutput
    from vllm.v1.metrics.stats import IterationStats


class ObservabilityIntegration:
    """Integration adapter for observability features.

    This class provides a clean interface for the vLLM engine to interact
    with observability features, enabling:
    - Creation of observable contexts for requests
    - Processing outputs for token-level profiling
    - Enhanced tracing on request completion

    When observability is disabled, all operations are no-ops with
    minimal performance overhead (single boolean check).
    """

    def __init__(
        self,
        config: Optional["ObservabilityConfig"],
        tracer: Optional[Tracer],
    ):
        """Initialize the observability integration.

        Args:
            config: ObservabilityConfig instance, or None if disabled.
            tracer: OpenTelemetry tracer instance, or None if tracing disabled.
        """
        self.config = config
        self.tracer = tracer
        # Check if enhanced features are enabled
        self._token_profiling_enabled = (
            config is not None
            and getattr(config, "token_level_profiling", False)
        )
        self._trace_logprobs = (
            config.trace_logprobs if config is not None else None
        )

    @property
    def enabled(self) -> bool:
        """Check if any observability features are enabled."""
        return self._token_profiling_enabled or self.tracer is not None

    @property
    def should_collect_logprobs(self) -> bool:
        """Check if logprobs should be collected for tracing."""
        return self._trace_logprobs is not None

    def create_context(
        self,
        tokenizer: Optional["TokenizerLike"] = None,
    ) -> Optional[ObservableContext]:
        """Create an observable context for a new request.

        Args:
            tokenizer: Optional tokenizer for decoding token IDs.

        Returns:
            ObservableContext if token profiling is enabled, None otherwise.
        """
        if not self._token_profiling_enabled:
            return None
        return ObservableContext.from_new_request(tokenizer=tokenizer)

    def process_output(
        self,
        context: Optional[ObservableContext],
        output: "EngineCoreOutput",
        delta_text: str,
    ) -> None:
        """Process an engine output to update the observable context.

        Args:
            context: ObservableContext to update, or None if disabled.
            output: Engine core output containing iteration stats and events.
            delta_text: Decoded text for the output tokens.
        """
        if context is None:
            return
        context.update_from_output(output, delta_text)

    def finish_request(
        self,
        _context: Optional[ObservableContext],  # Context is obtained from req_state.observable_context
        req_state: Any,  # RequestState
        engine_core_output: Optional["EngineCoreOutput"],
        iteration_stats: Optional["IterationStats"],
        trace_headers: Optional[Mapping[str, str]],
        attributes: Optional[dict[str, Any]] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        """Finish a request and perform tracing.

        Args:
            _context: ObservableContext with collected data (obtained from req_state).
            req_state: Request state containing all request metadata and stats.
            engine_core_output: Output from the engine core.
            iteration_stats: Statistics from the current iteration.
            trace_headers: HTTP headers for trace context propagation.
            attributes: Pre-computed span attributes from the caller.
            error: Optional exception if the request failed.
        """
        if self.tracer is None:
            return

        # Import here to avoid circular dependency
        from vllm.observability.trace_processor import do_enhanced_tracing

        do_enhanced_tracing(
            tracer=self.tracer,
            req_state=req_state,
            engine_core_output=engine_core_output,
            iteration_stats=iteration_stats,
            trace_headers=trace_headers,
            attributes=attributes,
            error=error,
        )

    def get_logprobs_count(self) -> Optional[int]:
        """Get the number of logprobs to collect for tracing.

        Returns:
            Number of top logprobs to collect, or None if not enabled.
        """
        return self._trace_logprobs