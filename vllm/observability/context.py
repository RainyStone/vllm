# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ObservableContext: Per-request context for collecting detailed observability data.

This module provides token-level profiling and detailed trace information
collection for each request.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union
import itertools

from vllm.tokenizers.detokenizer_utils import convert_ids_list_to_tokens

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.v1.engine import EngineCoreEvent, EngineCoreOutput
    from vllm.v1.outputs import IterStats

NONES = itertools.repeat(None)


@dataclass
class ObservableContext:
    """Context for collecting detailed observability data per request.

    This class collects token-level profiling data including:
    - Iteration statistics (batch size, waiting size, total tokens)
    - Timing information (scheduled time, token generation time)
    - Token candidates with logprobs for enhanced tracing
    - Engine events for detailed execution tracing
    """

    iter_batch_size: list[int] = field(default_factory=list)
    iter_waiting_size: list[int] = field(default_factory=list)
    iter_total_tokens_count: list[int] = field(default_factory=list)
    scheduled_time: list[int] = field(default_factory=list)
    token_time: list[int] = field(default_factory=list)
    candidate_token_ids: Optional[list[Union[int, list[int]]]] = field(
        default_factory=list
    )
    candidate_decoded_tokens: Optional[list[Union[str, list[str]]]] = field(
        default_factory=list
    )
    candidate_token_probs: Optional[list[list[float]]] = field(default_factory=list)
    tokenizer: Optional["TokenizerLike"] = None
    events: Optional[list["EngineCoreEvent"]] = field(default_factory=list)
    num_cached_tokens: Optional[int] = None
    not_empty: bool = False

    @classmethod
    def from_new_request(
        cls, tokenizer: Optional["TokenizerLike"] = None
    ) -> "ObservableContext":
        """Create a new ObservableContext for a request.

        Args:
            tokenizer: Optional tokenizer for decoding token IDs.

        Returns:
            A new ObservableContext instance.
        """
        return cls(
            iter_batch_size=[],
            iter_waiting_size=[],
            iter_total_tokens_count=[],
            scheduled_time=[],
            token_time=[],
            candidate_token_ids=[],
            candidate_decoded_tokens=[],
            candidate_token_probs=[],
            tokenizer=tokenizer,
            events=[],
        )

    def _update_iter_stats(
        self,
        iter_stats: "IterStats",
        new_token_ids: list[int],
        delta_text: str,
    ) -> None:
        """Update iteration statistics from output.

        Args:
            iter_stats: Iteration statistics from the engine.
            new_token_ids: New token IDs generated in this iteration.
            delta_text: Decoded text for the new tokens.
        """
        if not new_token_ids:
            return

        self.not_empty = True
        new_tokens_num = len(new_token_ids)
        first_schedule_time = self.scheduled_time[0] if self.scheduled_time else 0

        for _ in range(new_tokens_num):
            self.iter_total_tokens_count.append(iter_stats.iter_total_tokens_count)
            self.scheduled_time.append(
                iter_stats.token_scheduled_time - first_schedule_time
            )
            self.token_time.append(iter_stats.token_output_time - first_schedule_time)
            self.iter_batch_size.append(iter_stats.iter_batch_size)
            self.iter_waiting_size.append(iter_stats.iter_waiting_size)

        if self.num_cached_tokens is None:
            self.num_cached_tokens = iter_stats.num_cached_tokens

        # Handle logprobs if available
        if iter_stats.logprobs_tensors_for_trace:
            self._update_with_logprobs(iter_stats)
        else:
            self._update_without_logprobs(new_token_ids, delta_text)

    def _update_with_logprobs(self, iter_stats: "IterStats") -> None:
        """Update context with logprobs information.

        Args:
            iter_stats: Iteration statistics containing logprobs tensors.
        """
        logprobs_data = iter_stats.logprobs_tensors_for_trace
        for token_ids, logprobs in zip(logprobs_data.logprob_token_ids, logprobs_data.logprobs):
            token_ids_list = token_ids.tolist()
            # Detokenize (non-incrementally).
            decoded_tokens = (
                [None] * len(token_ids_list)
                if self.tokenizer is None
                else convert_ids_list_to_tokens(self.tokenizer, token_ids_list)
            )
            # Update with the Logprob dictionary for this pos.
            self.candidate_token_ids.append(token_ids_list)
            self.candidate_decoded_tokens.append(decoded_tokens)
            self.candidate_token_probs.append(logprobs.tolist())

    def _update_without_logprobs(
        self, new_token_ids: list[int], delta_text: str
    ) -> None:
        """Update context without logprobs information.

        Args:
            new_token_ids: New token IDs generated.
            delta_text: Decoded text for the tokens.
        """

        self.candidate_token_ids.extend(new_token_ids)
        if delta_text:
            self.candidate_decoded_tokens.append(delta_text)
            for _ in range(len(new_token_ids) - 1):
                self.candidate_decoded_tokens.append("")
        else:
            decoded_tokens = (
                NONES
                if self.tokenizer is None
                else convert_ids_list_to_tokens(self.tokenizer, new_token_ids)
            )
            self.candidate_decoded_tokens.extend(decoded_tokens)

    def _update_events(self, events: list["EngineCoreEvent"]) -> None:
        """Update with engine events.

        Args:
            events: List of engine events to record.
        """
        self.events.extend(events)

    def update_from_output(
        self, output: "EngineCoreOutput", delta_text: str
    ) -> None:
        """Update context from an engine core output.

        Args:
            output: Engine core output containing iteration stats and events.
            delta_text: Decoded text for the output tokens.
        """
        if output.iter_stats is not None:
            self._update_iter_stats(output.iter_stats, output.new_token_ids, delta_text)
        if output.events is not None:
            self._update_events(output.events)