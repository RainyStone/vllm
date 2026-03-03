# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Observability module for vLLM.

This module provides enhanced tracing and metrics collection capabilities,
including token-level profiling, logprobs tracing, and environment info injection.

Usage:
    from vllm.observability import (
        ObservabilityIntegration,
        ObservableContext,
    )
"""

from vllm.observability.context import ObservableContext
from vllm.observability.integration import ObservabilityIntegration

__all__ = [
    "ObservableContext",
    "ObservabilityIntegration",
]