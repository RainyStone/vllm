# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Environment and trace info extraction for observability.

This module provides utilities for extracting SOFA trace information
and environment metadata to be injected into traces.
"""

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass
class SofaTraceInfo:
    """SOFA trace information extracted from request headers.

    Attributes:
        sofa_trace_id: SOFA trace ID for distributed tracing.
        sofa_rpc_id: SOFA RPC ID for identifying the RPC call.
        request_id: External request ID.
        aigw_app_key_id: AIGW application key ID.
    """

    sofa_trace_id: Optional[str] = None
    sofa_rpc_id: Optional[str] = None
    request_id: Optional[str] = None
    aigw_app_key_id: Optional[str] = None


@dataclass
class EnvInfo:
    """Environment information extracted from system environment variables.

    This class collects deployment metadata useful for debugging and monitoring.

    Attributes:
        pod_ip: Pod IP address.
        idc: Data center identifier.
        model_service_id: Model service identifier.
        model_instance_id: Model instance identifier.
        pod_name: Pod name.
        hostname: Hostname of the machine.
        model_instance_name: Model instance name.
    """

    pod_ip: Optional[str] = None
    idc: Optional[str] = None
    model_service_id: Optional[str] = None
    model_instance_id: Optional[str] = None
    pod_name: Optional[str] = None
    hostname: Optional[str] = None
    model_instance_name: Optional[str] = None


def get_sofa_trace_info(parent_trace_headers: Mapping[str, str]) -> SofaTraceInfo:
    """Extract SOFA trace information from request headers.

    Args:
        parent_trace_headers: HTTP headers from the incoming request.

    Returns:
        SofaTraceInfo instance with extracted trace information.
    """
    sofa_trace_info = SofaTraceInfo()
    for key, value in parent_trace_headers.items():
        if key == "SOFA-TraceId":
            sofa_trace_info.sofa_trace_id = value
        elif key == "SOFA-RpcId":
            sofa_trace_info.sofa_rpc_id = value
        elif key == "X-Request-ID":
            sofa_trace_info.request_id = value
        elif key == "X-AIGW-APP-KeyId":
            sofa_trace_info.aigw_app_key_id = value
    return sofa_trace_info


def get_env_info() -> EnvInfo:
    """Extract environment information from system environment variables.

    Returns:
        EnvInfo instance with extracted environment metadata.
    """
    env_info = EnvInfo()
    if ip := os.getenv("POD_IP"):
        env_info.pod_ip = ip
    if idc := os.getenv("ALIPAY_APP_IDC"):
        env_info.idc = idc
    if model_service_id := os.getenv("MODEL_SERVICE_ID"):
        env_info.model_service_id = model_service_id
    if model_instance_id := os.getenv("MODEL_INSTANCE_NAME"):
        env_info.model_instance_id = model_instance_id
    if pod_name := os.getenv("ALIPAY_POD_NAME"):
        env_info.pod_name = pod_name
    if hostname := os.getenv("HOSTNAME"):
        env_info.hostname = hostname
    if model_instance_name := os.getenv("MODEL_INSTANCE_NAME"):
        env_info.model_instance_name = model_instance_name
    return env_info