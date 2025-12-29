# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ADK Integration Layer for Flow-based Agent Building Platform.

This package provides the core functionality to convert flow JSON definitions
into executable ADK agents. It supports multi-tenant environments with
dynamic agent creation, multiple LLM providers, and MCP tool integration.

Main Components:
    - FlowBuilder: Converts flow JSON into ADK agent hierarchies
    - FlowExecutor: Runs flows and manages sessions
    - FlowSchema: Validates and parses flow JSON

Usage:
    from adk_integration import FlowExecutor, FlowSchema

    # Create executor
    executor = FlowExecutor(
        get_flow_callback=get_flow_from_db,
        get_api_keys_callback=get_tenant_api_keys,
        session_service_factory=create_session_service,
    )

    # Run a flow
    async for event in executor.run_flow(
        tenant_id="tenant_123",
        flow_id="flow_456",
        user_id="user_789",
        session_id="session_abc",
        message="Hello!",
    ):
        print(event)
"""

from __future__ import annotations

from .schema.flow_schema import FlowSchema
from .builder.flow_builder import FlowBuilder
from .executor.flow_executor import FlowExecutor

__all__ = [
    "FlowSchema",
    "FlowBuilder",
    "FlowExecutor",
]

__version__ = "0.1.0"
