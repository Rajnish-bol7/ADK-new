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
Flow Builder - Main orchestrator for building ADK agents from flow definitions.

This is the central component that takes a flow JSON definition and converts it
into a complete ADK agent hierarchy ready for execution.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from google.adk import Agent

from ..schema.flow_schema import (
    FlowSchema,
    FlowNode,
    NodeType,
    AgentNodeData,
    ChatModelNodeData,
    DatabaseNodeData,
    MCPNodeData,
    ToolNodeData,
    CodeNodeData,
    HttpRequestNodeData,
)
from .agent_builder import AgentBuilder
from .tool_builder import ToolBuilder
from .model_factory import ModelFactory, DEFAULT_CALL_MODEL

logger = logging.getLogger(__name__)


class FlowBuildResult:
    """Result of building a flow."""

    def __init__(
        self,
        root_agent: Agent,
        metadata: Dict[str, Any],
        builder: "FlowBuilder",
    ):
        """
        Initialize the build result.

        Args:
            root_agent: The entry point agent
            metadata: Additional flow metadata
            builder: Reference to the builder for cleanup
        """
        self.root_agent = root_agent
        self.metadata = metadata
        self._builder = builder

    @property
    def trigger_type(self) -> str:
        """Get the trigger type (call_start, chat_start, webhook, manual)."""
        return self.metadata.get("trigger_type", "manual")

    @property
    def is_live(self) -> bool:
        """Check if this flow requires live/streaming."""
        return self.metadata.get("is_live", False)

    @property
    def flow_id(self) -> str:
        """Get the flow ID."""
        return self.metadata.get("flow_id", "")

    @property
    def flow_name(self) -> str:
        """Get the flow name."""
        return self.metadata.get("flow_name", "")

    @property
    def chat_enabled(self) -> bool:
        """Check if chat is enabled (always True - chat is default)."""
        return self.metadata.get("chat_enabled", True)

    @property
    def call_enabled(self) -> bool:
        """Check if call is enabled (only if call_start trigger exists)."""
        return self.metadata.get("call_enabled", False)

    @property
    def call_model(self) -> Optional[str]:
        """Get the model used for calls (gemini-2.0-flash-exp if call enabled)."""
        return self.metadata.get("call_model")

    async def cleanup(self) -> None:
        """Cleanup resources (MCP connections, etc.)."""
        await self._builder.cleanup()


class FlowBuilder:
    """
    Builds a complete ADK agent hierarchy from a flow definition.

    This is the main entry point for converting flow JSON into executable
    ADK agents.

    Example:
        builder = FlowBuilder(
            tenant_id="tenant_123",
            api_keys={"google": "...", "openai": "..."},
        )

        result = await builder.build(flow_schema)
        root_agent = result.root_agent

        # When done
        await result.cleanup()
    """

    def __init__(
        self,
        tenant_id: str,
        api_keys: Dict[str, str],
        custom_functions: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize the flow builder.

        Args:
            tenant_id: The tenant identifier
            api_keys: Dict of API keys by provider (google, openai, anthropic, etc.)
            custom_functions: Optional dict of custom function implementations
        """
        self.tenant_id = tenant_id
        self.api_keys = api_keys
        self.model_factory = ModelFactory(api_keys)
        self.tool_builder = ToolBuilder(
            tenant_id=tenant_id,
            custom_functions=custom_functions,
        )
        self.agent_builder = AgentBuilder(
            tenant_id=tenant_id,
            model_factory=self.model_factory,
            tool_builder=self.tool_builder,
        )

    async def build(self, flow: FlowSchema) -> FlowBuildResult:
        """
        Build the complete agent hierarchy from a flow.

        Args:
            flow: The validated flow schema

        Returns:
            FlowBuildResult containing root agent and metadata
        """
        # Validate flow first
        errors = flow.validate_flow()
        if errors:
            raise ValueError(f"Invalid flow: {'; '.join(errors)}")

        # 1. Categorize nodes by type
        nodes_by_type = self._categorize_nodes(flow.nodes)
        
        # 1b. Extract API keys from chatmodel nodes (if present)
        self._extract_api_keys_from_chatmodel_nodes(nodes_by_type)

        # 2. Build connection graph
        connections = self._build_connection_graph(flow)

        # 3. Build all tools first
        tools_by_node = await self._build_all_tools(nodes_by_type)

        # 4. Find the entry point (trigger node -> first agent)
        trigger_node, entry_agent_node = self._find_entry_point(
            flow.nodes,
            connections,
            nodes_by_type,
        )
        
        # 4b. If this is a CALL flow, override model to Gemini Live
        # (Only Gemini Live supports native audio for voice calls)
        is_call_flow = trigger_node and trigger_node.type == NodeType.CALL_START
        if is_call_flow:
            self._override_models_for_call(nodes_by_type)

        # 5. Map tools to their connected agents
        agent_tools = self._map_tools_to_agents(
            nodes_by_type,
            connections,
            tools_by_node,
        )

        # 6. Build agents with their tools and sub-agents
        root_agent = self._build_agent_hierarchy(
            entry_agent_node,
            connections,
            agent_tools,
            nodes_by_type,
        )

        # 7. Prepare metadata
        # Chat is ALWAYS enabled (default functionality)
        # Call is ONLY enabled if there's a call_start trigger
        has_call_trigger = NodeType.CALL_START in nodes_by_type
        
        metadata = {
            "trigger_type": trigger_node.type.value if trigger_node else "chat",  # Default to chat
            "trigger_data": (
                trigger_node.data.model_dump()
                if trigger_node and hasattr(trigger_node.data, "model_dump")
                else None
            ),
            "is_live": self._is_live_flow(nodes_by_type),
            "flow_id": flow.id,
            "flow_name": flow.flow_name,
            "tenant_id": self.tenant_id,
            "agent_count": len(nodes_by_type.get(NodeType.AGENT, [])),
            "tool_count": len(tools_by_node),
            # Chat is always available, Call only if trigger exists
            "chat_enabled": True,  # Chat is ALWAYS enabled (default)
            "call_enabled": has_call_trigger,  # Call only if call_start node exists
            "call_model": DEFAULT_CALL_MODEL if has_call_trigger else None,
        }

        logger.info(
            f"Built flow '{flow.flow_name}' with {metadata['agent_count']} agents "
            f"and {metadata['tool_count']} tools"
        )

        return FlowBuildResult(
            root_agent=root_agent,
            metadata=metadata,
            builder=self,
        )

    def _categorize_nodes(
        self,
        nodes: List[FlowNode],
    ) -> Dict[NodeType, List[FlowNode]]:
        """Group nodes by their type."""
        result: Dict[NodeType, List[FlowNode]] = {}
        for node in nodes:
            if node.type not in result:
                result[node.type] = []
            result[node.type].append(node)
        return result

    def _extract_api_keys_from_chatmodel_nodes(
        self,
        nodes_by_type: Dict[NodeType, List[FlowNode]],
    ) -> None:
        """
        Extract API keys from chatmodel nodes and add them to api_keys.
        
        This allows users to embed API keys directly in the flow (though
        it's recommended to store them in tenant settings instead).
        """
        chatmodel_nodes = nodes_by_type.get(NodeType.CHATMODEL, [])
        
        for node in chatmodel_nodes:
            if isinstance(node.data, ChatModelNodeData):
                data: ChatModelNodeData = node.data
                
                if data.apiKey:
                    # Map model name to provider
                    model_name_lower = data.modelName.lower()
                    
                    if "openai" in model_name_lower or "gpt" in model_name_lower:
                        if not self.api_keys.get("openai"):
                            self.api_keys["openai"] = data.apiKey
                            logger.debug("Extracted OpenAI API key from chatmodel node")
                    elif "anthropic" in model_name_lower or "claude" in model_name_lower:
                        if not self.api_keys.get("anthropic"):
                            self.api_keys["anthropic"] = data.apiKey
                            logger.debug("Extracted Anthropic API key from chatmodel node")
                    elif "google" in model_name_lower or "gemini" in model_name_lower:
                        if not self.api_keys.get("google"):
                            self.api_keys["google"] = data.apiKey
                            logger.debug("Extracted Google API key from chatmodel node")
                    elif "grok" in model_name_lower:
                        if not self.api_keys.get("xai"):
                            self.api_keys["xai"] = data.apiKey
                            logger.debug("Extracted Grok/xAI API key from chatmodel node")
                    elif "groq" in model_name_lower:
                        if not self.api_keys.get("groq"):
                            self.api_keys["groq"] = data.apiKey
                            logger.debug("Extracted Groq API key from chatmodel node")
                    elif "deepseek" in model_name_lower:
                        if not self.api_keys.get("deepseek"):
                            self.api_keys["deepseek"] = data.apiKey
                            logger.debug("Extracted Deepseek API key from chatmodel node")
                    
                    # Update model factory with new keys
                    self.model_factory = ModelFactory(self.api_keys)

    def _build_connection_graph(
        self,
        flow: FlowSchema,
    ) -> Dict[str, List[str]]:
        """
        Build adjacency list of connections.

        Returns:
            Dict mapping source node ID to list of target node IDs
        """
        graph: Dict[str, List[str]] = {}
        for conn in flow.connections:
            if conn.source not in graph:
                graph[conn.source] = []
            graph[conn.source].append(conn.target)
        return graph

    async def _build_all_tools(
        self,
        nodes_by_type: Dict[NodeType, List[FlowNode]],
    ) -> Dict[str, List[Any]]:
        """Build all tools and return mapping of node_id to tools."""
        tools_by_node: Dict[str, List[Any]] = {}

        # Build database tools
        for node in nodes_by_type.get(NodeType.DATABASE, []):
            if isinstance(node.data, DatabaseNodeData):
                tool = await self.tool_builder.build_database_tool(node.id, node.data)
                if tool:
                    tools_by_node[node.id] = [tool]

        # Build MCP tools
        for node in nodes_by_type.get(NodeType.MCP, []):
            if isinstance(node.data, MCPNodeData):
                tool = await self.tool_builder.build_mcp_tool(node.id, node.data)
                if tool:
                    tools_by_node[node.id] = [tool]

        # Build custom/function tools
        for node in nodes_by_type.get(NodeType.TOOL, []):
            if isinstance(node.data, ToolNodeData):
                tool = self.tool_builder.build_function_tool(node.id, node.data)
                if tool:
                    tools_by_node[node.id] = [tool]

        # Build code executor tools
        for node in nodes_by_type.get(NodeType.CODE, []):
            if isinstance(node.data, CodeNodeData):
                tool = self.tool_builder.build_code_executor_tool(node.id, node.data)
                if tool:
                    tools_by_node[node.id] = [tool]

        # Build HTTP request tools
        for node in nodes_by_type.get(NodeType.HTTP_REQUEST, []):
            if isinstance(node.data, HttpRequestNodeData):
                tool = self.tool_builder.build_http_request_tool(node.id, node.data)
                if tool:
                    tools_by_node[node.id] = [tool]

        return tools_by_node

    def _find_entry_point(
        self,
        nodes: List[FlowNode],
        connections: Dict[str, List[str]],
        nodes_by_type: Dict[NodeType, List[FlowNode]],
    ) -> Tuple[Optional[FlowNode], FlowNode]:
        """
        Find the trigger node and first agent node.

        Returns:
            Tuple of (trigger_node, entry_agent_node)
        """
        # Find trigger nodes (in priority order)
        trigger_types = [
            NodeType.CALL_START,
            NodeType.CHAT_START,
            NodeType.WEBHOOK,
        ]

        trigger_node = None
        for trigger_type in trigger_types:
            nodes_of_type = nodes_by_type.get(trigger_type, [])
            if nodes_of_type:
                trigger_node = nodes_of_type[0]
                break

        # If we found a trigger, find the connected agent
        if trigger_node:
            connected_ids = connections.get(trigger_node.id, [])
            for node in nodes:
                if node.id in connected_ids and node.type == NodeType.AGENT:
                    return trigger_node, node

        # If no trigger or no connected agent, find first agent node
        agent_nodes = nodes_by_type.get(NodeType.AGENT, [])
        if not agent_nodes:
            raise ValueError("No agent node found in flow")

        return trigger_node, agent_nodes[0]

    def _map_tools_to_agents(
        self,
        nodes_by_type: Dict[NodeType, List[FlowNode]],
        connections: Dict[str, List[str]],
        tools_by_node: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:
        """
        Map tools to the agents that should use them.

        An agent uses tools that are connected to it.

        Returns:
            Dict mapping agent node_id to list of tools
        """
        agent_tools: Dict[str, List[Any]] = {}

        # Build reverse connection map (target -> sources)
        reverse_connections: Dict[str, List[str]] = {}
        for source, targets in connections.items():
            for target in targets:
                if target not in reverse_connections:
                    reverse_connections[target] = []
                reverse_connections[target].append(source)

        # For each agent, collect tools connected to it
        for agent_node in nodes_by_type.get(NodeType.AGENT, []):
            agent_id = agent_node.id
            agent_tools[agent_id] = []

            # Check outgoing connections (agent -> tool)
            for target_id in connections.get(agent_id, []):
                if target_id in tools_by_node:
                    agent_tools[agent_id].extend(tools_by_node[target_id])

            # Check incoming connections (tool -> agent)
            for source_id in reverse_connections.get(agent_id, []):
                if source_id in tools_by_node:
                    agent_tools[agent_id].extend(tools_by_node[source_id])

        return agent_tools

    def _build_agent_hierarchy(
        self,
        entry_node: FlowNode,
        connections: Dict[str, List[str]],
        agent_tools: Dict[str, List[Any]],
        nodes_by_type: Dict[NodeType, List[FlowNode]],
        visited: Optional[set] = None,
    ) -> Agent:
        """
        Recursively build agent hierarchy following connections.

        Args:
            entry_node: The current agent node to build
            connections: Connection graph
            agent_tools: Mapping of agent_id to tools
            nodes_by_type: Nodes categorized by type
            visited: Set of visited node IDs (for cycle detection)

        Returns:
            The built Agent instance
        """
        if visited is None:
            visited = set()

        # Check for cycles
        if entry_node.id in visited:
            # Return existing agent to avoid infinite recursion
            existing = self.agent_builder.get_agent(entry_node.id)
            if existing:
                return existing
            raise ValueError(f"Circular reference detected at node {entry_node.id}")

        visited.add(entry_node.id)

        # Build node lookup
        all_nodes: Dict[str, FlowNode] = {}
        for type_nodes in nodes_by_type.values():
            for node in type_nodes:
                all_nodes[node.id] = node

        # Find connected sub-agents
        sub_agents: List[Agent] = []
        connected_ids = connections.get(entry_node.id, [])

        for target_id in connected_ids:
            target_node = all_nodes.get(target_id)
            if target_node and target_node.type == NodeType.AGENT:
                # Recursively build sub-agent
                sub_agent = self._build_agent_hierarchy(
                    target_node,
                    connections,
                    agent_tools,
                    nodes_by_type,
                    visited.copy(),  # Copy to allow parallel branches
                )
                sub_agents.append(sub_agent)

        # Get tools for this agent
        tools = agent_tools.get(entry_node.id, [])

        # Build this agent
        return self.agent_builder.build_agent(
            node=entry_node,
            tools=tools,
            sub_agents=sub_agents if sub_agents else None,
        )

    def _is_live_flow(
        self,
        nodes_by_type: Dict[NodeType, List[FlowNode]],
    ) -> bool:
        """
        Check if this flow requires live/streaming.

        A flow is "live" if:
        - It has a call_start trigger, OR
        - Any agent uses a live-capable model

        Returns:
            True if the flow requires live streaming
        """
        # Check for call_start trigger
        if NodeType.CALL_START in nodes_by_type:
            return True

        # Check for agents using live models
        for node in nodes_by_type.get(NodeType.AGENT, []):
            if isinstance(node.data, AgentNodeData):
                if self.model_factory.is_live_model(node.data.model):
                    return True

        return False

    def _override_models_for_call(
        self,
        nodes_by_type: Dict[NodeType, List[FlowNode]],
    ) -> None:
        """
        Override agent models to Gemini Live for call flows.
        
        Only Gemini Live supports native audio, so we force all agents
        in a call flow to use it regardless of what the user selected.
        """
        for node in nodes_by_type.get(NodeType.AGENT, []):
            if isinstance(node.data, AgentNodeData):
                original_model = node.data.model
                if not self.model_factory.is_live_model(original_model):
                    node.data.model = DEFAULT_CALL_MODEL
                    logger.info(
                        f"Call flow: Changed agent '{node.data.agentName}' model "
                        f"from '{original_model}' to '{DEFAULT_CALL_MODEL}' "
                        "(only Gemini Live supports voice calls)"
                    )

    async def cleanup(self) -> None:
        """Clean up resources (MCP connections, etc.)."""
        await self.tool_builder.cleanup()
        self.agent_builder.clear_cache()
        logger.debug(f"Cleaned up flow builder for tenant {self.tenant_id}")


async def build_flow_from_json(
    flow_json: Dict[str, Any],
    tenant_id: str,
    api_keys: Dict[str, str],
    custom_functions: Optional[Dict[str, Callable]] = None,
) -> FlowBuildResult:
    """
    Convenience function to build a flow from raw JSON.

    Args:
        flow_json: The flow definition as a dict
        tenant_id: The tenant identifier
        api_keys: API keys for model providers
        custom_functions: Optional custom function implementations

    Returns:
        FlowBuildResult containing the root agent and metadata

    Example:
        result = await build_flow_from_json(
            flow_json={"flow_name": "My Flow", "nodes": [...], ...},
            tenant_id="tenant_123",
            api_keys={"google": "AIza..."},
        )
        agent = result.root_agent
    """
    # Parse and validate the flow JSON
    flow = FlowSchema(
        tenant_id=tenant_id,
        **flow_json,
    )

    # Build the flow
    builder = FlowBuilder(
        tenant_id=tenant_id,
        api_keys=api_keys,
        custom_functions=custom_functions,
    )

    return await builder.build(flow)
