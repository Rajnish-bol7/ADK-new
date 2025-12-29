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
Flow Schema Definitions.

This module defines Pydantic models for validating and parsing flow JSON
definitions from the frontend flow builder UI.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


class NodeType(str, Enum):
    """Types of nodes that can appear in a flow."""

    AGENT = "agent"
    CALL_START = "call_start"
    CHAT_START = "chat_start"
    DATABASE = "database"
    MCP = "mcp"
    TOOL = "tool"
    CONDITION = "condition"
    LOOP = "loop"
    # Add more node types as needed
    WEBHOOK = "webhook"
    HTTP_REQUEST = "http_request"
    CODE = "code"
    CHATMODEL = "chatmodel"  # LLM model configuration node


class Position(BaseModel):
    """Node position in the flow canvas."""

    x: float
    y: float


class AgentNodeData(BaseModel):
    """Data for agent nodes."""

    agentName: str = "assistant"
    agentType: Literal["agent", "sequential", "parallel", "loop"] = "agent"
    model: str = "gemini-2.5-flash"
    promptDescription: str = ""
    promptInstructions: str = ""
    includeChatHistory: bool = True
    reasoningEffort: Literal["low", "medium", "high"] = "medium"
    outputFormat: Literal["text", "json"] = "text"
    # Optional: sub-agents for orchestration
    subAgents: List[str] = Field(default_factory=list)
    # Optional: temperature and other model params
    temperature: Optional[float] = None
    maxTokens: Optional[int] = None
    
    # Database behavior configuration
    # - "db_only": Only answer from connected database, say "I don't know" if not found
    # - "db_preferred": Prefer database info, fall back to general knowledge
    # - "general": Use general knowledge (ignore database restriction)
    # - "auto": Let the flow maker decide via prompt
    knowledgeSource: Literal["db_only", "db_preferred", "general", "auto"] = "auto"


class CallStartNodeData(BaseModel):
    """Data for call/voice trigger nodes."""

    callType: Literal["incoming", "outgoing"] = "incoming"
    phoneNumber: str = ""
    callerName: str = ""
    callDirection: Literal["incoming", "outgoing"] = "incoming"
    # Voice settings
    voiceId: Optional[str] = None
    language: str = "en-US"


class ChatStartNodeData(BaseModel):
    """Data for chat trigger nodes."""

    platform: str = "web"  # web, whatsapp, telegram, slack, etc.
    welcomeMessage: Optional[str] = None
    # Channel-specific settings
    channelId: Optional[str] = None


class ChatModelNodeData(BaseModel):
    """Data for chat model configuration nodes.
    
    These nodes configure which LLM provider/model to use.
    API keys should ideally be stored in tenant settings, not in the flow.
    """

    modelName: str = "OpenAI Chat Model"
    modelProvider: Literal[
        "openai", "anthropic", "google", "grok", "groq", "deepseek", "mistral", "cohere", "ollama"
    ] = "openai"
    # API key - prefer to use tenant's stored keys instead
    apiKey: str = ""
    # Model-specific settings
    temperature: Optional[float] = None
    maxTokens: Optional[int] = None
    # Specific model version (e.g., gpt-4, gpt-4o, claude-3-opus)
    modelVersion: str = ""


class DatabaseNodeData(BaseModel):
    """Data for database nodes - connects via MCP."""

    databaseType: Literal[
        "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Redis", "BigQuery"
    ]
    host: str = ""
    port: str = ""
    databaseName: str = ""
    username: str = ""
    password: str = ""  # Should be encrypted/referenced from secrets
    connectionString: str = ""
    query: str = ""
    isDisabled: bool = False

    @field_validator("port")
    @classmethod
    def set_default_port(cls, v: str, info) -> str:
        """Set default port based on database type if not provided."""
        if v:
            return v
        # Get database type from the data being validated
        db_type = info.data.get("databaseType", "")
        default_ports = {
            "MySQL": "3306",
            "PostgreSQL": "5432",
            "MongoDB": "27017",
            "SQLite": "",
            "Redis": "6379",
            "BigQuery": "",
        }
        return default_ports.get(db_type, "")


class MCPNodeData(BaseModel):
    """Data for MCP (Model Context Protocol) tool nodes.
    
    Supports two modes:
    1. Platform mode: Use a pre-configured platform (googleCalendar, slack, etc.)
    2. Custom mode: Provide MCP server URL/command directly
    """

    # Platform-based configuration (e.g., googleCalendar, slack, github)
    platform: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # Custom MCP server configuration
    mcpServerUrl: str = ""
    mcpServerCommand: str = ""
    mcpServerArgs: List[str] = Field(default_factory=list)
    toolFilter: List[str] = Field(default_factory=list)
    connectionType: Literal["stdio", "sse", "http"] = "stdio"
    # Optional: environment variables for the MCP server
    envVars: Dict[str, str] = Field(default_factory=dict)


class ToolNodeData(BaseModel):
    """Data for custom tool nodes."""

    toolName: str
    toolType: Literal["function", "openapi", "google_search", "code_executor", "custom"]
    description: str = ""
    # For OpenAPI tools
    openapiSpec: Optional[str] = None
    operationId: Optional[str] = None
    # For function tools - reference to a registered function
    functionRef: Optional[str] = None
    # Tool parameters schema (JSON Schema format)
    parametersSchema: Optional[Dict[str, Any]] = None


class ConditionNodeData(BaseModel):
    """Data for condition/branching nodes."""

    conditionType: Literal["if_else", "switch"] = "if_else"
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    # Example condition: {"field": "response", "operator": "contains", "value": "yes"}


class LoopNodeData(BaseModel):
    """Data for loop nodes."""

    loopType: Literal["for_each", "while", "count"] = "count"
    maxIterations: int = 10
    breakCondition: Optional[str] = None


class WebhookNodeData(BaseModel):
    """Data for webhook trigger nodes."""

    webhookPath: str = ""
    httpMethod: Literal["GET", "POST", "PUT", "DELETE"] = "POST"
    authType: Literal["none", "api_key", "bearer", "basic"] = "none"


class HttpRequestNodeData(BaseModel):
    """Data for HTTP request nodes."""

    url: str = ""
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET"
    headers: Dict[str, str] = Field(default_factory=dict)
    body: Optional[str] = None
    timeout: int = 30


class CodeNodeData(BaseModel):
    """Data for code execution nodes."""

    language: Literal["python", "javascript"] = "python"
    code: str = ""
    sandbox: bool = True


# Union type for all possible node data types
NodeDataType = Union[
    AgentNodeData,
    CallStartNodeData,
    ChatStartNodeData,
    ChatModelNodeData,
    DatabaseNodeData,
    MCPNodeData,
    ToolNodeData,
    ConditionNodeData,
    LoopNodeData,
    WebhookNodeData,
    HttpRequestNodeData,
    CodeNodeData,
    Dict[str, Any],  # Fallback for unknown node types
]


class FlowNode(BaseModel):
    """A single node in the flow."""

    id: str
    type: NodeType
    position: Position
    data: NodeDataType
    width: int = 200
    height: int = 120
    selected: bool = False
    positionAbsolute: Optional[Position] = None
    dragging: bool = False

    @field_validator("data", mode="before")
    @classmethod
    def parse_node_data(cls, v: Any, info) -> NodeDataType:
        """Parse node data based on node type."""
        if isinstance(v, dict):
            # Get the node type from the values being validated
            node_type_str = info.data.get("type")
            if isinstance(node_type_str, NodeType):
                node_type_str = node_type_str.value

            type_mapping = {
                "agent": AgentNodeData,
                "call_start": CallStartNodeData,
                "chat_start": ChatStartNodeData,
                "chatmodel": ChatModelNodeData,
                "database": DatabaseNodeData,
                "mcp": MCPNodeData,
                "tool": ToolNodeData,
                "condition": ConditionNodeData,
                "loop": LoopNodeData,
                "webhook": WebhookNodeData,
                "http_request": HttpRequestNodeData,
                "code": CodeNodeData,
            }

            data_class = type_mapping.get(node_type_str)
            if data_class:
                try:
                    return data_class(**v)
                except Exception:
                    # If parsing fails, return as dict
                    return v
        return v


class FlowConnection(BaseModel):
    """Connection between nodes (edge in the flow graph)."""

    id: str
    source: str  # Source node ID
    target: str  # Target node ID
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    # Optional condition for conditional routing
    condition: Optional[str] = None
    label: Optional[str] = None


class FlowSettings(BaseModel):
    """Global flow settings."""

    apiKeySource: Literal["platform", "tenant"] = "platform"
    sessionStorage: Literal["memory", "database", "redis"] = "database"
    memoryEnabled: bool = True
    maxSessionDuration: int = 3600  # seconds
    defaultModel: str = "gemini-2.5-flash"
    # Rate limiting
    maxRequestsPerMinute: int = 60
    # Logging
    logLevel: Literal["debug", "info", "warning", "error"] = "info"


class FlowSchema(BaseModel):
    """
    Complete flow definition.

    This is the main schema that validates the entire flow JSON
    received from the frontend.
    """

    id: str = Field(alias="flow_id", default="")
    tenant_id: str = ""
    flow_name: str
    description: str = ""
    nodes: List[FlowNode]
    connections: List[FlowConnection] = Field(default_factory=list, alias="edges")
    settings: FlowSettings = Field(default_factory=FlowSettings)
    version: int = 1
    is_active: bool = True
    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        populate_by_name = True

    def get_nodes_by_type(self, node_type: NodeType) -> List[FlowNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes if n.type == node_type]

    def get_node_by_id(self, node_id: str) -> Optional[FlowNode]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_connections_from(self, node_id: str) -> List[FlowConnection]:
        """Get all connections originating from a node."""
        return [c for c in self.connections if c.source == node_id]

    def get_connections_to(self, node_id: str) -> List[FlowConnection]:
        """Get all connections targeting a node."""
        return [c for c in self.connections if c.target == node_id]

    def get_entry_point(self) -> Optional[FlowNode]:
        """
        Find the entry point node (trigger node).

        Priority: call_start > chat_start > webhook > first agent
        """
        trigger_priority = [
            NodeType.CALL_START,
            NodeType.CHAT_START,
            NodeType.WEBHOOK,
        ]

        for trigger_type in trigger_priority:
            nodes = self.get_nodes_by_type(trigger_type)
            if nodes:
                return nodes[0]

        # Fallback to first agent node
        agent_nodes = self.get_nodes_by_type(NodeType.AGENT)
        return agent_nodes[0] if agent_nodes else None

    def validate_flow(self) -> List[str]:
        """
        Validate the flow for common issues.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Must have at least one agent
        agent_nodes = self.get_nodes_by_type(NodeType.AGENT)
        if not agent_nodes:
            errors.append("Flow must have at least one agent node")

        # Check for orphan nodes (nodes with no connections)
        connected_nodes = set()
        for conn in self.connections:
            connected_nodes.add(conn.source)
            connected_nodes.add(conn.target)

        # Trigger nodes and single agents don't need connections
        if len(self.nodes) > 1:
            for node in self.nodes:
                if node.id not in connected_nodes:
                    # Only warn for non-trigger nodes
                    if node.type not in (
                        NodeType.CALL_START,
                        NodeType.CHAT_START,
                        NodeType.WEBHOOK,
                    ):
                        errors.append(f"Node '{node.id}' has no connections")

        # Check for cycles (simple check)
        # TODO: Implement proper cycle detection for complex flows

        return errors
