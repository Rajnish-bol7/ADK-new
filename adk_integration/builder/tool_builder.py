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
Tool Builder for creating ADK tools from flow node configurations.

This module handles the creation of various tool types including:
- Database tools (via MCP)
- MCP toolsets
- Function tools
- OpenAPI tools
- Built-in tools (Google Search, Code Executor)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from google.adk.tools import FunctionTool
from google.adk.tools.base_tool import BaseTool

from ..schema.flow_schema import (
    DatabaseNodeData,
    MCPNodeData,
    ToolNodeData,
    CodeNodeData,
    HttpRequestNodeData,
)

logger = logging.getLogger(__name__)


# Default MCP server commands for different database types
DEFAULT_MCP_SERVERS: Dict[str, Dict[str, Any]] = {
    "MySQL": {
        "command": "python",
        "args": ["-m", "mcp_mysql_server"],
    },
    "PostgreSQL": {
        "command": "python",
        "args": ["-m", "mcp_postgres_server"],
    },
    "MongoDB": {
        "command": "python",
        "args": ["-m", "mcp_mongodb_server"],
    },
    "SQLite": {
        "command": "python",
        "args": ["-m", "mcp_sqlite_server"],
    },
    "Redis": {
        "command": "python",
        "args": ["-m", "mcp_redis_server"],
    },
    "BigQuery": {
        "command": "python",
        "args": ["-m", "mcp_bigquery_server"],
    },
}

# MCP server configurations for popular platforms
PLATFORM_MCP_SERVERS: Dict[str, Dict[str, Any]] = {
    "googleCalendar": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-google-calendar"],
        "env_keys": ["apiKey", "calendarId"],
        "env_mapping": {
            "apiKey": "GOOGLE_API_KEY",
            "calendarId": "GOOGLE_CALENDAR_ID",
        },
    },
    "slack": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-slack"],
        "env_keys": ["token"],
        "env_mapping": {
            "token": "SLACK_BOT_TOKEN",
        },
    },
    "github": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-github"],
        "env_keys": ["token"],
        "env_mapping": {
            "token": "GITHUB_TOKEN",
        },
    },
    "gmail": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-gmail"],
        "env_keys": ["credentials"],
        "env_mapping": {
            "credentials": "GMAIL_CREDENTIALS",
        },
    },
    "notion": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-notion"],
        "env_keys": ["token"],
        "env_mapping": {
            "token": "NOTION_TOKEN",
        },
    },
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-filesystem"],
        "env_keys": ["path"],
    },
    "puppeteer": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-puppeteer"],
    },
    "memory": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-memory"],
    },
}


class ToolBuilder:
    """
    Builds ADK tools from flow node configurations.

    This class handles the conversion of tool-related nodes (database, MCP, tool)
    into ADK-compatible tool objects.

    Example:
        builder = ToolBuilder(tenant_id="tenant_123")

        # Build a database tool
        db_tool = await builder.build_database_tool(
            node_id="node_1",
            data=DatabaseNodeData(databaseType="PostgreSQL", ...)
        )

        # Don't forget to cleanup when done
        await builder.cleanup()
    """

    def __init__(
        self,
        tenant_id: str,
        custom_functions: Optional[Dict[str, Callable]] = None,
        mcp_server_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize the tool builder.

        Args:
            tenant_id: The tenant identifier for isolation
            custom_functions: Optional dict of custom function implementations
            mcp_server_overrides: Optional overrides for MCP server commands
        """
        self.tenant_id = tenant_id
        self.custom_functions = custom_functions or {}
        self.mcp_servers = {**DEFAULT_MCP_SERVERS, **(mcp_server_overrides or {})}
        self._mcp_toolsets: List[Any] = []  # Track for cleanup

    async def build_database_tool(
        self,
        node_id: str,
        data: DatabaseNodeData,
    ) -> Optional[Any]:
        """
        Create an MCP toolset for database access.

        Args:
            node_id: The node identifier
            data: Database node configuration

        Returns:
            McpToolset instance or None if disabled
        """
        if data.isDisabled:
            logger.debug(f"Database node {node_id} is disabled, skipping")
            return None

        try:
            # Import MCP tools (may not be installed)
            from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
        except ImportError:
            logger.error("MCP tools not installed. Install with: pip install mcp")
            return None

        # Get MCP server config for this database type
        server_config = self.mcp_servers.get(data.databaseType)
        if not server_config:
            logger.error(f"No MCP server configured for {data.databaseType}")
            return None

        # Build connection arguments
        args = list(server_config.get("args", []))

        # Add connection parameters
        if data.host:
            args.extend(["--host", data.host])
        if data.port:
            args.extend(["--port", data.port])
        if data.databaseName:
            args.extend(["--database", data.databaseName])
        if data.username:
            args.extend(["--user", data.username])
        if data.connectionString:
            args.extend(["--connection-string", data.connectionString])

        # Build environment variables (for sensitive data like password)
        env: Dict[str, str] = {}
        if data.password:
            env["DB_PASSWORD"] = data.password

        try:
            connection_params = StdioConnectionParams(
                command=server_config["command"],
                args=args,
                env=env if env else None,
            )

            toolset = McpToolset(
                connection_params=connection_params,
                tool_name_prefix=f"{node_id}_",  # Prefix to avoid conflicts
            )

            self._mcp_toolsets.append(toolset)
            logger.info(f"Created database toolset for node {node_id}")
            return toolset

        except Exception as e:
            logger.error(f"Failed to create database toolset for {node_id}: {e}")
            return None

    async def build_mcp_tool(
        self,
        node_id: str,
        data: MCPNodeData,
    ) -> Optional[Any]:
        """
        Create an MCP toolset from node configuration.

        Supports two modes:
        1. Platform mode: Use a pre-configured platform (googleCalendar, slack, etc.)
        2. Custom mode: Provide MCP server URL/command directly

        Args:
            node_id: The node identifier
            data: MCP node configuration

        Returns:
            McpToolset instance or None on error
        """
        try:
            from google.adk.tools.mcp_tool import (
                McpToolset,
                StdioConnectionParams,
                SseConnectionParams,
                StreamableHTTPConnectionParams,
            )
        except ImportError:
            logger.error("MCP tools not installed")
            return None

        try:
            # Check if this is a platform-based MCP node
            if data.platform:
                return await self._build_platform_mcp_tool(node_id, data)
            
            # Build connection params based on type (custom mode)
            if data.connectionType == "stdio":
                if not data.mcpServerCommand:
                    logger.error(f"MCP node {node_id} missing server command")
                    return None

                connection_params = StdioConnectionParams(
                    command=data.mcpServerCommand,
                    args=data.mcpServerArgs,
                    env=data.envVars if data.envVars else None,
                )

            elif data.connectionType == "sse":
                if not data.mcpServerUrl:
                    logger.error(f"MCP node {node_id} missing server URL")
                    return None

                connection_params = SseConnectionParams(
                    url=data.mcpServerUrl,
                )

            elif data.connectionType == "http":
                if not data.mcpServerUrl:
                    logger.error(f"MCP node {node_id} missing server URL")
                    return None

                connection_params = StreamableHTTPConnectionParams(
                    url=data.mcpServerUrl,
                )

            else:
                logger.error(f"Unknown MCP connection type: {data.connectionType}")
                return None

            # Create toolset
            tool_filter = data.toolFilter if data.toolFilter else None

            toolset = McpToolset(
                connection_params=connection_params,
                tool_filter=tool_filter,
                tool_name_prefix=f"{node_id}_",
            )

            self._mcp_toolsets.append(toolset)
            logger.info(f"Created MCP toolset for node {node_id}")
            return toolset

        except Exception as e:
            logger.error(f"Failed to create MCP toolset for {node_id}: {e}")
            return None

    async def _build_platform_mcp_tool(
        self,
        node_id: str,
        data: MCPNodeData,
    ) -> Optional[Any]:
        """
        Build an MCP tool from a platform configuration.
        
        Args:
            node_id: The node identifier
            data: MCP node configuration with platform set
            
        Returns:
            McpToolset instance or None on error
        """
        try:
            from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
        except ImportError:
            logger.error("MCP tools not installed")
            return None
        
        platform = data.platform
        platform_config = PLATFORM_MCP_SERVERS.get(platform)
        
        if not platform_config:
            logger.error(f"Unknown MCP platform: {platform}. Supported: {list(PLATFORM_MCP_SERVERS.keys())}")
            return None
        
        try:
            # Build command and args from platform config
            command = platform_config.get("command", "npx")
            args = list(platform_config.get("args", []))
            
            # Build environment variables from config
            env: Dict[str, str] = {}
            env_mapping = platform_config.get("env_mapping", {})
            
            for config_key, env_var in env_mapping.items():
                if config_key in data.config:
                    env[env_var] = str(data.config[config_key])
            
            # Add any extra config values not in mapping as environment variables
            for key, value in data.config.items():
                if key not in env_mapping and value:
                    # Convert camelCase to UPPER_SNAKE_CASE
                    env_key = ''.join(['_' + c if c.isupper() else c.upper() for c in key]).lstrip('_')
                    env[env_key] = str(value)
            
            connection_params = StdioConnectionParams(
                command=command,
                args=args,
                env=env if env else None,
            )
            
            toolset = McpToolset(
                connection_params=connection_params,
                tool_name_prefix=f"{node_id}_{platform}_",
            )
            
            self._mcp_toolsets.append(toolset)
            logger.info(f"Created {platform} MCP toolset for node {node_id}")
            return toolset
            
        except Exception as e:
            logger.error(f"Failed to create {platform} MCP toolset for {node_id}: {e}")
            return None

    def build_function_tool(
        self,
        node_id: str,
        data: ToolNodeData,
    ) -> Optional[BaseTool]:
        """
        Create a function tool from node configuration.

        Args:
            node_id: The node identifier
            data: Tool node configuration

        Returns:
            Tool instance or None on error
        """
        try:
            # Built-in tools
            if data.toolType == "google_search":
                from google.adk.tools import google_search

                return google_search

            if data.toolType == "code_executor":
                from google.adk.code_executors import BuiltInCodeExecutor

                return BuiltInCodeExecutor()

            # OpenAPI tools
            if data.toolType == "openapi" and data.openapiSpec:
                return self._build_openapi_tool(node_id, data)

            # Custom function tools
            if data.toolType == "function":
                return self._build_custom_function_tool(node_id, data)

            logger.warning(f"Could not build tool for node {node_id}: {data.toolName}")
            return None

        except Exception as e:
            logger.error(f"Failed to build function tool for {node_id}: {e}")
            return None

    def _build_openapi_tool(
        self,
        node_id: str,
        data: ToolNodeData,
    ) -> Optional[Any]:
        """Build a tool from OpenAPI specification."""
        try:
            from google.adk.tools.openapi_tool import OpenAPIToolset

            toolset = OpenAPIToolset(
                spec_str=data.openapiSpec,
                tool_filter=[data.operationId] if data.operationId else None,
            )

            logger.info(f"Created OpenAPI toolset for node {node_id}")
            return toolset

        except ImportError:
            logger.error("OpenAPI tools not available")
            return None
        except Exception as e:
            logger.error(f"Failed to create OpenAPI tool: {e}")
            return None

    def _build_custom_function_tool(
        self,
        node_id: str,
        data: ToolNodeData,
    ) -> Optional[FunctionTool]:
        """Build a custom function tool."""
        # Look up the function by reference
        func_ref = data.functionRef or data.toolName

        if func_ref in self.custom_functions:
            func = self.custom_functions[func_ref]
            return FunctionTool(func)

        logger.warning(f"Custom function '{func_ref}' not found for node {node_id}")
        return None

    def build_http_request_tool(
        self,
        node_id: str,
        data: HttpRequestNodeData,
    ) -> Optional[FunctionTool]:
        """
        Create an HTTP request tool.

        Args:
            node_id: The node identifier
            data: HTTP request node configuration

        Returns:
            FunctionTool for making HTTP requests
        """
        import httpx

        async def make_http_request(
            url: str = data.url,
            method: str = data.method,
            headers: Dict[str, str] = None,
            body: str = None,
        ) -> Dict[str, Any]:
            """Make an HTTP request."""
            async with httpx.AsyncClient(timeout=data.timeout) as client:
                request_headers = {**data.headers, **(headers or {})}
                response = await client.request(
                    method=method,
                    url=url,
                    headers=request_headers,
                    content=body,
                )
                return {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.text,
                }

        return FunctionTool(make_http_request)

    def build_code_executor_tool(
        self,
        node_id: str,
        data: CodeNodeData,
    ) -> Optional[Any]:
        """
        Create a code executor tool.

        Args:
            node_id: The node identifier
            data: Code node configuration

        Returns:
            Code executor instance
        """
        try:
            if data.sandbox:
                from google.adk.code_executors import BuiltInCodeExecutor

                return BuiltInCodeExecutor()
            else:
                # Use unsafe executor only if explicitly requested
                from google.adk.code_executors import UnsafeLocalCodeExecutor

                logger.warning(
                    f"Using unsafe code executor for node {node_id}. "
                    "This should not be used in production!"
                )
                return UnsafeLocalCodeExecutor()

        except ImportError as e:
            logger.error(f"Code executor not available: {e}")
            return None

    async def cleanup(self) -> None:
        """Close all MCP connections and cleanup resources."""
        for toolset in self._mcp_toolsets:
            try:
                if hasattr(toolset, "close"):
                    await toolset.close()
            except Exception as e:
                logger.warning(f"Error closing MCP toolset: {e}")

        self._mcp_toolsets.clear()
        logger.debug(f"Cleaned up tool builder for tenant {self.tenant_id}")

    def register_custom_function(
        self,
        name: str,
        func: Callable,
    ) -> None:
        """
        Register a custom function that can be used as a tool.

        Args:
            name: Function name/reference
            func: The function implementation
        """
        self.custom_functions[name] = func
        logger.debug(f"Registered custom function: {name}")
