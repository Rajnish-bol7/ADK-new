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
Agent Builder for creating ADK agents from flow node configurations.

This module handles the creation of various agent types including:
- LlmAgent (standard agents with LLM)
- SequentialAgent (runs sub-agents in sequence)
- ParallelAgent (runs sub-agents concurrently)
- LoopAgent (repeats sub-agents until condition)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from google.adk import Agent
from google.adk.agents import SequentialAgent, ParallelAgent, LoopAgent

from ..schema.flow_schema import FlowNode, AgentNodeData
from .model_factory import ModelFactory
from .tool_builder import ToolBuilder

logger = logging.getLogger(__name__)


class AgentBuilder:
    """
    Builds ADK agents from flow node configurations.

    This class handles the conversion of agent nodes into ADK Agent objects,
    supporting various agent types and configurations.

    Example:
        model_factory = ModelFactory(api_keys={"google": "..."})
        tool_builder = ToolBuilder(tenant_id="tenant_123")
        agent_builder = AgentBuilder(
            tenant_id="tenant_123",
            model_factory=model_factory,
            tool_builder=tool_builder,
        )

        agent = agent_builder.build_agent(
            node=agent_node,
            tools=[some_tool],
            sub_agents=[child_agent],
        )
    """

    def __init__(
        self,
        tenant_id: str,
        model_factory: ModelFactory,
        tool_builder: ToolBuilder,
    ):
        """
        Initialize the agent builder.

        Args:
            tenant_id: The tenant identifier
            model_factory: Factory for creating LLM instances
            tool_builder: Builder for creating tools
        """
        self.tenant_id = tenant_id
        self.model_factory = model_factory
        self.tool_builder = tool_builder
        self._built_agents: Dict[str, Agent] = {}

    def build_agent(
        self,
        node: FlowNode,
        tools: Optional[List[Any]] = None,
        sub_agents: Optional[List[Agent]] = None,
    ) -> Agent:
        """
        Build an ADK Agent from a flow node.

        Args:
            node: The agent node from the flow
            tools: List of tools to attach to this agent
            sub_agents: List of sub-agents for multi-agent orchestration

        Returns:
            An ADK Agent instance
        """
        data: AgentNodeData = node.data

        # Determine agent type and build accordingly
        agent_type = data.agentType

        if agent_type == "sequential":
            agent = self._build_sequential_agent(node, data, sub_agents)
        elif agent_type == "parallel":
            agent = self._build_parallel_agent(node, data, sub_agents)
        elif agent_type == "loop":
            agent = self._build_loop_agent(node, data, sub_agents)
        else:
            agent = self._build_llm_agent(node, data, tools, sub_agents)

        # Cache the built agent
        self._built_agents[node.id] = agent
        logger.info(f"Built agent '{data.agentName}' from node {node.id}")

        return agent

    def _build_llm_agent(
        self,
        node: FlowNode,
        data: AgentNodeData,
        tools: Optional[List[Any]],
        sub_agents: Optional[List[Agent]],
    ) -> Agent:
        """Build a standard LLM-powered agent."""
        # Check if this agent has database tools connected
        has_database = self._has_database_tools(tools)
        
        # Build instruction from prompt fields
        instruction = self._build_instruction(data, has_database=has_database)

        # Get the LLM model
        model = self.model_factory.create_model(
            model_name=data.model,
            temperature=data.temperature,
            max_tokens=data.maxTokens,
        )

        # Build agent config
        agent_kwargs: Dict[str, Any] = {
            "name": data.agentName,
            "model": model,
            "instruction": instruction,
            "description": data.promptDescription or f"Agent {data.agentName}",
        }

        # Add tools if provided
        if tools:
            agent_kwargs["tools"] = tools

        # Add sub-agents if provided
        if sub_agents:
            agent_kwargs["sub_agents"] = sub_agents

        return Agent(**agent_kwargs)

    def _has_database_tools(self, tools: Optional[List[Any]]) -> bool:
        """
        Check if any of the tools are database-related.
        
        Args:
            tools: List of tools
            
        Returns:
            True if database tools are present
        """
        if not tools:
            return False
            
        for tool in tools:
            # Check for MCP database tools or any tool with "database" in name
            tool_name = getattr(tool, 'name', '') or str(tool)
            if any(keyword in tool_name.lower() for keyword in ['database', 'db', 'sql', 'query', 'mysql', 'postgres', 'mongo']):
                return True
        
        return False

    def _build_sequential_agent(
        self,
        node: FlowNode,
        data: AgentNodeData,
        sub_agents: Optional[List[Agent]],
    ) -> SequentialAgent:
        """Build a SequentialAgent that runs sub-agents in order."""
        if not sub_agents:
            logger.warning(
                f"SequentialAgent {data.agentName} has no sub-agents, "
                "it will do nothing."
            )
            sub_agents = []

        return SequentialAgent(
            name=data.agentName,
            sub_agents=sub_agents,
            description=data.promptDescription or f"Sequential agent {data.agentName}",
        )

    def _build_parallel_agent(
        self,
        node: FlowNode,
        data: AgentNodeData,
        sub_agents: Optional[List[Agent]],
    ) -> ParallelAgent:
        """Build a ParallelAgent that runs sub-agents concurrently."""
        if not sub_agents:
            logger.warning(
                f"ParallelAgent {data.agentName} has no sub-agents, "
                "it will do nothing."
            )
            sub_agents = []

        return ParallelAgent(
            name=data.agentName,
            sub_agents=sub_agents,
            description=data.promptDescription or f"Parallel agent {data.agentName}",
        )

    def _build_loop_agent(
        self,
        node: FlowNode,
        data: AgentNodeData,
        sub_agents: Optional[List[Agent]],
    ) -> LoopAgent:
        """Build a LoopAgent that repeats sub-agents until a condition."""
        if not sub_agents:
            logger.warning(
                f"LoopAgent {data.agentName} has no sub-agents, " "it will do nothing."
            )
            sub_agents = []

        return LoopAgent(
            name=data.agentName,
            sub_agents=sub_agents,
            max_iterations=10,  # Could make this configurable
        )

    def _build_instruction(
        self,
        data: AgentNodeData,
        has_database: bool = False,
    ) -> str:
        """
        Build agent instruction from prompt fields.

        Args:
            data: Agent node data containing prompt fields
            has_database: Whether this agent has database tools connected

        Returns:
            Formatted instruction string
        """
        parts: List[str] = []

        # Add main description
        if data.promptDescription:
            parts.append(data.promptDescription)

        # Add specific instructions
        if data.promptInstructions:
            parts.append(f"\n\nInstructions:\n{data.promptInstructions}")

        # Add database behavior instruction based on knowledgeSource setting
        if has_database and data.knowledgeSource != "auto":
            db_instruction = self._get_database_instruction(data.knowledgeSource)
            if db_instruction:
                parts.append(f"\n\n{db_instruction}")

        # Add output format instruction
        if data.outputFormat == "json":
            parts.append(
                "\n\nIMPORTANT: Always respond with valid JSON format only. "
                "Do not include any text outside of the JSON structure."
            )

        # Default instruction if nothing provided
        if not parts:
            return "You are a helpful assistant."

        return "\n".join(parts)

    def _get_database_instruction(self, knowledge_source: str) -> str:
        """
        Get the instruction text for database behavior.
        
        Args:
            knowledge_source: The knowledge source setting
            
        Returns:
            Instruction text for the specified behavior
        """
        if knowledge_source == "db_only":
            return (
                "IMPORTANT - DATABASE ONLY MODE:\n"
                "You MUST only answer questions using information from the connected database.\n"
                "- Always query the database first before answering.\n"
                "- If the information is not in the database, respond with: "
                "'I don't have that information in my database.'\n"
                "- Do NOT use your general knowledge to answer questions.\n"
                "- Do NOT make up or assume information that isn't in the database."
            )
        elif knowledge_source == "db_preferred":
            return (
                "DATABASE PREFERRED MODE:\n"
                "- First, try to find the answer in the connected database.\n"
                "- If the information is found in the database, use that.\n"
                "- Only if the database doesn't have the information, you may use your general knowledge.\n"
                "- Always clearly indicate when using general knowledge vs database info."
            )
        elif knowledge_source == "general":
            return (
                "GENERAL KNOWLEDGE MODE:\n"
                "- You can use your general knowledge to answer questions.\n"
                "- The database is available as a supplementary tool when needed.\n"
                "- Use your judgment on when to query the database vs using general knowledge."
            )
        
        return ""  # For "auto", let the flow maker control via prompt

    def get_agent(self, node_id: str) -> Optional[Agent]:
        """
        Get a previously built agent by node ID.

        Args:
            node_id: The node identifier

        Returns:
            The Agent if found, None otherwise
        """
        return self._built_agents.get(node_id)

    def get_all_agents(self) -> Dict[str, Agent]:
        """
        Get all built agents.

        Returns:
            Dict mapping node IDs to Agent instances
        """
        return self._built_agents.copy()

    def clear_cache(self) -> None:
        """Clear the agent cache."""
        self._built_agents.clear()
        logger.debug("Cleared agent cache")


def create_agent_from_config(
    config: Dict[str, Any],
    api_keys: Dict[str, str],
) -> Agent:
    """
    Convenience function to create a single agent from a config dict.

    This is useful for simple cases where you just need one agent.

    Args:
        config: Agent configuration dict with fields:
            - name: Agent name
            - model: Model name
            - instruction: Agent instruction
            - tools: Optional list of tools
        api_keys: API keys for the model provider

    Returns:
        An ADK Agent instance

    Example:
        agent = create_agent_from_config(
            config={
                "name": "my_agent",
                "model": "gemini-2.5-flash",
                "instruction": "You are a helpful assistant.",
            },
            api_keys={"google": "AIza..."},
        )
    """
    model_factory = ModelFactory(api_keys)
    model = model_factory.create_model(
        model_name=config.get("model", "gemini-2.5-flash"),
        temperature=config.get("temperature"),
        max_tokens=config.get("max_tokens"),
    )

    return Agent(
        name=config.get("name", "agent"),
        model=model,
        instruction=config.get("instruction", "You are a helpful assistant."),
        description=config.get("description", ""),
        tools=config.get("tools", []),
    )
