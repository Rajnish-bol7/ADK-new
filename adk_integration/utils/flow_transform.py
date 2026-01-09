"""
Flow format transformation utilities.

Converts between React Flow format (frontend) and n8n-like format (backend/API).
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def transform_react_flow_to_n8n(
    react_flow_json: Dict[str, Any],
    flow_name: str,
    flow_id: str,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transform React Flow JSON to n8n-like format.
    
    Args:
        react_flow_json: React Flow format JSON from frontend
        flow_name: Name of the flow
        flow_id: UUID of the flow
        tenant_id: Optional tenant ID for credential references
        user_id: Optional user ID who created the flow
    
    Returns:
        n8n-like format JSON
    """
    nodes = react_flow_json.get("nodes", [])
    edges = react_flow_json.get("edges", [])
    
    if not nodes:
        logger.warning(f"Empty nodes array in React Flow JSON for flow {flow_id}")
        return {
            "flowId": flow_id,
            "flowName": flow_name,
            "tenantId": tenant_id or "",
            "userId": user_id or "",
            "nodes": [],
            "connections": {},
            "settings": {
                "saveExecutionProgress": True,
                "saveManualExecutions": True,
                "saveDataErrorExecution": "all",
                "saveDataSuccessExecution": "all",
                "executionTimeout": 3600,
                "timezone": "UTC",
                "executionOrder": "v1"
            },
            "active": True,
            "meta": {
                "templateCredsSetupCompleted": False
            }
        }
    
    # Build node lookup for name resolution
    node_lookup = {}
    for node in nodes:
        node_lookup[node["id"]] = node
    
    # Transform nodes
    n8n_nodes = []
    for node in nodes:
        node_data = node.get("data", {})
        
        # Extract node name from data - use user-provided name or generate based on type
        node_name = (
            node_data.get("agentName") or
            node_data.get("modelName") or
            node_data.get("callAgentName") or
            node_data.get("toolName") or
            None  # Don't fallback to ID yet
        )
        
        # If no user-provided name, generate one based on type and type-specific field
        if not node_name:
            node_type = node["type"]
            
            # Generate name based on node type and its specific field
            if node_type == "agent":
                agent_type = node_data.get("agentType", "agent")
                node_name = f"{node_type}.{agent_type}"
            elif node_type == "call":
                # For call nodes, use "call" if no callAgentName provided
                node_name = "call"
            elif node_type == "chatmodel":
                provider = node_data.get("modelProvider", "openai")
                node_name = f"{node_type}.{provider}"
            elif node_type == "database":
                db_type = node_data.get("databaseType", "Database")
                node_name = f"{node_type}.{db_type}"
            elif node_type == "tool":
                tool_type = node_data.get("toolType", "tool")
                node_name = f"{node_type}.{tool_type}"
            elif node_type == "condition":
                condition_type = node_data.get("conditionType", "if_else")
                node_name = f"{node_type}.{condition_type}"
            elif node_type == "loop":
                loop_type = node_data.get("loopType", "count")
                node_name = f"{node_type}.{loop_type}"
            elif node_type == "mcp":
                platform = node_data.get("platform") or node_data.get("mcpServerUrl")
                if platform:
                    node_name = f"{node_type}.{platform}"
                else:
                    node_name = "mcp"
            elif node_type == "webhook":
                webhook_path = node_data.get("webhookPath")
                if webhook_path:
                    node_name = f"{node_type}.{webhook_path}"
                else:
                    node_name = "webhook"
            elif node_type == "http_request":
                method = node_data.get("method", "GET")
                node_name = f"{node_type}.{method}"
            elif node_type == "code":
                language = node_data.get("language", "python")
                node_name = f"{node_type}.{language}"
            else:
                # Final fallback: use node type if we can't determine a better name
                node_name = node_type
        
        # Build n8n node
        n8n_node = {
            "id": node["id"],
            "name": node_name,
            "type": node['type'],  # No prefix - just the node type
            "typeVersion": 1,
            "position": [node["position"]["x"], node["position"]["y"]],
            "parameters": {},
            "disabled": node_data.get("isDisabled", False),
        }
        
        # Copy all data fields to parameters (except those that go elsewhere)
        excluded_params = {"apiKey", "isDisabled"}  # Handle separately
        for key, value in node_data.items():
            if key not in excluded_params:
                n8n_node["parameters"][key] = value
        
        # Handle credentials (API keys) - add to parameters instead of credentials section
        if "apiKey" in node_data and node_data["apiKey"]:
            # Reference credentials instead of storing directly
            credential_name = f"{node_name} API Key"
            credential_id = f"tenant-{tenant_id}-credential" if tenant_id else "credential-id"
            
            # Determine credential type based on node type
            if node["type"] == "chatmodel":
                provider = node_data.get("modelProvider", "openai")
                if provider == "openai":
                    credential_type = "openAiApi"
                elif provider == "anthropic":
                    credential_type = "anthropicApi"
                elif provider == "google":
                    credential_type = "googlePalmApi"
                else:
                    credential_type = "apiKey"
            else:
                credential_type = "apiKey"
            
            # Add credential info to parameters instead of credentials section
            n8n_node["parameters"][credential_type] = {
                "id": credential_id,
                "name": credential_name
            }
        
        n8n_nodes.append(n8n_node)
    
    # Transform edges to connections
    connections = {}
    for edge in edges:
        source_id = edge["source"]
        target_id = edge["target"]
        
        source_node = node_lookup.get(source_id)
        target_node = node_lookup.get(target_id)
        
        if not source_node or not target_node:
            continue
        
        # Get node names
        source_data = source_node.get("data", {})
        source_name = (
            source_data.get("agentName") or
            source_data.get("modelName") or
            source_data.get("callAgentName") or
            source_data.get("toolName") or
            source_id
        )
        
        target_data = target_node.get("data", {})
        target_name = (
            target_data.get("agentName") or
            target_data.get("modelName") or
            target_data.get("callAgentName") or
            target_data.get("toolName") or
            target_id
        )
        
        # Map sourceHandle to output channel
        source_handle = edge.get("sourceHandle", "main")
        
        # Map React Flow handles to n8n output channels
        output_channel_map = {
            "chat-model-out": "ai_languageModel",
            "memory-out": "ai_memory",
            "database-out": "ai_memory",
            "tool-out": "ai_tool",
            "agent-out": "main",
            "agent-in": "main",  # For incoming connections
        }
        
        # Default to "main" if not mapped
        output_channel = output_channel_map.get(source_handle.replace("-out", "").replace("-in", ""), "main")
        
        # Initialize connection structure
        if source_name not in connections:
            connections[source_name] = {}
        if output_channel not in connections[source_name]:
            connections[source_name][output_channel] = [[]]
        
        # Add connection
        connections[source_name][output_channel][0].append({
            "node": target_name,
            "type": output_channel,
            "index": 0
        })
    
    # Build n8n format JSON
    n8n_json = {
        "flowId": flow_id,
        "flowName": flow_name,
        "tenantId": tenant_id or "",
        "userId": user_id or "",
        "nodes": n8n_nodes,
        "connections": connections,
        "settings": {
            "saveExecutionProgress": True,
            "saveManualExecutions": True,
            "saveDataErrorExecution": "all",
            "saveDataSuccessExecution": "all",
            "executionTimeout": 3600,
            "timezone": "UTC",
            "executionOrder": "v1"
        },
        "active": True,
        "meta": {
            "templateCredsSetupCompleted": False
        }
    }
    
    return n8n_json


def transform_n8n_to_react_flow(
    n8n_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Transform n8n-like format back to React Flow format.
    
    Args:
        n8n_json: n8n-like format JSON
    
    Returns:
        React Flow format JSON
    """
    nodes = n8n_json.get("nodes", [])
    connections = n8n_json.get("connections", {})
    
    # Build name to ID lookup
    name_to_id = {node["name"]: node["id"] for node in nodes}
    
    # Transform nodes
    react_nodes = []
    for node in nodes:
        # Remove prefix if present (for backward compatibility), otherwise use as-is
        node_type = node["type"]
        if node_type.startswith("adk-nodes-base."):
            node_type = node_type.replace("adk-nodes-base.", "")
        
        react_node = {
            "id": node["id"],
            "type": node_type,
            "position": {
                "x": node["position"][0],
                "y": node["position"][1]
            },
            "data": {},
            "width": 200,
            "height": 120,
            "selected": False,
            "dragging": False,
        }
        
        # Copy parameters back to data (including credential info if present)
        for key, value in node.get("parameters", {}).items():
            react_node["data"][key] = value
        
        # Handle credentials from old format (if present) - convert to parameters in data
        if "credentials" in node:
            for cred_type, cred_info in node["credentials"].items():
                react_node["data"][cred_type] = cred_info
        
        # Add isDisabled if present
        if node.get("disabled"):
            react_node["data"]["isDisabled"] = True
        
        # Restore positionAbsolute (same as position for now)
        react_node["positionAbsolute"] = react_node["position"].copy()
        
        react_nodes.append(react_node)
    
    # Transform connections to edges
    edges = []
    edge_counter = 0
    
    for source_name, outputs in connections.items():
        source_id = name_to_id.get(source_name)
        if not source_id:
            continue
        
        for output_channel, branches in outputs.items():
            if not branches or not branches[0]:
                continue
            
            # Map n8n channels to React Flow handles
            handle_map = {
                "ai_languageModel": "chat-model-out",
                "ai_memory": "memory-out",
                "ai_tool": "tool-out",
                "main": "agent-out",
            }
            
            source_handle = handle_map.get(output_channel, "main")
            
            for branch in branches:
                for connection in branch:
                    target_name = connection["node"]
                    target_id = name_to_id.get(target_name)
                    
                    if not target_id:
                        continue
                    
                    # Determine target handle based on output channel
                    if output_channel == "ai_languageModel":
                        target_handle = "chat-model-in"
                    elif output_channel == "ai_memory":
                        target_handle = "memory-in"
                    elif output_channel == "ai_tool":
                        target_handle = "tool-in"
                    else:
                        target_handle = "agent-in"
                    
                    edges.append({
                        "id": f"e{source_id}-{target_id}-{edge_counter}",
                        "source": source_id,
                        "target": target_id,
                        "sourceHandle": source_handle,
                        "targetHandle": target_handle,
                        "animated": True,
                        "style": {
                            "stroke": "#000",
                            "strokeWidth": 2,
                            "strokeDasharray": "5 5"
                        },
                        "markerEnd": {
                            "type": "arrowclosed",
                            "color": "#000",
                            "width": 20,
                            "height": 20
                        }
                    })
                    edge_counter += 1
    
    return {
        "nodes": react_nodes,
        "edges": edges
    }

