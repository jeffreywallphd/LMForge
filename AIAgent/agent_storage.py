#!/usr/bin/env python3
"""
Agent Storage System
Handles saving, loading, and managing custom user-created agents
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class AgentStorage:
    """Manages storage and retrieval of custom agents."""
    
    def __init__(self, storage_dir: str = "custom_agents"):
        """Initialize agent storage."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.agents_file = self.storage_dir / "agents.json"
        self.agents_code_dir = self.storage_dir / "code"
        self.agents_code_dir.mkdir(exist_ok=True)
        
        # Initialize agents file if it doesn't exist
        if not self.agents_file.exists():
            self._save_agents({})
    
    def _load_agents(self) -> Dict[str, Any]:
        """Load all agents from storage."""
        try:
            with open(self.agents_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_agents(self, agents: Dict[str, Any]):
        """Save all agents to storage."""
        with open(self.agents_file, 'w', encoding='utf-8') as f:
            json.dump(agents, f, indent=2, ensure_ascii=False, default=str)
    
    def create_agent(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent."""
        agents = self._load_agents()
        
        # Generate ID if not provided
        agent_id = agent_data.get('id') or f"agent_{len(agents) + 1}_{int(datetime.now().timestamp())}"
        
        # Create agent object
        agent = {
            "id": agent_id,
            "name": agent_data.get("name", "Untitled Agent"),
            "description": agent_data.get("description", ""),
            "tools": agent_data.get("tools", []),
            "code": agent_data.get("code", ""),
            "config": agent_data.get("config", {}),
            "avatar": agent_data.get("avatar", "🤖"),
            "tags": agent_data.get("tags", []),
            "isPublic": agent_data.get("isPublic", False),
            "createdAt": datetime.now().isoformat(),
            "updatedAt": datetime.now().isoformat(),
            "createdBy": agent_data.get("createdBy", "user"),
            "status": "active"
        }
        
        # Save agent
        agents[agent_id] = agent
        
        # Generate and save agent code
        agent_code = self._generate_agent_code(agent)
        code_file = self.agents_code_dir / f"{agent_id}.py"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(agent_code)
        
        self._save_agents(agents)
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent by ID."""
        agents = self._load_agents()
        return agents.get(agent_id)
    
    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get all agents."""
        agents = self._load_agents()
        return list(agents.values())
    
    def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing agent."""
        agents = self._load_agents()
        if agent_id not in agents:
            return None
        
        agent = agents[agent_id]
        agent.update(updates)
        agent["updatedAt"] = datetime.now().isoformat()
        
        # Regenerate code if tools or config changed
        if "tools" in updates or "config" in updates:
            agent_code = self._generate_agent_code(agent)
            code_file = self.agents_code_dir / f"{agent_id}.py"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(agent_code)
        
        self._save_agents(agents)
        return agent
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        agents = self._load_agents()
        if agent_id not in agents:
            return False
        
        del agents[agent_id]
        
        # Delete code file
        code_file = self.agents_code_dir / f"{agent_id}.py"
        if code_file.exists():
            code_file.unlink()
        
        self._save_agents(agents)
        return True
    
    def _generate_agent_code(self, agent: Dict[str, Any]) -> str:
        """Generate Python code for an MCP agent."""
        agent_id = agent["id"]
        agent_name = agent["name"].lower().replace(" ", "_")
        tools = agent.get("tools", [])
        
        # Generate tool definitions
        tool_definitions = []
        tool_handlers = []
        
        for tool in tools:
            tool_name = tool.get("name", "").lower().replace(" ", "_")
            tool_desc = tool.get("description", "")
            tool_type = tool.get("type", "function")
            tool_config = tool.get("config", {})
            
            # Generate input schema
            input_schema = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            # Add parameters from config
            if "parameters" in tool_config:
                for param in tool_config["parameters"]:
                    param_name = param.get("name", "")
                    param_type = param.get("type", "string")
                    param_desc = param.get("description", "")
                    param_required = param.get("required", False)
                    
                    input_schema["properties"][param_name] = {
                        "type": param_type,
                        "description": param_desc
                    }
                    
                    if param_required:
                        input_schema["required"].append(param_name)
            
            # Tool definition
            tool_def = f"""        Tool(
            name="{tool_name}",
            description="{tool_desc}",
            inputSchema={json.dumps(input_schema, indent=12)},
        )"""
            tool_definitions.append(tool_def)
            
            # Tool handler
            handler_code = self._generate_tool_handler(tool, tool_config)
            tool_handlers.append(handler_code)
        
        # Generate complete agent code
        agent_code = f'''#!/usr/bin/env python3
"""
Custom MCP Agent: {agent["name"]}
{agent.get("description", "")}
Generated by LMForge Agent Platform
"""

import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Initialize MCP Server
app = Server("{agent_id}")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for this agent."""
    return [
{chr(10).join(tool_definitions)}
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute the requested tool."""
    {self._generate_tool_router(tools)}
    
    raise ValueError(f"Unknown tool: {{name}}")

{chr(10).join(tool_handlers)}

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
'''
        return agent_code
    
    def _generate_tool_handler(self, tool: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate handler code for a tool."""
        tool_name = tool.get("name", "").lower().replace(" ", "_")
        tool_type = tool.get("type", "function")
        code_template = config.get("code", "")
        
        if code_template:
            # Use custom code template
            return f'''
async def handle_{tool_name}(arguments: dict) -> list[TextContent]:
    """Handle {tool.get("name", "")} tool."""
    try:
{self._indent_code(code_template, 8)}
        result = {{"success": True, "result": result}}
        return [TextContent(type="text", text=json.dumps(result))]
    except Exception as e:
        error_result = {{"success": False, "error": str(e)}}
        return [TextContent(type="text", text=json.dumps(error_result))]
'''
        else:
            # Generate default handler based on type
            if tool_type == "http_request":
                return self._generate_http_handler(tool_name)
            elif tool_type == "data_transform":
                return self._generate_transform_handler(tool_name)
            else:
                return self._generate_default_handler(tool_name)
    
    def _generate_tool_router(self, tools: List[Dict[str, Any]]) -> str:
        """Generate tool routing logic."""
        if not tools:
            return 'raise ValueError("No tools defined")'
        
        router_lines = []
        for tool in tools:
            tool_name = tool.get("name", "").lower().replace(" ", "_")
            router_lines.append(f'    if name == "{tool_name}":')
            router_lines.append(f'        return await handle_{tool_name}(arguments)')
        
        return '\n'.join(router_lines)
    
    def _generate_http_handler(self, tool_name: str) -> str:
        """Generate HTTP request handler."""
        return f'''
async def handle_{tool_name}(arguments: dict) -> list[TextContent]:
    """Handle {tool_name} HTTP request."""
    import requests
    try:
        url = arguments.get("url", "")
        method = arguments.get("method", "GET")
        headers = arguments.get("headers", {{}})
        data = arguments.get("data", None)
        
        response = requests.request(method, url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        
        result = {{"success": True, "status_code": response.status_code, "data": response.text}}
        return [TextContent(type="text", text=json.dumps(result))]
    except Exception as e:
        error_result = {{"success": False, "error": str(e)}}
        return [TextContent(type="text", text=json.dumps(error_result))]
'''
    
    def _generate_transform_handler(self, tool_name: str) -> str:
        """Generate data transformation handler."""
        return f'''
async def handle_{tool_name}(arguments: dict) -> list[TextContent]:
    """Handle {tool_name} data transformation."""
    try:
        data = arguments.get("data", "")
        # Add your transformation logic here
        result = {{"success": True, "transformed": data}}
        return [TextContent(type="text", text=json.dumps(result))]
    except Exception as e:
        error_result = {{"success": False, "error": str(e)}}
        return [TextContent(type="text", text=json.dumps(error_result))]
'''
    
    def _generate_default_handler(self, tool_name: str) -> str:
        """Generate default handler."""
        return f'''
async def handle_{tool_name}(arguments: dict) -> list[TextContent]:
    """Handle {tool_name} tool."""
    try:
        # Add your custom logic here
        result = {{"success": True, "message": "Tool executed successfully"}}
        return [TextContent(type="text", text=json.dumps(result))]
    except Exception as e:
        error_result = {{"success": False, "error": str(e)}}
        return [TextContent(type="text", text=json.dumps(error_result))]
'''
    
    def _indent_code(self, code: str, indent: int) -> str:
        """Indent code block."""
        lines = code.split('\n')
        return '\n'.join(' ' * indent + line for line in lines)
    
    def get_agent_code_path(self, agent_id: str) -> Path:
        """Get the file path for an agent's code."""
        return self.agents_code_dir / f"{agent_id}.py"


# Global instance
agent_storage = AgentStorage()

