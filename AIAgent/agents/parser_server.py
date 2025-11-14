#!/usr/bin/env python3
# Copyright 2025 LMForge
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
HTML Parser MCP Agent
Extracts text content from HTML using BeautifulSoup4.
"""

import asyncio
import json
from typing import Any, Dict
from bs4 import BeautifulSoup
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Initialize MCP Server
app = Server("parser-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for HTML parsing."""
    return [
        Tool(
            name="parse_html",
            description="Extracts text content from HTML, removing scripts and styles",
            inputSchema={
                "type": "object",
                "properties": {
                    "html": {
                        "type": "string",
                        "description": "The HTML content to parse",
                    },
                },
                "required": ["html"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute the requested tool."""
    if name != "parse_html":
        raise ValueError(f"Unknown tool: {name}")

    html = arguments.get("html")
    if html is None:
        raise ValueError("HTML content is required")

    try:
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'meta', 'noscript']):
            script_or_style.decompose()
        
        # Extract text
        text = soup.get_text()
        
        # Basic cleanup
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        result = {
            "text": text,
            "success": True,
        }
        
        return [TextContent(type="text", text=json.dumps(result))]
    
    except Exception as e:
        error_result = {
            "text": "",
            "error": str(e),
            "success": False,
        }
        return [TextContent(type="text", text=json.dumps(error_result))]


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

