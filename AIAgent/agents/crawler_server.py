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
Web Crawler MCP Agent
Fetches web pages and returns raw HTML content.
"""

import asyncio
import json
from typing import Any, Dict
import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Initialize MCP Server
app = Server("crawler-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for web crawling."""
    return [
        Tool(
            name="fetch_url",
            description="Fetches the raw HTML content from a given URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch",
                    },
                },
                "required": ["url"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute the requested tool."""
    if name != "fetch_url":
        raise ValueError(f"Unknown tool: {name}")

    url = arguments.get("url")
    if not url:
        raise ValueError("URL is required")

    try:
        # Fetch the URL with a timeout
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        result = {
            "url": url,
            "status_code": response.status_code,
            "html": response.text,
        }
        
        return [TextContent(type="text", text=json.dumps(result))]
    
    except requests.RequestException as e:
        error_result = {
            "url": url,
            "error": str(e),
            "html": "",
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

