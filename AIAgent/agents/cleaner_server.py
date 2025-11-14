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
Text Cleaner MCP Agent
Normalizes and cleans text content.
"""

import asyncio
import json
import re
from typing import Any, Dict
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Initialize MCP Server
app = Server("cleaner-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for text cleaning."""
    return [
        Tool(
            name="clean_text",
            description="Normalizes whitespace and removes unwanted characters from text",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content to clean",
                    },
                },
                "required": ["text"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute the requested tool."""
    if name != "clean_text":
        raise ValueError(f"Unknown tool: {name}")

    text = arguments.get("text")
    if text is None:
        raise ValueError("Text content is required")

    try:
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple consecutive newlines
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Remove special control characters but keep basic punctuation
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Trim leading/trailing whitespace
        text = text.strip()
        
        # Remove URLs (optional - comment out if you want to keep URLs)
        # text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses (optional)
        # text = re.sub(r'\S+@\S+', '', text)
        
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

