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
MCP Client Registration and Orchestration
Manages MCP agents and provides high-level workflow functions.
"""

import os
import json
import asyncio
from typing import Dict, Any
from mcp_use import MCPClient

# Remove MCPAgent type hint since we're not using it


class LMForgeOrchestrator:
    """Orchestrates MCP agents for the LMForge workflow."""
    
    def __init__(self):
        """Initialize the orchestrator with MCP clients."""
        self.clients: Dict[str, MCPClient] = {}
        self.agents: Dict[str, Any] = {}  # Sessions, not MCPAgent
        
    async def initialize(self):
        """Initialize all MCP clients and agents."""
        # MCP server configurations - 2 agents: Crawler, Parser
        import sys
        import subprocess
        
        mcp_servers = {
            "crawler": {
                "command": sys.executable,  # Use current Python interpreter
                "args": ["agents/crawler_server.py"],
                "env": os.environ.copy()  # Pass environment variables
            },
            "parser": {
                "command": sys.executable,
                "args": ["agents/parser_server.py"],
                "env": os.environ.copy()
            }
        }
        
        # Initialize each MCP client
        for name, config in mcp_servers.items():
            try:
                # Create MCP client from config
                # Add the server to the client
                client = MCPClient()
                client.add_server(name, config)
                
                # Create all sessions
                await client.create_all_sessions()
                
                # Get the session
                session = client.get_session(name)
                if not session:
                    raise RuntimeError(f"Failed to create session for {name}")
                
                # Store client and session
                self.clients[name] = client
                self.agents[name] = session
                
                print(f"✓ Initialized {name} agent")
            except Exception as e:
                error_msg = str(e)
                # More helpful error messages
                if "fileno" in error_msg.lower():
                    print(f"✗ Failed to initialize {name} agent: Windows stdio issue")
                    print(f"  Using fallback implementation (direct function calls)")
                    print(f"  MCP agents will work better on Linux/Mac or in Colab")
                elif "langchain" in error_msg.lower() or "ModuleNotFoundError" in error_msg:
                    print(f"✗ Failed to initialize {name} agent: Missing dependencies")
                    print(f"  Install: pip install langchain-openai")
                elif "OPENAI_API_KEY" in error_msg:
                    print(f"✗ Failed to initialize {name} agent: OPENAI_API_KEY not set")
                else:
                    print(f"✗ Failed to initialize {name} agent: {error_msg}")
                # Continue with other agents even if one fails
    
    async def crawl_url(self, url: str) -> Dict[str, Any]:
        """
        Crawl a URL and return the raw HTML.
        
        Args:
            url: The URL to crawl
            
        Returns:
            Dict containing the HTML content and metadata
        """
        # Try MCP agent first
        if "crawler" in self.agents:
            try:
                session = self.agents["crawler"]
                result = await session.call_tool("fetch_url", {"url": url})
                
                # Parse the result - mcp-use 1.4.0 returns CallToolResult object
                if result and hasattr(result, 'content'):
                    # Get the first content item
                    if result.content and len(result.content) > 0:
                        content_item = result.content[0]
                        result_text = content_item.text if hasattr(content_item, 'text') else str(content_item)
                        return json.loads(result_text)
                raise RuntimeError("No result from crawler")
            except Exception as e:
                print(f"⚠ MCP crawler failed, using fallback: {e}")
        
        # Fallback: Direct implementation
        import requests
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            return {
                "url": url,
                "status_code": response.status_code,
                "html": response.text,
            }
        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "html": "",
            }
    
    async def parse_html(self, html: str) -> Dict[str, Any]:
        """
        Parse HTML and extract text content.
        
        Args:
            html: The HTML content to parse
            
        Returns:
            Dict containing the parsed text
        """
        # Try MCP agent first
        if "parser" in self.agents:
            try:
                session = self.agents["parser"]
                result = await session.call_tool("parse_html", {"html": html})
                
                # Parse the result
                if result and hasattr(result, 'content') and result.content:
                    content_item = result.content[0]
                    result_text = content_item.text if hasattr(content_item, 'text') else str(content_item)
                    return json.loads(result_text)
                raise RuntimeError("No result from parser")
            except Exception as e:
                print(f"⚠ MCP parser failed, using fallback: {e}")
        
        # Fallback: Direct implementation
        from bs4 import BeautifulSoup
        try:
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
            
            return {
                "text": text,
                "success": True,
            }
        except Exception as e:
            return {
                "text": "",
                "error": str(e),
                "success": False,
            }
    
    def clean_text_simple(self, text: str) -> str:
        """
        Simple text cleaning (no agent needed for demo).
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        import re
        # Basic cleanup
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    async def generate_qa(self, text: str) -> Dict[str, Any]:
        """
        Generate Q&A pairs from text using QA Generator agent.
        
        Args:
            text: The text to generate Q&A from
            
        Returns:
            Dict containing the Q&A pairs
        """
        if "qa_generator" not in self.agents:
            raise RuntimeError("Q&A generator agent not initialized")
        
        session = self.agents["qa_generator"]
        result = await session.call_tool("generate_qa", {"text": text})
        
        # Parse the result
        if result and hasattr(result, 'content') and result.content:
            content_item = result.content[0]
            result_text = content_item.text if hasattr(content_item, 'text') else str(content_item)
            return json.loads(result_text)
        raise RuntimeError("No result from Q&A generator")
    
    async def run_full_flow(self, url: str) -> Dict[str, Any]:
        """
        Run the complete workflow: crawl → parse → clean.
        
        Args:
            url: The URL to process
            
        Returns:
            Dict containing results from each stage
        """
        # Step 1: Crawl
        crawl_result = await self.crawl_url(url)
        if "error" in crawl_result:
            return {
                "url": url,
                "error": crawl_result["error"],
                "stage": "crawl"
            }
        
        html_content = crawl_result.get("html", "")
        if not html_content:
            return {
                "url": url,
                "error": "No HTML content received",
                "stage": "crawl"
            }
        
        # Step 2: Parse
        parse_result = await self.parse_html(html_content)
        if not parse_result.get("success", False):
            return {
                "url": url,
                "error": parse_result.get("error", "Parse failed"),
                "stage": "parse"
            }
        
        # Step 3: Simple clean (no agent needed)
        parsed_text = parse_result.get("text", "")
        cleaned_text = self.clean_text_simple(parsed_text)
        
        return {
            "url": url,
            "raw_html": html_content[:500] + ("..." if len(html_content) > 500 else ""),
            "parsed_text": parsed_text[:500] + ("..." if len(parsed_text) > 500 else ""),
            "cleaned_text": cleaned_text,
            "success": True
        }
    
    async def close(self):
        """Close all MCP clients."""
        for name, client in self.clients.items():
            try:
                # Close all sessions
                await client.close_all_sessions()
                print(f"✓ Closed {name} client")
            except Exception as e:
                print(f"✗ Failed to close {name} client: {str(e)}")


# Global orchestrator instance
orchestrator = LMForgeOrchestrator()


async def get_orchestrator() -> LMForgeOrchestrator:
    """
    Get or create the global orchestrator instance.
    
    Returns:
        The initialized orchestrator
    """
    if not orchestrator.clients:
        await orchestrator.initialize()
    return orchestrator


if __name__ == "__main__":
    # Test the orchestrator
    async def test():
        orch = await get_orchestrator()
        result = await orch.run_full_flow("https://example.com")
        print(json.dumps(result, indent=2))
        await orch.close()
    
    asyncio.run(test())

