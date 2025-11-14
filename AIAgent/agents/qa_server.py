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
Q&A Generator MCP Agent
Generates question/answer pairs from text using OpenAI LLM.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Initialize MCP Server
app = Server("qa-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for Q&A generation."""
    return [
        Tool(
            name="generate_qa",
            description="Generates 3 concise question/answer pairs from the provided text using AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content to generate Q&A from",
                    },
                },
                "required": ["text"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute the requested tool."""
    if name != "generate_qa":
        raise ValueError(f"Unknown tool: {name}")

    text = arguments.get("text")
    if not text:
        raise ValueError("Text content is required")

    try:
        # Initialize OpenAI LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=api_key
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at creating concise, relevant questions and answers from text content."),
            ("user", """Based on the following text, generate exactly 3 question-answer pairs.
Each question should be clear and specific, and each answer should be concise (1-2 sentences).

Text: {text}

Format your response as a JSON array with this structure:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]

Only return the JSON array, nothing else.""")
        ])
        
        # Generate Q&A pairs
        chain = prompt | llm
        response = await asyncio.to_thread(
            chain.invoke,
            {"text": text[:3000]}  # Limit text length for API
        )
        
        # Parse the response
        response_text = response.content.strip()
        
        # Try to extract JSON from the response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        qa_pairs = json.loads(response_text)
        
        result = {
            "qa_pairs": qa_pairs,
            "success": True,
        }
        
        return [TextContent(type="text", text=json.dumps(result))]
    
    except json.JSONDecodeError as e:
        # If JSON parsing fails, create a fallback response
        error_result = {
            "qa_pairs": [
                {
                    "question": "What is the main topic of this text?",
                    "answer": "The text discusses various topics that require proper parsing."
                },
                {
                    "question": "How is this content structured?",
                    "answer": "The content follows a standard format."
                },
                {
                    "question": "What can be learned from this text?",
                    "answer": "This text provides information on the subject matter."
                }
            ],
            "error": f"JSON parsing error: {str(e)}",
            "success": False,
        }
        return [TextContent(type="text", text=json.dumps(error_result))]
    
    except Exception as e:
        error_result = {
            "qa_pairs": [],
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

