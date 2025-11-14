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
FastAPI Application
Provides REST API endpoints for the LMForge workflow.
"""

import os
import sys
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import orchestrator
from main import get_orchestrator

# Import agent storage
try:
    from agent_storage import agent_storage
except ImportError:
    # Fallback if agent_storage not available
    agent_storage = None
    print("⚠️ Warning: agent_storage not available. Agent creation will be disabled.")

# Load environment variables
load_dotenv()


# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    print("🚀 Starting LMForge MCP-Use Backend...")
    try:
        orchestrator = await get_orchestrator()
        print("✓ All agents initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize agents: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    yield
    
    # Shutdown
    print("🛑 Shutting down LMForge MCP-Use Backend...")
    try:
        orchestrator = await get_orchestrator()
        await orchestrator.close()
        print("✓ All agents closed successfully")
    except Exception as e:
        print(f"✗ Failed to close agents: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="LMForge MCP-Use Backend",
    description="Open-source backend framework using MCP-Use and FastAPI",
    version="1.0.0",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class RunFlowRequest(BaseModel):
    """Request model for the run-flow endpoint."""
    url: str = Field(..., description="The URL to process")


class RunFlowResponse(BaseModel):
    """Response model for the run-flow endpoint."""
    url: str
    raw_html: str
    parsed_text: str
    cleaned_text: str
    success: bool = True


class GenerateQARequest(BaseModel):
    """Request model for the generate-qa endpoint."""
    text: str = Field(..., description="The text to generate Q&A from")


class QAPair(BaseModel):
    """A single question/answer pair."""
    question: str
    answer: str


class GenerateQAResponse(BaseModel):
    """Response model for the generate-qa endpoint."""
    qa_pairs: list[QAPair]
    success: bool = True


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LMForge MCP-Use Backend",
        "version": "1.0.0",
        "endpoints": {
            "run_flow": "/run-flow",
            "generate_qa": "/generate-qa",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "LMForge MCP-Use Backend"
    }


@app.post("/run-flow", response_model=RunFlowResponse)
async def run_flow(request: RunFlowRequest) -> Dict[str, Any]:
    """
    Execute the complete workflow: crawl → parse → clean.
    
    This endpoint:
    1. Crawls the provided URL
    2. Parses the HTML to extract text
    3. Cleans and normalizes the text
    
    Args:
        request: RunFlowRequest containing the URL
        
    Returns:
        RunFlowResponse with results from each stage
        
    Raises:
        HTTPException: If the workflow fails
    """
    try:
        orchestrator = await get_orchestrator()
        result = await orchestrator.run_full_flow(request.url)
        
        if "error" in result:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": result["error"],
                    "stage": result.get("stage", "unknown")
                }
            )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/generate-qa", response_model=GenerateQAResponse)
async def generate_qa(request: GenerateQARequest) -> Dict[str, Any]:
    """
    Generate 3 concise question/answer pairs from provided text.
    
    This endpoint uses an AI language model to:
    1. Analyze the provided text
    2. Generate relevant questions
    3. Provide concise answers
    
    Args:
        request: GenerateQARequest containing the text
        
    Returns:
        GenerateQAResponse with the Q&A pairs
        
    Raises:
        HTTPException: If Q&A generation fails
    """
    try:
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=400,
                detail="OPENAI_API_KEY environment variable is not set"
            )
        
        orchestrator = await get_orchestrator()
        result = await orchestrator.generate_qa(request.text)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": result.get("error", "Q&A generation failed"),
                    "qa_pairs": result.get("qa_pairs", [])
                }
            )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# Agent Management Endpoints
class CreateAgentRequest(BaseModel):
    """Request model for creating an agent."""
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    tools: list = Field(default=[], description="List of tools for the agent")
    code: str = Field(default="", description="Custom code for the agent")
    config: dict = Field(default={}, description="Agent configuration")
    avatar: str = Field(default="🤖", description="Agent avatar emoji")
    tags: list[str] = Field(default=[], description="Agent tags")
    isPublic: bool = Field(default=False, description="Whether agent is public")
    createdBy: str = Field(default="user", description="Creator identifier")


class UpdateAgentRequest(BaseModel):
    """Request model for updating an agent."""
    name: Optional[str] = None
    description: Optional[str] = None
    tools: Optional[list] = None
    code: Optional[str] = None
    config: Optional[dict] = None
    avatar: Optional[str] = None
    tags: Optional[list[str]] = None
    isPublic: Optional[bool] = None


@app.post("/api/agents", status_code=201)
async def create_agent(request: CreateAgentRequest) -> Dict[str, Any]:
    """
    Create a new custom agent.
    
    Args:
        request: CreateAgentRequest with agent details
        
    Returns:
        Created agent object
    """
    if not agent_storage:
        raise HTTPException(status_code=503, detail="Agent storage not available")
    try:
        agent_data = request.dict()
        agent = agent_storage.create_agent(agent_data)
        return agent
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create agent: {str(e)}"
        )


@app.get("/api/agents")
async def list_agents() -> Dict[str, Any]:
    """
    List all custom agents.
    
    Returns:
        List of all agents
    """
    if not agent_storage:
        raise HTTPException(status_code=503, detail="Agent storage not available")
    try:
        agents = agent_storage.get_all_agents()
        return {
            "agents": agents,
            "count": len(agents)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list agents: {str(e)}"
        )


@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str) -> Dict[str, Any]:
    """
    Get a specific agent by ID.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Agent object
    """
    if not agent_storage:
        raise HTTPException(status_code=503, detail="Agent storage not available")
    try:
        agent = agent_storage.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent: {str(e)}"
        )


@app.put("/api/agents/{agent_id}")
async def update_agent(agent_id: str, request: UpdateAgentRequest) -> Dict[str, Any]:
    """
    Update an existing agent.
    
    Args:
        agent_id: Agent identifier
        request: UpdateAgentRequest with updates
        
    Returns:
        Updated agent object
    """
    if not agent_storage:
        raise HTTPException(status_code=503, detail="Agent storage not available")
    try:
        updates = {k: v for k, v in request.dict().items() if v is not None}
        agent = agent_storage.update_agent(agent_id, updates)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update agent: {str(e)}"
        )


@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str) -> Dict[str, Any]:
    """
    Delete an agent.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Success message
    """
    if not agent_storage:
        raise HTTPException(status_code=503, detail="Agent storage not available")
    try:
        success = agent_storage.delete_agent(agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"success": True, "message": "Agent deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete agent: {str(e)}"
        )


@app.get("/api/agents/{agent_id}/code")
async def get_agent_code(agent_id: str):
    """
    Get the generated code for an agent.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Agent code as text
    """
    if not agent_storage:
        raise HTTPException(status_code=503, detail="Agent storage not available")
    try:
        agent = agent_storage.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        code_path = agent_storage.get_agent_code_path(agent_id)
        if not code_path.exists():
            raise HTTPException(status_code=404, detail="Agent code not found")
        
        with open(code_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(code, media_type="text/plain")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent code: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

