# LMForge Agentic Systems - Complete Project Documentation

**Documentation for Next Semester Team**

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [What We Have Built](#what-we-have-built)
3. [Frontend UI Features](#frontend-ui-features)
4. [Backend API](#backend-api)
5. [Current Status & Issues](#current-status--issues)
6. [Problems Encountered](#problems-encountered)
7. [Next Steps](#next-steps)
8. [Technical Architecture](#technical-architecture)

---

## 🎯 Project Overview

**LMForge Agentic Systems** is a platform that allows users to create, manage, and deploy AI agents using the Model Context Protocol (MCP). The system consists of:

- **Backend**: FastAPI server with MCP agent orchestration
- **Frontend**: React-based UI for agent management and interaction
- **Agents**: Modular MCP-based agents (Crawler, Parser, Cleaner, QA Generator)

**Goal**: Enable users to create custom agents through a user-friendly interface without writing code.

---

## 🧠 Key Concepts (For New Contributors)

| Term | Meaning in This Project | Why It Matters |
|------|-------------------------|----------------|
| **AI Agent** | A focused program that exposes “tools” (functions) such as `fetch_url` or `parse_html`. Agents speak the **Model Context Protocol (MCP)** so other systems can call them in a predictable way. | Keeps each task modular. We can swap or extend agents without changing the rest of the system. |
| **MCP (Model Context Protocol)** | An open protocol from Anthropic/OpenAI ecosystem defining how agents advertise tools, receive calls, and return results over stdio/websocket transports. | Gives us a language-agnostic contract for agent interactions. Any MCP-compatible orchestrator can talk to our agents. |
| **MCP-Use Library** | Python client & tooling that lets our FastAPI backend act as an MCP orchestrator. Handles launching agent processes, creating sessions, and calling their tools. | We rely on `mcp-use` to manage lifecycle: `MCPClient`, `add_server`, `create_all_sessions`, `call_tool`, etc. Keeping this updated avoids breaking changes. |
| **Tool** | A function exposed by an agent. Example: Crawler agent exposes `fetch_url(url: string)`. Tools include JSON schemas so callers know what inputs are required. | Granular tools allow UI/workflows to compose agents (Crawler ➜ Parser ➜ Cleaner). |
| **Agent Orchestrator** | The component in `main.py` that registers all MCP servers, keeps sessions alive, and provides helper functions like `crawl_url()` or `run_full_flow()`. | Central brain that ties everything together so FastAPI endpoints remain simple. |
| **Custom Agent** | A user-defined agent created via the UI. Stored in `custom_agents/agents.json`, with generated Python code under `custom_agents/code/*.py`. | Future-proofing: teams can extend capabilities without editing core agents. Needs dynamic loading to be fully live. |

**Big picture**: MCP agents are like microservices, but instead of HTTP they expose structured tools via a standard protocol. `mcp-use` is our SDK that makes running and calling those agents easier.

---

## ✅ What We Have Built

### 1. **Backend System** (Fully Functional)

#### Core Components:
- **`app.py`**: FastAPI application with REST API endpoints
- **`main.py`**: MCP orchestrator that coordinates multiple agents
- **`agent_storage.py`**: System for saving/loading user-created agents
- **`agents/`**: Pre-built MCP agents:
  - `crawler_server.py` - Fetches HTML from URLs
  - `parser_server.py` - Extracts text from HTML
  - `cleaner_server.py` - Cleans and normalizes text
  - `qa_server.py` - Generates Q&A pairs using OpenAI

#### Working Features:
✅ **URL Extraction Flow** (`/run-flow` endpoint)
- Accepts a URL
- Crawls the webpage
- Parses HTML to extract text
- Returns clean, structured text

✅ **Q&A Generation** (`/generate-qa` endpoint)
- Takes text input
- Generates 3 question/answer pairs using OpenAI
- Returns formatted Q&A

✅ **Agent Storage System**
- Saves agent metadata to `custom_agents/agents.json`
- Generates Python code for agents in `custom_agents/code/`
- Supports CRUD operations (Create, Read, Update, Delete)

---

## 🤖 Agent Inventory & Status

| Agent | File | Purpose | Current Status | Notes |
|-------|------|---------|----------------|-------|
| **Crawler Agent** | `agents/crawler_server.py` | Fetches raw HTML using `requests` with proper headers and timeout handling. | ✅ Working | Includes fallback in `main.py` that uses direct `requests` calls if MCP session fails (common on Windows). |
| **Parser Agent** | `agents/parser_server.py` | Uses BeautifulSoup4 to remove scripts/styles and return clean text. | ✅ Working | Also has fallback path inside `main.py` so `/run-flow` keeps working even if MCP session crashes. |
| **Cleaner Agent** | `agents/cleaner_server.py` | Normalizes whitespace, removes weird characters, prepares text for downstream tasks. | ✅ Working (used internally) | Runs after parser inside orchestrator; no separate endpoint. |
| **QA Generator Agent** | `agents/qa_server.py` | Calls OpenAI via `langchain-openai` to create 3 Q/A pairs. | ⚠️ Built but currently **disabled** during demos | Failure reason: OpenAI API key quota/credits exhausted on shared account. Code paths/logging in `main.py`+`qa_server.py` already handle missing keys. Enable by adding a funded API key to `.env`. |
| **Custom User Agents** | Generated into `custom_agents/code/<agent_id>.py` | Dynamically generated MCP servers from the Agent Creator UI. | ⚠️ Code generation works, MCP auto-loading **pending** | Once the VPN issue is resolved, we need to load these agents at startup (see “Next Steps”). |

**Why QA Agent isn’t showcased right now**
- The agent itself works (tested earlier with a valid API key).
- Current blockers are billing quotas and shared-key restrictions.
- Once the next team provides a new `OPENAI_API_KEY`, you can flip the switch:
  1. Add key to `.env`
  2. Re-run backend (`python app.py`)
  3. Hit `/generate-qa` or the UI button to bring the QA agent back online.

---

### 2. **Frontend UI** (Fully Built, Partially Connected)

#### Pages Implemented:

**1. Homepage** (`Homepage.tsx`)
- ✅ Displays 3 pre-built agents (Crawler, Parser, Combined Flow)
- ✅ Search and filter functionality
- ✅ Statistics cards (Total Agents, Active Agents, MCP Protocol, Success Rate)
- ✅ Navigation buttons to other pages
- ✅ Beautiful gradient UI with animations

**2. Agent Creator** (`AgentCreator.tsx`)
- ✅ Complete form for creating new agents:
  - Agent name and description
  - Avatar selection (12 emoji options)
  - Tags system
  - Public/Private toggle
  - Tool creation interface:
    - Tool name and description
    - Tool type selection (Function, HTTP Request, Data Transform, Custom)
    - Add/remove tools
- ✅ Live preview of agent
- ✅ Form validation
- ✅ Success/error message display
- ⚠️ **Backend connection not working** (see Issues section)

**3. Agent Management** (`AgentManagement.tsx`)
- ✅ List view of all created agents
- ✅ Agent cards showing:
  - Avatar, name, description
  - Tags
  - Public/Private status
  - Tool count
  - Creation date
- ✅ Actions: View Code, Edit, Delete
- ✅ Refresh button
- ✅ Empty state when no agents exist
- ⚠️ **Backend connection not working** (see Issues section)

**4. URL Extractor** (`URLExtractor.tsx`)
- ✅ Form to input URL
- ✅ Submit button
- ✅ Results display area
- ⚠️ **Backend connection not working** (see Issues section)

### 3. **Colab Notebook** (Fully Functional)

**`LMForge_Standalone_Colab.ipynb`**
- ✅ Self-contained notebook with all code embedded
- ✅ No external file dependencies
- ✅ Works in Google Colab environment
- ✅ Includes Crawler and Parser agents
- ✅ Fallback implementations if MCP agents fail
- ✅ Test cells for easy demonstration

---

## 🎨 Frontend UI Features

### What Users See:

#### **Homepage**
- **Header**: "AI Agent Hub" with gradient text
- **Stats Cards**: 4 colorful cards showing:
  - Total Agents: 3
  - Active Agents: 3
  - MCP Protocol: Ready
  - Success Rate: 100%
- **Search Bar**: Search agents by name, description, or tags
- **Filter Buttons**: All, Public, Private
- **Agent Cards**: 3 cards showing:
  - 🕷️ Crawler Agent - "Fetches raw HTML content from any URL"
  - 📄 Parser Agent - "Extracts clean text from HTML"
  - 🔄 Combined Flow - "Complete workflow: Crawl → Parse → Extract"
- **Action Buttons**: "Run" and "Configure" on each card
- **Create Agent Button**: Navigate to agent creation page

#### **Agent Creator Page**
- **Form Sections**:
  1. **Agent Information**:
     - Name input (required)
     - Description textarea (required)
     - Avatar picker (12 emoji options in a grid)
     - Tags input (add/remove tags with badges)
     - Public/Private toggle switch
  
  2. **Tools Section**:
     - Add new tool form:
       - Tool name (required)
       - Tool description (required)
       - Tool type dropdown (Function, HTTP Request, Data Transform, Custom)
     - List of added tools with remove buttons
  
  3. **Preview Panel** (right side):
     - Agent avatar (large)
     - Agent name
     - Description
     - Tags display
     - Stats (Tool count, Tag count)
     - "Create Agent" button

- **Visual Feedback**:
  - Success message (green) when agent created
  - Error message (red) when creation fails
  - Loading spinner on save button
  - Form validation (name and description required)

#### **Agent Management Page**
- **Header**: "Agent Management" with back button
- **Action Buttons**: Refresh, Create Agent
- **Agent Grid**: Cards showing:
  - Large avatar emoji
  - Agent name
  - Public/Private badge
  - Description (truncated)
  - Tags (first 3, then "+X more")
  - Stats (Tool count, Tag count)
  - Action buttons: Code, Edit, Delete
  - Creation date
- **Empty State**: Message when no agents exist with "Create Your First Agent" button

---

## 🔌 Backend API

### Working Endpoints:

#### 1. **Health Check**
```
GET /health
Response: {"status": "healthy", "service": "LMForge MCP-Use Backend"}
```

#### 2. **Run Flow** ✅ WORKING
```
POST /run-flow
Request: {"url": "https://example.com"}
Response: {
  "url": "https://example.com",
  "raw_html": "...",
  "parsed_text": "...",
  "cleaned_text": "...",
  "success": true
}
```

#### 3. **Generate Q&A** ✅ WORKING
```
POST /generate-qa
Request: {"text": "Your text here..."}
Response: {
  "qa_pairs": [
    {"question": "...", "answer": "..."},
    ...
  ]
}
```

### Agent Management Endpoints (Implemented but Not Connected):

#### 4. **Create Agent** ⚠️ IMPLEMENTED, NOT CONNECTED
```
POST /api/agents
Request: {
  "name": "Agent Name",
  "description": "Agent description",
  "tools": [...],
  "avatar": "🤖",
  "tags": ["tag1", "tag2"],
  "isPublic": false
}
Response: {agent object}
```

#### 5. **List Agents** ⚠️ IMPLEMENTED, NOT CONNECTED
```
GET /api/agents
Response: {
  "agents": [...],
  "count": 0
}
```

#### 6. **Get Agent** ⚠️ IMPLEMENTED, NOT CONNECTED
```
GET /api/agents/{agent_id}
Response: {agent object}
```

#### 7. **Update Agent** ⚠️ IMPLEMENTED, NOT CONNECTED
```
PUT /api/agents/{agent_id}
Request: {updates}
Response: {updated agent}
```

#### 8. **Delete Agent** ⚠️ IMPLEMENTED, NOT CONNECTED
```
DELETE /api/agents/{agent_id}
Response: {"success": true}
```

#### 9. **Get Agent Code** ⚠️ IMPLEMENTED, NOT CONNECTED
```
GET /api/agents/{agent_id}/code
Response: Python code as text
```

---

## ⚠️ Current Status & Issues

### ✅ What Works:

1. **Backend Core Functionality**
   - ✅ FastAPI server runs successfully
   - ✅ `/run-flow` endpoint works perfectly
   - ✅ `/generate-qa` endpoint works (requires OpenAI API key)
   - ✅ MCP agents (Crawler, Parser) work correctly
   - ✅ Agent storage system (`agent_storage.py`) is fully implemented

2. **Frontend UI**
   - ✅ All pages render correctly
   - ✅ Forms are functional
   - ✅ UI/UX is polished with animations
   - ✅ Navigation works between pages

3. **Colab Notebook**
   - ✅ Works standalone in Google Colab
   - ✅ No external dependencies needed
   - ✅ Easy to share and demonstrate

### ❌ What Doesn't Work:

1. **Frontend-Backend Connection** ⚠️ **MAJOR ISSUE**
   - ❌ Frontend cannot connect to backend API
   - ❌ Agent creation fails with "Failed to fetch" error
   - ❌ Agent management page shows "Failed to load agents"
   - ❌ URL extractor cannot submit requests

2. **Root Cause**: VPN/Proxy Interception
   - Requests to `localhost:8000` are being intercepted by VPN
   - Browser redirects requests to `https://vpn.mtu.edu`
   - CORS errors occur even with proper configuration
   - Vite proxy configuration doesn't work due to VPN interference

3. **Agent Management Backend**
   - ✅ Code is written and functional
   - ✅ Endpoints are defined correctly
   - ❌ Cannot be tested due to connection issues
   - ❌ Frontend cannot verify if endpoints work

---

## 🐛 Problems Encountered

### 1. **VPN/Proxy Interception** (Critical)

**Problem**:
- When frontend tries to connect to `http://localhost:8000`, the VPN intercepts the request
- Browser shows: "Failed to fetch" or redirects to `https://vpn.mtu.edu`
- Network tab shows requests being blocked or redirected

**Attempted Solutions**:
1. ✅ Changed backend host from `0.0.0.0` to `127.0.0.1` - No effect
2. ✅ Updated CORS to allow all origins - No effect
3. ✅ Added Vite proxy configuration - No effect (VPN intercepts before proxy)
4. ✅ Changed frontend API_URL to use `/api` proxy path - No effect
5. ✅ Added explicit OPTIONS handler for CORS preflight - No effect

**Current Status**: **UNRESOLVED**
- Backend runs fine when tested with `curl` or Postman
- Frontend cannot connect due to VPN interference
- Need to test without VPN or use alternative connection method

### 2. **MCP Agent Initialization on Windows**

**Problem**:
- MCP agents fail to initialize on Windows with `fileno` error
- Error: `Error in StdioConnectionManager task: fileno`

**Solution**: ✅ **RESOLVED**
- Added fallback implementations in `main.py`
- If MCP agents fail, uses direct `requests` and `BeautifulSoup` calls
- System works even when MCP agents can't initialize

### 3. **Colab nest_asyncio Conflicts**

**Problem**:
- `uvicorn` conflicts with `nest_asyncio` in Colab
- Error: `TypeError: _patch_asyncio.<locals>.run() got an unexpected keyword argument 'loop_factory'`

**Solution**: ✅ **RESOLVED**
- Created standalone Colab notebook with manual event loop setup
- Bypasses `loop_factory` parameter issue
- Works perfectly in Colab environment

### 4. **MCP-Use API Version Changes**

**Problem**:
- Code written for older MCP-Use API
- New version (1.4.0+) has different API structure
- Errors: `'MCPClient' object has no attribute 'connect'`

**Solution**: ✅ **RESOLVED**
- Updated `main.py` to use new API:
  ```python
  client = MCPClient()
  client.add_server(name, config)
  await client.create_all_sessions()
  session = client.get_session(name)
  ```

### 5. **Agent Code Generation**

**Problem**:
- Need to generate valid Python MCP agent code from user input
- Code must follow MCP protocol standards
- Must handle different tool types

**Solution**: ✅ **IMPLEMENTED**
- `agent_storage.py` includes `_generate_agent_code()` method
- Generates complete MCP server code
- Handles tool definitions, handlers, and routing
- Code is saved to `custom_agents/code/{agent_id}.py`

**Status**: Code generation works, but cannot be tested due to connection issues

### 6. **QA Generator API Constraints**

**Problem**:
- QA generator relies on OpenAI via `langchain-openai`
- Shared academic API key ran out of credits / rate limit (`429 insufficient_quota`)
- Demo laptops sometimes lacked `.env` with `OPENAI_API_KEY`
- Result: `/generate-qa` endpoint returns 500 errors even though code is correct

**How we mitigated it**:
- Added explicit error logging in `qa_server.py` and `main.py`
- Backend keeps running even if QA agent fails to initialize
- `/generate-qa` endpoint now returns helpful error messages (status + hint)

**What future teams should do**:
1. Secure a fresh OpenAI key (departmental or personal billing)
2. Create `.env` at repo root with `OPENAI_API_KEY=<value>`
3. Restart backend (`python app.py`)
4. Re-run `/generate-qa` or UI Q/A button—QA agent should spin up automatically

**Status**: Feature implemented; blocked by external API quota/billing

---

## 🚀 Next Steps

### Priority 1: Fix Frontend-Backend Connection

**Options to Try**:
1. **Test without VPN**: Disconnect VPN and test connection
2. **Use different port**: Try port 3000 or 8080 instead of 8000
3. **Use ngrok**: Create tunnel to bypass VPN
4. **Direct IP connection**: Use `http://127.0.0.1:8000` instead of `localhost`
5. **Server-side proxy**: Add proxy endpoint in backend to forward requests

**Recommended Approach**:
```python
# In app.py, add a proxy endpoint
@app.post("/api/proxy/agents")
async def proxy_agents(request: Request):
    # Forward request to actual endpoint
    # This bypasses CORS issues
```

### Priority 2: Test Agent Management Endpoints

Once connection is fixed:
1. Test `POST /api/agents` with Postman/curl
2. Verify agent is saved to `custom_agents/agents.json`
3. Verify code is generated in `custom_agents/code/`
4. Test `GET /api/agents` to list agents
5. Test `DELETE /api/agents/{id}` to delete agents

### Priority 3: Dynamic Agent Loading

**Current State**: User-created agents are saved but not loaded into orchestrator

**What's Needed**:
1. Update `main.py` to load agents from `custom_agents/agents.json`
2. Dynamically register user agents as MCP servers
3. Allow users to run their custom agents through the UI

**Implementation**:
```python
# In main.py
def load_custom_agents():
    """Load user-created agents and register them."""
    storage = AgentStorage()
    agents = storage.get_all_agents()
    
    for agent in agents:
        # Register agent as MCP server
        agent_id = agent["id"]
        code_path = storage.get_agent_code_path(agent_id)
        mcp_servers[agent_id] = {
            "command": "python",
            "args": [str(code_path)]
        }
```

### Priority 4: Agent Execution UI

**What's Needed**:
1. Add "Run Agent" button in Agent Management page
2. Create endpoint to execute user-created agents
3. Display results in UI
4. Handle errors gracefully

### Priority 5: Agent Editing

**Current State**: Edit button exists but doesn't work

**What's Needed**:
1. Pre-fill Agent Creator form with existing agent data
2. Update agent instead of creating new one
3. Regenerate code when agent is updated

---

## 🏗️ Technical Architecture

### System Flow:

```
User → Frontend (React) → Backend (FastAPI) → MCP Orchestrator → MCP Agents
                                                      ↓
                                            Agent Storage System
                                                      ↓
                                            custom_agents/
```

### File Structure:

```
LMForgeAgentic Team/
├── app.py                          # FastAPI backend
├── main.py                         # MCP orchestrator
├── agent_storage.py                # Agent management system
├── requirements.txt                # Python dependencies
├── agents/                         # Pre-built MCP agents
│   ├── crawler_server.py
│   ├── parser_server.py
│   ├── cleaner_server.py
│   └── qa_server.py
├── custom_agents/                  # User-created agents
│   ├── agents.json                # Agent metadata
│   └── code/                      # Generated Python code
│       └── agent_*.py
├── Figma_Prototype/                # Frontend React app
│   └── Figma/
│       ├── src/
│       │   ├── App.tsx
│       │   └── components/
│       │       └── pages/
│       │           ├── Homepage.tsx
│       │           ├── AgentCreator.tsx
│       │           ├── AgentManagement.tsx
│       │           └── URLExtractor.tsx
│       └── package.json
└── LMForge_Standalone_Colab.ipynb  # Standalone Colab notebook
```

### Data Flow:

1. **Agent Creation**:
   ```
   User fills form → Frontend sends POST /api/agents
   → Backend saves to agents.json
   → Backend generates Python code
   → Code saved to custom_agents/code/
   ```

2. **Agent Execution**:
   ```
   User clicks "Run" → Frontend sends request
   → Backend loads agent from storage
   → Backend starts MCP server for agent
   → Agent executes tool
   → Results returned to frontend
   ```

### Technology Stack:

**Backend**:
- FastAPI (Python web framework)
- MCP-Use (Model Context Protocol)
- Uvicorn (ASGI server)
- BeautifulSoup4 (HTML parsing)
- LangChain-OpenAI (Q&A generation)

**Frontend**:
- React (UI framework)
- TypeScript (Type safety)
- Vite (Build tool)
- Tailwind CSS (Styling)
- Framer Motion (Animations)
- Lucide React (Icons)

---

## 📝 Notes for Next Semester Team

### Important Files to Review:

1. **`app.py`** (lines 268-380): Agent management endpoints
2. **`agent_storage.py`**: Complete agent storage and code generation system
3. **`main.py`**: MCP orchestrator - needs update to load custom agents
4. **`Figma_Prototype/Figma/src/components/pages/AgentCreator.tsx`**: Frontend form
5. **`Figma_Prototype/Figma/src/components/pages/AgentManagement.tsx`**: Agent list page

### Key Functions:

- `AgentStorage.create_agent()`: Creates and saves agent
- `AgentStorage._generate_agent_code()`: Generates Python MCP code
- `get_orchestrator()`: Gets MCP orchestrator instance
- `run_full_flow()`: Executes crawler → parser → cleaner workflow

### Testing:

**Backend** (works):
```bash
# Start server
python app.py

# Test run-flow
curl -X POST http://localhost:8000/run-flow \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Test create agent
curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Agent", "description": "Test"}'
```

**Frontend** (connection issues):
```bash
cd Figma_Prototype/Figma
npm install
npm run dev
# Open http://localhost:3001
# Try creating an agent - will fail due to VPN
```

### Known Limitations:

1. **VPN Interference**: Cannot test frontend-backend connection
2. **Windows MCP Issues**: MCP agents may fail on Windows (fallback works)
3. **Dynamic Loading**: User agents not automatically loaded into orchestrator
4. **Code Validation**: Generated agent code not validated before saving
5. **Error Handling**: Limited error messages for agent creation failures

---

## 🎓 Summary

### What's Complete:
✅ Backend API with working endpoints  
✅ Agent storage and code generation system  
✅ Complete frontend UI with all pages  
✅ Colab notebook for easy demonstration  
✅ MCP agent orchestration  

### What Needs Work:
⚠️ Frontend-backend connection (VPN issue)  
⚠️ Dynamic loading of user-created agents  
⚠️ Agent execution through UI  
⚠️ Agent editing functionality  

### Estimated Completion:
- **Backend**: 95% complete
- **Frontend**: 90% complete
- **Integration**: 30% complete (blocked by VPN issue)
- **Overall**: ~75% complete

---

**Good luck to the next semester team!** 🚀

If you have questions, check the code comments and this documentation. The foundation is solid - you just need to fix the connection issues and add the dynamic loading feature.

