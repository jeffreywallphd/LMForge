# LMForge MCP-Use + FastAPI Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A fully open-source backend framework for LMForge that combines MCP-Use (Model Context Protocol) with FastAPI, designed to run seamlessly in Google Colab.

## Features

- **MCP-Based Agent Orchestration**: Modular agents using the Model Context Protocol
- **FastAPI Backend**: High-performance REST API endpoints
- **Web Crawling & Parsing**: Extract and process web content
- **AI-Powered Q&A Generation**: Automatically generate question/answer pairs from text
- **Google Colab Ready**: Pre-configured for easy deployment in Colab notebooks
- **Fully Open Source**: Apache-2.0 licensed

## Architecture

```
LMForgeAgentic Team/
├── app.py                  # FastAPI application with endpoints
├── main.py                 # MCP client registration and orchestration
├── requirements.txt        # Python dependencies
├── agents/
│   ├── crawler_server.py   # Web crawling agent
│   ├── parser_server.py    # HTML parsing agent
│   ├── cleaner_server.py   # Text cleaning agent
│   └── qa_server.py        # Q&A generation agent
├── LMForge_Standalone_Colab.ipynb  # Google Colab notebook
├── custom_agents/          # User-created agents storage
├── agent_storage.py        # Agent management system
├── Figma_Prototype/        # Frontend React application
├── LICENSE                 # Apache-2.0 license
└── README.md              # This file
```

## Quick Start

### Option 1: Google Colab (Recommended)

1. Open `LMForge_Standalone_Colab.ipynb` in Google Colab
2. Set your OpenAI API key in the notebook (if using QA Generator)
3. Run all cells from top to bottom
4. Use the provided test cells to interact with the API

### Option 2: Local Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd LMForgeAgentic\ Team

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run the server
python app.py
```

## API Endpoints

### 1. Run Full Flow

**Endpoint**: `POST /run-flow`

Executes the complete workflow: crawl → parse → clean a web page.

**Request Body**:
```json
{
  "url": "https://example.com"
}
```

**Response**:
```json
{
  "url": "https://example.com",
  "raw_html": "...",
  "parsed_text": "...",
  "cleaned_text": "..."
}
```

### 2. Generate Q&A

**Endpoint**: `POST /generate-qa`

Generates 3 concise question/answer pairs from provided text using AI.

**Request Body**:
```json
{
  "text": "Your text content here..."
}
```

**Response**:
```json
{
  "qa_pairs": [
    {
      "question": "What is...?",
      "answer": "..."
    },
    {
      "question": "How does...?",
      "answer": "..."
    },
    {
      "question": "Why is...?",
      "answer": "..."
    }
  ]
}
```

## Agent System

The framework uses a modular MCP-based agent architecture:

### Crawler Agent (`crawler_server.py`)
- Fetches web pages
- Handles HTTP requests
- Returns raw HTML content

### Parser Agent (`parser_server.py`)
- Extracts text from HTML
- Uses BeautifulSoup4
- Removes scripts and styles

### Cleaner Agent (`cleaner_server.py`)
- Normalizes whitespace
- Removes special characters
- Formats clean text

### Q&A Generator Agent (`qa_server.py`)
- Uses OpenAI LLM
- Generates contextual questions
- Provides concise answers

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required for Q&A generation)
- `PORT`: Server port (default: 8000)

### MCP Configuration

Agents are registered in `main.py` using the MCP protocol:

```python
mcp_servers = {
    "crawler": {"command": "python", "args": ["agents/crawler_server.py"]},
    "parser": {"command": "python", "args": ["agents/parser_server.py"]},
    "cleaner": {"command": "python", "args": ["agents/cleaner_server.py"]},
    "qa_generator": {"command": "python", "args": ["agents/qa_server.py"]}
}
```

## Dependencies

- **fastapi**: Modern web framework
- **uvicorn**: ASGI server
- **mcp-use**: Model Context Protocol client
- **langchain-openai**: OpenAI integration
- **python-dotenv**: Environment variable management
- **nest_asyncio**: Async support for Colab
- **requests**: HTTP library
- **beautifulsoup4**: HTML parsing

## Development

### Adding New Agents

1. Create a new agent file in the `agents/` directory
2. Implement the agent using MCP standards
3. Register the agent in `main.py`
4. Add corresponding endpoints in `app.py`

### Testing

```bash
# Test the run-flow endpoint
curl -X POST http://localhost:8000/run-flow \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Test the generate-qa endpoint
curl -X POST http://localhost:8000/generate-qa \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

## Google Colab Tips

- Use `nest_asyncio.apply()` to enable async operations
- Run uvicorn in a background thread for interactive testing
- Set environment variables using `os.environ`
- Use `ngrok` for external access (optional)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

Common issues and solutions:

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
# Or use: ./kill_server.sh
```

### MCPClient Connection Issues
Make sure to use the latest `main.py` which properly initializes MCP clients:
```python
client = MCPClient.from_dict(config)  # Returns client directly
await client.connect()  # Then connect
```

### Colab nest_asyncio Conflicts
See `COLAB_QUICKFIX.md` for detailed solutions.

For more issues, see `TROUBLESHOOTING.md`.

## How to Use

### Quick Start

1. **Start the server:**
   ```bash
   python app.py
   ```

2. **Open API docs in browser:**
   ```
   http://localhost:8000/docs
   ```

3. **Test the `/run-flow` endpoint:**
   - Click on `POST /run-flow`
   - Click "Try it out"
   - Paste a URL: `{"url": "https://example.com"}`
   - Click "Execute"
   - See the results!

### What Happens:

1. **Crawler Agent** fetches the HTML from the URL
2. **Parser Agent** extracts clean text from HTML
3. **Result** shows raw HTML, parsed text, and cleaned text

## Why MCP Agents are Different

Traditional scraping is monolithic - all code in one place. Our MCP agent system is:

- **Modular** - Each agent is independent
- **Protocol-Based** - Uses Model Context Protocol standard
- **Scalable** - Easy to add new agents
- **Reusable** - Agents work across projects
- **Orchestrated** - Central coordination system

See [DEMO_GUIDE.md](DEMO_GUIDE.md) for detailed explanation.

## Support

For issues, questions, or contributions, please open an issue on GitHub.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Uses [MCP-Use](https://github.com/modelcontextprotocol/python-sdk) for agent orchestration
- Powered by [OpenAI](https://openai.com/) for Q&A generation

---

Made with ❤️ for the LMForge community

