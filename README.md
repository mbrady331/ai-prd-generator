# 🚀 AI-Powered PRD Generator

A sophisticated Product Requirements Document (PRD) generator that uses multiple AI agents to research, analyze, and create comprehensive PRDs. Built with FastAPI, LangGraph, and a beautiful modern frontend.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **Multi-Agent Architecture**: 4 specialized AI agents work in parallel to research different aspects
  - Market Research Agent
  - User Research Agent  
  - Technical Research Agent
  - Risk Assessment Agent
- **Web Search Integration**: Real-time market and competitive research via Tavily/SerpAPI
- **Intelligent Fallbacks**: Works with or without API keys, includes test mode
- **Beautiful UI**: Modern, responsive frontend with gradient design
- **Observability**: Optional integration with Arize for tracing and monitoring
- **Professional Output**: Generates comprehensive, well-structured PRDs

## 🏗️ Architecture

```
Frontend (HTML/JS) → FastAPI Backend → LangGraph → Multiple AI Agents
                                          ↓
                                    Research Tools (Web Search)
                                          ↓
                                    LLM Synthesis → PRD Output
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the template
cp .env.template .env

# Edit .env and add your API keys (at minimum, you need one LLM key)
```

**Required:**
- `OPENAI_API_KEY` OR `OPENROUTER_API_KEY` (for LLM functionality)

**Optional (for enhanced research):**
- `TAVILY_API_KEY` (recommended for web search)
- `SERPAPI_API_KEY` (alternative search API)

### 3. Run the Application

```bash
# Start the backend server
cd backend
python main.py
```

The server will start on `http://localhost:8000`

### 4. Access the Frontend

Open your browser to `http://localhost:8000` to see the beautiful PRD generator interface.

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes* | OpenAI API key for LLM |
| `OPENROUTER_API_KEY` | Yes* | Alternative LLM provider |
| `OPENROUTER_MODEL` | No | Model to use (default: gpt-4o-mini) |
| `TAVILY_API_KEY` | No | Web search API for research |
| `SERPAPI_API_KEY` | No | Alternative web search API |
| `ARIZE_SPACE_ID` | No | For observability tracing |
| `ARIZE_API_KEY` | No | For observability tracing |
| `TEST_MODE` | No | Set to "1" for test mode (mock LLM) |

*At least one LLM provider is required

### Test Mode

Set `TEST_MODE=1` in your `.env` file to run without API keys. This uses a mock LLM for testing the application flow.

## 📋 Usage

1. **Fill out the form** with your product details:
   - Basic information (name, version, manager)
   - Problem statement and solution
   - Target users
   - Key features with priorities
   - Technical requirements
   - Success metrics
   - Timeline and milestones
   - Risks and constraints

2. **Click "Generate PRD"** - The system will:
   - Send your data to multiple AI research agents
   - Conduct web research (if APIs configured)
   - Synthesize findings into a comprehensive PRD
   - Return a professional markdown document

3. **Copy and use** the generated PRD for your product development

## 🛠️ Development

### Project Structure

```
prd-agent/
├── backend/
│   └── main.py          # FastAPI application with AI agents
├── frontend/
│   └── index.html       # Beautiful web interface
├── requirements.txt     # Python dependencies
├── .env.template       # Environment configuration template
└── README.md           # This file
```

### API Endpoints

- `GET /` - Serves the frontend
- `GET /health` - Health check
- `POST /generate-prd` - Generate PRD (main endpoint)

### Adding New Agents

The system is designed to be extensible. To add a new research agent:

1. Create a new agent function in `main.py`
2. Add it to the LangGraph workflow
3. Create corresponding tools if needed
4. Update the state schema

## 🔍 How It Works

1. **Input Processing**: User fills out comprehensive form
2. **Parallel Research**: 4 agents simultaneously research different aspects:
   - Market trends and competition
   - User needs and pain points  
   - Technical considerations
   - Risk assessment
3. **Tool Integration**: Agents use web search APIs for real-time data
4. **Synthesis**: Final agent combines all research into structured PRD
5. **Output**: Professional markdown document ready for use

## 🎯 Example PRD Request

```json
{
  "product_name": "TaskFlow Pro",
  "executive_summary": "A project management tool for remote teams",
  "problem_statement": "Remote teams struggle with task coordination and progress visibility",
  "primary_users": "Remote team managers and project coordinators",
  "features": [
    {
      "name": "Task Board",
      "description": "Kanban-style task management",
      "priority": "P0"
    }
  ],
  "platforms": ["Web", "Mobile"],
  "tech_stack": "React, Node.js, PostgreSQL"
}
```

## 🚨 Troubleshooting

### Common Issues

1. **"No API key configured"** - Add at least one LLM API key to `.env`
2. **Frontend not loading** - Ensure backend is running on port 8000
3. **Search not working** - Add Tavily or SerpAPI key for enhanced research
4. **Slow responses** - Research agents can take 30-60 seconds for comprehensive analysis

### Debug Mode

Set `TEST_MODE=1` to run with mock responses for testing the application flow.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - feel free to use this for your projects!

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- AI orchestration via [LangGraph](https://github.com/langchain-ai/langgraph)
- Beautiful UI inspired by modern design principles
- Research capabilities powered by [Tavily](https://tavily.com/) and [SerpAPI](https://serpapi.com/)

---

**Happy PRD generating! 🎉**
