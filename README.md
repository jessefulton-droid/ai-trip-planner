# AI Surf Trip Planner

A **production-ready multi-agent system** for planning perfect surf trips. This repo demonstrates three essential AI engineering patterns that students can study, modify, and adapt for their own use cases - now specialized for surf travel planning.

## What You'll Learn

- 🤖 **Multi-Agent Orchestration**: 4 specialized surf agents running in parallel using LangGraph
- 🔍 **RAG (Retrieval-Augmented Generation)**: Vector search over 25 world-class surf spots with fallback strategies
- 🌐 **API Integration**: Real-time web search with graceful degradation (LLM fallback)
- 📊 **Observability**: Production tracing with Arize for debugging and evaluation
- 🏄 **Surf-Specific Intelligence**: Break analysis, swell forecasts, surf etiquette, and session planning

**Perfect for:** Learning agentic AI systems through a practical surf trip planning application.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Surfer's Request                            │
│         (surf destination, duration, skill, preferences)         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   FastAPI Endpoint      │
                    │   + Session Tracking    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   LangGraph Workflow    │
                    │   (Parallel Execution)  │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   ┌────▼─────┐           ┌─────▼──────┐         ┌──────▼─────┐
   │Surf Spot │           │Surf Budget │         │Local Surf  │
   │ Research │           │   Agent    │         │ Culture    │
   └────┬─────┘           └─────┬──────┘         └──────┬─────┘
        │                        │                        │
        │ Tools:                 │ Tools:                 │ Tools + RAG:
        │ • surf_spot_info       │ • surf_trip_budget     │ • local_surf_scene
        │ • surf_forecast_brief  │ • surf_services_pricing│ • surf_etiquette
        │ • visa_surf_gear_brief │                        │ • secret_spots
        │                        │                        │ • Vector search
        │                        │                        │   (25 surf spots)
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                            ┌────▼─────┐
                            │Surf Trip │
                            │Itinerary │
                            │  Agent   │
                            └────┬─────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Surf Trip Itinerary    │
                    │  + Session Timing       │
                    │  + Tool Call Metadata   │
                    └─────────────────────────┘

All agents, tools, and LLM calls → Arize Observability Platform
```

## Learning Paths

### 🎓 Beginner Path
1. **Setup & Run** (15 min)
   - Clone repo, configure `.env` with OpenAI key
   - Start server: `./start.sh`
   - Test API: `python "test scripts/test_api.py"`

2. **Observe & Understand** (30 min)
   - Make surf trip planning requests (try "Pipeline, Hawaii" or "J-Bay, South Africa")
   - View traces in Arize dashboard
   - Understand surf agent execution flow and tool calls

3. **Experiment with Prompts** (30 min)
   - Modify surf agent prompts in `backend/main.py`
   - Change surf tool descriptions
   - See how it affects surf itineraries

### 🚀 Intermediate Path
1. **Enable Advanced Features** (20 min)
   - Set `ENABLE_RAG=1` to use vector search
   - Add `TAVILY_API_KEY` for real-time web search
   - Compare results with/without these features

2. **Add Custom Data** (45 min)
   - Add your own surf spot to `backend/data/surf_spots.json`
   - Test RAG retrieval with your data
   - Understand fallback strategies

3. **Create a New Tool** (1 hour)
   - Add a new tool (e.g., `tide_calculator`, `board_recommender`)
   - Integrate it into a surf agent
   - Test and trace the new tool calls

### 💪 Advanced Path
1. **Change the Domain** (2-3 hours)
   - Use Cursor AI to help transform the system
   - Example: Change from "surf trip planner" to another specialized domain
   - Modify state, agents, and tools for your use case

2. **Add a New Agent** (2 hours)
   - Create a 5th agent (e.g., "surf gear specialist", "tide analyzer")
   - Update the LangGraph workflow
   - Test parallel vs sequential execution

3. **Implement Evaluations** (2 hours)
   - Use `test scripts/synthetic_data_gen.py` as a base
   - Create evaluation criteria for surf trip quality
   - Set up automated evals in Arize

## Common Use Cases (Built by Students)

Students have successfully adapted this codebase for:

- **📝 PR Description Generator**
  - Agents: Code Analyzer, Context Gatherer, Description Writer
  - Replaces travel tools with GitHub API calls
  - Used by tech leads to auto-generate PR descriptions

- **🎯 Customer Support Analyst**
  - Agents: Ticket Classifier, Knowledge Base Search, Response Generator
  - RAG over support docs instead of local guides
  - Routes tickets and drafts responses

- **🔬 Research Assistant**
  - Agents: Web Searcher, Academic Search, Citation Manager, Synthesizer
  - Web search for papers + RAG over personal library
  - Generates research summaries with citations

- **📱 Content Planning System**
  - Agents: SEO Researcher, Social Media Planner, Blog Scheduler
  - Tools for keyword research, trend analysis
  - Creates cross-platform content calendars

- **🏗️ Architecture Review Agent**
  - Agents: Code Scanner, Pattern Detector, Best Practices Checker
  - RAG over architecture docs
  - Reviews PRs for architectural concerns

**💡 Your Turn**: Use Cursor AI to help you adapt this system for your domain!

## Quickstart

1) Requirements
- Python 3.10+ (Docker optional)

2) Configure environment
- Copy `backend/.env.example` to `backend/.env`.
- Set one LLM key: `OPENAI_API_KEY=...` or `OPENROUTER_API_KEY=...`.
- **Recommended**: `ARIZE_SPACE_ID` and `ARIZE_API_KEY` for complete observability.
  - See [ARIZE_SETUP.md](./ARIZE_SETUP.md) for quick setup guide
  - See [ARIZE_OBSERVABILITY.md](./ARIZE_OBSERVABILITY.md) for complete docs

3) Install dependencies
```bash
cd backend
uv pip install -r requirements.txt   # faster, deterministic installs
# If uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh
# Fallback: pip install -r requirements.txt
```

4) Run
```bash
# make sure you are back in the root directory of ai-trip-planner
cd ..
./start.sh                      # starts backend on 8000; serves minimal UI at '/'
# or
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

5) Open
- Frontend: http://localhost:3000
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
 - Minimal UI: http://localhost:8000/

Docker (optional)
```bash
docker-compose up --build
```

## Project Structure
- `backend/`: FastAPI app (`main.py`), LangGraph surf agents, tracing hooks.
- `backend/data/surf_spots.json`: Curated database of 25 world-class surf destinations.
- `frontend/index.html`: Surf trip planner UI served by backend at `/`.
- `optional/airtable/`: Airtable integration (optional, not on critical path).
- `test scripts/`: `test_api.py`, `synthetic_data_gen.py` for quick checks/evals.
- Root: `start.sh`, `docker-compose.yml`, `README.md`.

## Development Commands
- Backend (dev): `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- API smoke test: `python "test scripts"/test_api.py`
- Synthetic evals: `python "test scripts"/synthetic_data_gen.py --base-url http://localhost:8000 --count 12`

## API
- POST `/plan-trip` → returns a generated surf trip itinerary.
  Example body:
  ```json
  {"destination":"Pipeline, North Shore, Hawaii","duration":"7 days","budget":"moderate","surf_preferences":"reef breaks, barrels","skill_level":"advanced"}
  ```
- GET `/health` → simple status.

## Arize AX Observability (Recommended)

This application is fully instrumented with **Arize AX** for production-grade observability:

### What You Get
- 🎯 **Multi-Agent Workflow Visualization**: See all 4 agents (research, budget, local, itinerary) executing in parallel
- 🤖 **LLM Tracing**: Track every prompt, completion, token usage, and latency
- 🔧 **Tool Call Monitoring**: Debug which tools agents call and their results
- 📊 **RAG Observability**: Visualize vector search, document retrieval, and scores
- 👤 **Session Tracking**: Follow user journeys across multiple requests
- ⚠️ **Error Analysis**: Get detailed exception traces with full context
- ⏱️ **Performance Metrics**: Identify bottlenecks and optimize latency
- 💰 **Cost Tracking**: Monitor token usage and LLM costs

### Setup (5 minutes)

1. Sign up at [https://app.arize.com](https://app.arize.com) (free tier available)
2. Get your Space ID and API Key from Space Settings
3. Add to `backend/.env`:
   ```bash
   ARIZE_SPACE_ID=your-space-id
   ARIZE_API_KEY=your-api-key
   ```
4. Start the server - tracing is automatic!

### Quick Start

```bash
# 1. Install dependencies
cd backend && pip install -r requirements.txt

# 2. Configure Arize credentials
cp .env.example .env
nano .env  # Add ARIZE_SPACE_ID and ARIZE_API_KEY

# 3. Start server
cd .. && ./start.sh

# 4. Make a request
curl -X POST http://localhost:8000/plan-trip \
  -H "Content-Type: application/json" \
  -d '{"destination": "Pipeline, Hawaii", "duration": "7 days", "session_id": "test-123"}'

# 5. View trace at https://app.arize.com
```

### Documentation

- **Quick Setup**: [ARIZE_SETUP.md](./ARIZE_SETUP.md) - Get started in 5 minutes
- **Complete Guide**: [ARIZE_OBSERVABILITY.md](./ARIZE_OBSERVABILITY.md) - Full documentation

### What's Instrumented

✅ Auto-instrumentation via OpenInference:
- LangChain/LangGraph workflows
- OpenAI API calls
- LiteLLM (for other providers)

✅ Manual instrumentation for agents:
- Proper OpenInference span kinds (AGENT, RETRIEVER, CHAIN)
- Input/output tracking
- Prompt template versioning
- RAG document retrieval with scores
- Error handling and status codes
- Custom metadata and tags

### View in Arize

After making requests, explore your traces:
1. Go to [app.arize.com](https://app.arize.com)
2. Select your Space
3. Find project: **ai-surf-trip-planner**
4. View traces, analyze performance, test prompts in Playground

## Optional Features

### RAG: Vector Search for Surf Spots

The local surf culture agent can use vector search to retrieve curated surf spot information from a database of 25 world-class destinations:

- **Enable**: Set `ENABLE_RAG=1` in your `.env` file
- **Requirements**: Requires `OPENAI_API_KEY` for embeddings
- **Data**: Uses curated surf spots from `backend/data/surf_spots.json`
- **Coverage**: Pipeline, J-Bay, Teahupo'o, Uluwatu, Trestles, Mentawais, and 19 more world-class breaks
- **Benefits**: Provides grounded, cited surf recommendations with break characteristics
- **Learning**: Great example of production RAG patterns with fallback strategies

When disabled (default), the local agent uses LLM-generated responses.

See `RAG.md` for detailed documentation.

### Web Search: Real-Time Surf Data

Tools can call real web search APIs (Tavily or SerpAPI) for up-to-date surf information:

- **Enable**: Add `TAVILY_API_KEY` or `SERPAPI_API_KEY` to your `.env` file
- **Benefits**: Real-time data for surf forecasts, swell conditions, break reports, surf services pricing, etc.
- **Fallback**: Without API keys, tools automatically fall back to LLM-generated responses
- **Learning**: Demonstrates graceful degradation and multi-tier fallback patterns
- **Future**: Could integrate Surfline, Magicseaweed, or Stormglass APIs for real surf forecasts

Recommended: Tavily (free tier: 1000 searches/month) - https://tavily.com

## Next Steps

1. **🎯 Start Simple**: Get it running, make some requests, view traces
2. **🔍 Explore Code**: Read through `backend/main.py` to understand patterns
3. **🛠️ Modify Prompts**: Change agent behaviors to see what happens
4. **🚀 Enable Features**: Try RAG and web search
5. **💡 Build Your Own**: Use Cursor to transform it into your agent system

## Troubleshooting

- **401/empty results**: Verify `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in `backend/.env`
- **No traces**: Ensure Arize credentials are set and reachable
- **Port conflicts**: Stop existing services on 3000/8000 or change ports
- **RAG not working**: Check `ENABLE_RAG=1` and `OPENAI_API_KEY` are both set
- **Slow responses**: Web search APIs may timeout; LLM fallback will handle it

## Deploy on Render
- This repo includes `render.yaml`. Connect your GitHub repo in Render and deploy as a Web Service.
- Render will run: `pip install -r backend/requirements.txt` and `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Set `OPENAI_API_KEY` (or `OPENROUTER_API_KEY`) and optional Arize vars in the Render dashboard.
