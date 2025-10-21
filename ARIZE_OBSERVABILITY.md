# Arize AX Observability Setup Guide

This document explains how to use **Arize AX** (not Phoenix) for complete observability of your multi-agent surf trip planner.

## 🎯 What You'll Get

With Arize AX observability, you can visualize and debug:

- **Multi-Agent Workflows**: See how 4 specialized agents (research, budget, local, itinerary) execute in parallel
- **LLM Interactions**: Track all prompts, completions, token usage, and latency
- **Tool Calls**: Monitor which tools each agent calls and their results
- **RAG Operations**: Visualize vector search, document retrieval, and context injection
- **Session Tracking**: Follow user journeys across multiple requests
- **Error Analysis**: Identify failures with detailed exception traces
- **Performance Metrics**: Analyze execution times and bottlenecks

## 📋 Prerequisites

1. **Arize Account**: Sign up at [https://app.arize.com](https://app.arize.com)
2. **API Credentials**: Get your Space ID and API Key from Space Settings

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

The requirements include:
- `arize-otel` - Arize's OpenTelemetry convenience wrapper
- `openinference-instrumentation-*` - Auto-instrumentation for LangChain, OpenAI, LiteLLM
- `openinference-semconv` - Semantic conventions for AI observability
- `opentelemetry-*` - Core OpenTelemetry SDK and exporters

### 2. Configure Environment Variables

Create or update `backend/.env`:

```bash
# Arize AX Configuration (REQUIRED for tracing)
ARIZE_SPACE_ID=your-space-id-here
ARIZE_API_KEY=your-api-key-here

# LLM Provider (at least one required)
OPENAI_API_KEY=your-openai-key
# OR
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_MODEL=openai/gpt-4o-mini

# Optional: Enable RAG (requires OpenAI for embeddings)
ENABLE_RAG=1

# Optional: Web Search APIs
TAVILY_API_KEY=your-tavily-key
# OR
SERPAPI_API_KEY=your-serpapi-key
```

### 3. Start the Application

```bash
# From project root
./start.sh

# OR from backend directory
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
✅ Arize AX tracing initialized successfully
   Project: ai-surf-trip-planner
   View traces at: https://app.arize.com
```

### 4. Make a Request

Test the API:

```bash
curl -X POST http://localhost:8000/plan-trip \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Pipeline, North Shore, Hawaii",
    "duration": "7 days",
    "budget": "moderate",
    "surf_preferences": "reef breaks, barrels",
    "skill_level": "advanced",
    "session_id": "test-session-123",
    "user_id": "test-user"
  }'
```

### 5. View Traces in Arize

1. Go to [https://app.arize.com](https://app.arize.com)
2. Select your Space
3. Find the project: **ai-surf-trip-planner**
4. Browse traces, filter by session/user, and analyze performance

## 🔍 What's Instrumented

### Auto-Instrumentation (Automatic)

The following are automatically traced with **zero code changes** in your business logic:

1. **LangChain Components**
   - All chains, agents, and tools
   - Input/output tracking
   - Execution metadata

2. **OpenAI Calls**
   - Chat completions
   - Token usage
   - Model parameters
   - Latency

3. **LiteLLM** (if using other providers)
   - Universal LLM gateway
   - Multi-provider support

### Manual Instrumentation (Custom)

Enhanced instrumentation has been added to:

1. **Agent Nodes**
   - `research_agent`: Surf spot intelligence gathering
   - `budget_agent`: Cost analysis and breakdown
   - `local_agent`: Culture research + RAG retrieval
   - `itinerary_agent`: Final synthesis orchestrator

   Each agent includes:
   - OpenInference span kind (AGENT)
   - Input/output values
   - Metadata (destination, duration, skill level, etc.)
   - Event markers (started, tool calls, completed)
   - Error handling with exception tracking
   - Status codes (OK/ERROR)

2. **RAG Operations**
   - Vector search spans (RETRIEVER kind)
   - Retrieved documents with scores
   - Document metadata
   - Context injection tracking

3. **Workflow Orchestration**
   - Parent span wrapping entire LangGraph execution
   - Session and user tracking
   - Multi-agent coordination
   - Execution time metrics
   - Tool call aggregation

4. **Prompt Templates**
   - Template versioning for A/B testing
   - Variable tracking for Playground integration
   - Template evolution over time

## 📊 Key Features in Arize

### 1. Trace Visualization

**Waterfall View**: See the complete execution timeline:
```
surf_trip_planner_workflow (parent)
├── research_agent (parallel)
│   ├── LLM call (tool planning)
│   ├── surf_spot_info (tool)
│   ├── surf_forecast_brief (tool)
│   └── LLM call (synthesis)
├── budget_agent (parallel)
│   ├── LLM call (tool planning)
│   ├── surf_trip_budget (tool)
│   └── LLM call (synthesis)
├── local_agent (parallel)
│   ├── rag_retrieval (vector search)
│   ├── LLM call (tool planning)
│   ├── local_surf_scene (tool)
│   └── LLM call (synthesis)
└── itinerary_agent (sequential)
    └── LLM call (final synthesis)
```

### 2. Session Tracking

Group traces by session to see user journeys:
- Filter by `session.id`
- Track conversation flow
- Analyze multi-turn interactions

### 3. Prompt Playground

Experiment with prompts directly in Arize:
- View all prompt templates and versions
- Modify variables and test
- Compare outputs side-by-side
- Export improved prompts back to code

### 4. Cost Tracking

Monitor LLM costs:
- Token usage per request
- Cost breakdown by agent
- Identify expensive operations

### 5. Performance Analysis

Optimize latency:
- Slowest agents
- Tool execution times
- LLM call duration
- RAG retrieval speed

### 6. Error Analysis

Debug failures:
- Exception traces
- Error patterns
- Failure rates by destination/agent

### 7. Custom Metrics

Filter and analyze by custom attributes:
- `agent.type`: research_agent, budget_agent, etc.
- `agent.destination`: Pipeline, J-Bay, etc.
- `agent.skill_level`: intermediate, advanced, expert
- `agent.rag_enabled`: true/false
- `workflow.tool_calls_total`: number of tools used

## 🎛️ Advanced Configuration

### Hide Sensitive Data

Modify trace settings in `backend/main.py`:

```python
trace_config = TraceConfig(
    hide_inputs=False,           # Hide all inputs
    hide_outputs=False,          # Hide all outputs
    hide_input_messages=False,   # Hide LLM input messages
    hide_output_messages=False,  # Hide LLM output messages
    hide_input_images=True,      # Hide image data
    hide_embedding_vectors=False # Hide embedding vectors
)
```

### Custom Metadata

Add more metadata in agent functions:

```python
with using_attributes(
    tags=["custom_tag"],
    metadata={
        "custom_field": "custom_value",
        "experiment_id": "exp-123"
    }
):
    # Your agent code
```

### Alternative: Manual Tracer Configuration

If you need more control, replace `register()` with manual setup:

```python
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

# Set resource attributes
resource = Resource(attributes={
    "model_id": "surf-trip-multi-agent",
    "model_version": "v1.0",
    "service.name": "ai-surf-trip-planner"
})

# Configure tracer provider
tracer_provider = trace_sdk.TracerProvider(resource=resource)

# Configure OTLP exporter
exporter = OTLPSpanExporter(
    endpoint="https://otlp.arize.com/v1",
    headers={
        "space_id": ARIZE_SPACE_ID,
        "api_key": ARIZE_API_KEY
    }
)

# Add batch processor (more efficient than simple processor)
tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

# Set as global provider
trace_api.set_tracer_provider(tracer_provider)
```

## 🐛 Troubleshooting

### No Traces Appearing

1. **Check Credentials**:
   ```bash
   echo $ARIZE_SPACE_ID
   echo $ARIZE_API_KEY
   ```

2. **Verify Initialization**:
   Look for startup message:
   ```
   ✅ Arize AX tracing initialized successfully
   ```

3. **Check Network**:
   Ensure you can reach `https://otlp.arize.com/v1`

4. **Force Flush** (for testing):
   Add this after a request in development:
   ```python
   from opentelemetry import trace
   trace.get_tracer_provider().force_flush()
   ```

### Traces Not Grouped by Session

Ensure you're passing `session_id` and `user_id` in requests:

```json
{
  "destination": "Bali",
  "duration": "10 days",
  "session_id": "unique-session-id",
  "user_id": "user-email-or-id"
}
```

### Missing RAG Traces

1. Enable RAG: `ENABLE_RAG=1`
2. Ensure OpenAI API key is set (for embeddings)
3. Check that `local_agent` is being called

### High Trace Volume

Implement sampling in production:

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

# Sample 10% of traces
sampler = TraceIdRatioBased(0.1)
tracer_provider = trace_sdk.TracerProvider(sampler=sampler, resource=resource)
```

## 📚 Learning Resources

### Arize AX Documentation
- **Tracing Guide**: [https://arize.com/docs/ax/tracing-assistant](https://arize.com/docs/ax/tracing-assistant)
- **LangGraph Integration**: [https://arize.com/docs/ax/tracing/langgraph](https://arize.com/docs/ax/tracing/langgraph)
- **OpenInference Spec**: [https://arize.com/docs/ax/tracing/openinference](https://arize.com/docs/ax/tracing/openinference)

### MCP Tracing Assistant

Install the Arize MCP server in Cursor for AI-assisted instrumentation:

```bash
# Add to Cursor MCP config
claude mcp add arize-tracing-assistant uvx arize-tracing-assistant@latest
```

Then ask Cursor:
- "How do I add span attributes?"
- "Can you instrument this RAG pipeline?"
- "Where can I find my Arize keys?"

## 🎯 Next Steps

1. **Run Your First Trace**: Follow Quick Start above
2. **Explore in Arize**: Click through traces, view agents, analyze tools
3. **Enable RAG**: Set `ENABLE_RAG=1` to see retrieval spans
4. **Add Web Search**: Configure Tavily/SerpAPI for real-time data
5. **Run Evaluations**: Use Arize's eval framework to assess quality
6. **Create Dashboards**: Build custom views for your use case
7. **Set Up Alerts**: Monitor errors, latency, costs

## 💡 Best Practices

1. **Always Use Session IDs**: Critical for debugging user issues
2. **Add Descriptive Metadata**: Makes filtering and analysis easier
3. **Version Your Prompts**: Use template versioning for A/B tests
4. **Monitor Costs**: Set up alerts for token usage spikes
5. **Test Error Paths**: Verify exceptions are properly traced
6. **Use Sampling in Prod**: Reduce costs while maintaining visibility
7. **Create Dashboards**: Track key metrics (latency, costs, errors)

## 🆘 Support

- **Arize Support**: [support@arize.com](mailto:support@arize.com)
- **Slack Community**: [Join Arize Slack](https://arize.com/slack)
- **Documentation**: [https://arize.com/docs/ax](https://arize.com/docs/ax)
- **GitHub Issues**: Report instrumentation bugs in this repo

---

**Happy Tracing! 🏄‍♂️📊**

