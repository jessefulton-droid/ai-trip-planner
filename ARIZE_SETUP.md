# 🚀 Arize AX Setup - Quick Reference

This is a condensed setup guide. For complete documentation, see [ARIZE_OBSERVABILITY.md](./ARIZE_OBSERVABILITY.md).

## Prerequisites

1. **Arize Account**: [Sign up here](https://app.arize.com) (free tier available)
2. **Get API Credentials**:
   - Log in to Arize
   - Go to **Space Settings** (gear icon)
   - Copy your **Space ID** and **API Key**

## Installation

```bash
# 1. Navigate to backend directory
cd backend

# 2. Install dependencies (includes all Arize packages)
pip install -r requirements.txt

# 3. Create environment file
cp .env.example .env

# 4. Edit .env and add your credentials
nano .env  # or use your favorite editor
```

## Minimum Configuration

Edit `backend/.env`:

```bash
# REQUIRED: Arize credentials
ARIZE_SPACE_ID=your-space-id-here
ARIZE_API_KEY=your-api-key-here

# REQUIRED: At least one LLM provider
OPENAI_API_KEY=your-openai-key-here
```

## Run and Test

```bash
# Start the server (from project root)
./start.sh

# You should see:
# ✅ Arize AX tracing initialized successfully
#    Project: ai-surf-trip-planner
#    View traces at: https://app.arize.com

# Test the API
curl -X POST http://localhost:8000/plan-trip \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Pipeline, Hawaii",
    "duration": "7 days",
    "skill_level": "advanced",
    "session_id": "test-123"
  }'
```

## View Traces

1. Go to [https://app.arize.com](https://app.arize.com)
2. Select your Space
3. Find project: **ai-surf-trip-planner**
4. Click on any trace to see:
   - Multi-agent workflow execution
   - LLM calls with prompts and responses
   - Tool calls and results
   - RAG retrieval (if enabled)
   - Performance metrics

## What's Traced

✅ **Multi-Agent Workflow**: All 4 agents (research, budget, local, itinerary)  
✅ **LLM Calls**: OpenAI/OpenRouter completions with tokens and latency  
✅ **Tool Calls**: All tool executions and results  
✅ **RAG Operations**: Vector search and document retrieval (if enabled)  
✅ **Session Tracking**: User journeys across requests  
✅ **Error Handling**: Exception traces with full context  
✅ **Prompt Templates**: Versioned prompts for Playground integration  

## Enable Additional Features

### Enable RAG (Recommended)

```bash
# In backend/.env
ENABLE_RAG=1
```

This enables vector search over 25 curated surf spots in the local agent.

### Enable Web Search

```bash
# Get a Tavily API key: https://tavily.com (1000 free searches/month)
TAVILY_API_KEY=your-tavily-key-here
```

This provides real-time surf data instead of LLM-generated content.

## Troubleshooting

### No traces appearing?

1. **Check credentials**:
   ```bash
   cat backend/.env | grep ARIZE
   ```

2. **Verify startup message**:
   Look for "✅ Arize AX tracing initialized successfully"

3. **Force flush** (for testing):
   Add to `backend/main.py` after a request:
   ```python
   trace.get_tracer_provider().force_flush()
   ```

### Import errors?

```bash
cd backend
pip install -r requirements.txt --upgrade
```

### Still stuck?

- See full docs: [ARIZE_OBSERVABILITY.md](./ARIZE_OBSERVABILITY.md)
- Contact Arize support: [support@arize.com](mailto:support@arize.com)
- Join Slack: [arize.com/slack](https://arize.com/slack)

## Key Arize AX Features to Explore

1. **Trace View**: See the waterfall timeline of your multi-agent system
2. **Sessions**: Filter by `session_id` to track user journeys
3. **Prompt Playground**: Test and improve prompts directly in Arize
4. **Cost Tracking**: Monitor token usage and LLM costs
5. **Performance**: Identify slow agents and optimize
6. **Evaluations**: Run automated quality checks on outputs
7. **Dashboards**: Create custom views for key metrics

## Next Steps

✅ Complete this setup  
⬜ Make a test request and view trace in Arize  
⬜ Enable RAG and see retrieval spans  
⬜ Add web search for real-time data  
⬜ Explore prompt playground  
⬜ Set up custom dashboards  
⬜ Run evaluations on your outputs  

---

**For complete documentation**, see [ARIZE_OBSERVABILITY.md](./ARIZE_OBSERVABILITY.md)

