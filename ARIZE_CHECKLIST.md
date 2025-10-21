# ✅ Arize AX Setup Checklist

Use this checklist to set up and verify Arize AX observability for your surf trip planner.

## 📋 Setup Steps

### 1. Get Arize Credentials
- [ ] Sign up at [https://app.arize.com](https://app.arize.com)
- [ ] Navigate to Space Settings (gear icon)
- [ ] Copy your **Space ID**
- [ ] Copy your **API Key**

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```
- [ ] All packages installed without errors
- [ ] Verify: `pip list | grep arize-otel`
- [ ] Verify: `pip list | grep openinference`

### 3. Configure Environment
```bash
cp backend/.env.example backend/.env
nano backend/.env  # or your favorite editor
```

Add these **required** values:
- [ ] `ARIZE_SPACE_ID=your-space-id`
- [ ] `ARIZE_API_KEY=your-api-key`
- [ ] `OPENAI_API_KEY=your-openai-key` (or OPENROUTER_API_KEY)

Optional but recommended:
- [ ] `ENABLE_RAG=1` (to see RAG retrieval traces)
- [ ] `TAVILY_API_KEY=your-key` (for real-time web search)

### 4. Start the Server
```bash
cd ..  # back to project root
./start.sh
```

- [ ] Server starts without errors
- [ ] You see: `✅ Arize AX tracing initialized successfully`
- [ ] You see: `Project: ai-surf-trip-planner`
- [ ] Server is running on: `http://localhost:8000`

### 5. Run Test Request
```bash
python "test scripts/test_arize_tracing.py"
```

- [ ] Test script runs successfully
- [ ] API health check passes
- [ ] Test request completes
- [ ] Session ID is generated
- [ ] Instructions for viewing trace are displayed

### 6. Verify in Arize
Go to [https://app.arize.com](https://app.arize.com):

- [ ] Can log in to Arize
- [ ] Can see your Space
- [ ] Project `ai-surf-trip-planner` exists
- [ ] At least one trace is visible
- [ ] Can click on trace to see details

### 7. Explore Your First Trace

In the trace view, verify you can see:
- [ ] **Parent span**: `surf_trip_planner_workflow`
- [ ] **4 agent spans**: research_agent, budget_agent, local_agent, itinerary_agent
- [ ] **LLM calls**: OpenAI completions with prompts
- [ ] **Tool calls**: Various surf-related tools
- [ ] **Metadata**: destination, duration, skill_level, etc.
- [ ] **Session ID**: Matches the test session ID
- [ ] **Execution time**: Total workflow duration
- [ ] **Status**: All spans show OK (green)

If RAG is enabled:
- [ ] **RAG span**: `rag_retrieval` (RETRIEVER kind)
- [ ] **Documents**: Retrieved surf spots with scores
- [ ] **Metadata**: Document sources and metadata

## 🎯 Feature Verification

### Basic Tracing
- [ ] Can see trace timeline (waterfall view)
- [ ] Can expand/collapse spans
- [ ] Can view span attributes
- [ ] Can see prompt templates and variables
- [ ] Can see LLM input and output messages

### Session Tracking
- [ ] Can filter traces by `session.id`
- [ ] Can filter traces by `user.id`
- [ ] Can group related requests

### Performance Analysis
- [ ] Can see execution times for each span
- [ ] Can identify slow operations
- [ ] Can view token counts
- [ ] Can calculate costs (if OpenAI)

### Error Analysis
- [ ] Can see error status codes
- [ ] Can view exception details
- [ ] Can trace error propagation

### Prompt Playground (in Arize UI)
- [ ] Can find prompt templates
- [ ] Can view template versions
- [ ] Can modify variables
- [ ] Can test different prompts

### Custom Filtering
Try filtering by these attributes:
- [ ] `agent.type` = "research_agent"
- [ ] `agent.destination` = "Pipeline, Hawaii"
- [ ] `agent.skill_level` = "advanced"
- [ ] `agent.rag_enabled` = true
- [ ] `workflow.agents_count` = 4

## 🔧 Troubleshooting

If you encounter issues, check:

### No traces appearing
- [ ] Verify ARIZE_SPACE_ID is correct
- [ ] Verify ARIZE_API_KEY is correct
- [ ] Check server logs for initialization message
- [ ] Verify network can reach `otlp.arize.com`
- [ ] Try force flush: `trace.get_tracer_provider().force_flush()`

### Import errors
- [ ] Run `pip install -r requirements.txt --upgrade`
- [ ] Check Python version >= 3.10
- [ ] Verify virtual environment is activated

### Traces not grouped by session
- [ ] Verify `session_id` is passed in requests
- [ ] Check that `using_attributes` includes session_id
- [ ] Ensure context is properly propagated

### RAG spans missing
- [ ] Set `ENABLE_RAG=1` in .env
- [ ] Verify OPENAI_API_KEY is set (needed for embeddings)
- [ ] Check that `local_agent` is executing
- [ ] Look for "Starting RAG retrieval" event

### Server won't start
- [ ] Check all required env vars are set
- [ ] Verify LLM API key is valid
- [ ] Check port 8000 is available
- [ ] Review error messages in logs

## 📚 Next Steps

Once everything is working:

### Immediate (10 minutes)
- [ ] Make multiple requests to see different traces
- [ ] Try different destinations (Pipeline, J-Bay, Bali, etc.)
- [ ] Vary skill levels (intermediate, advanced, expert)
- [ ] Enable RAG and compare traces
- [ ] Add web search and see real-time data

### Short-term (1 hour)
- [ ] Create a custom dashboard in Arize
- [ ] Experiment with prompts in Playground
- [ ] Set up cost tracking alerts
- [ ] Configure performance monitors
- [ ] Test error scenarios and view traces

### Long-term (ongoing)
- [ ] Run evaluations on output quality
- [ ] A/B test different prompt versions
- [ ] Optimize slow agents based on traces
- [ ] Set up production monitoring
- [ ] Analyze user sessions for improvements

## 📖 Documentation Quick Links

- **This Project**:
  - Quick Setup: [ARIZE_SETUP.md](./ARIZE_SETUP.md)
  - Complete Guide: [ARIZE_OBSERVABILITY.md](./ARIZE_OBSERVABILITY.md)
  - Implementation Details: [ARIZE_IMPLEMENTATION_SUMMARY.md](./ARIZE_IMPLEMENTATION_SUMMARY.md)

- **Arize Documentation**:
  - Tracing Assistant: https://arize.com/docs/ax/tracing-assistant
  - LangGraph Guide: https://arize.com/docs/ax/tracing/langgraph
  - OpenInference Spec: https://github.com/Arize-ai/openinference

## 💡 Tips

1. **Always use session_id**: Makes debugging much easier
2. **Check server logs**: They tell you if tracing initialized
3. **Use descriptive metadata**: Makes filtering and analysis powerful
4. **Enable RAG**: The retrieval traces are really insightful
5. **Try Prompt Playground**: It's great for experimentation
6. **Set up dashboards**: Track what matters to your use case
7. **Monitor costs**: Token usage can add up quickly

## 🆘 Getting Help

If you're stuck:
1. Review the troubleshooting section above
2. Check [ARIZE_OBSERVABILITY.md](./ARIZE_OBSERVABILITY.md)
3. Read Arize docs: https://arize.com/docs/ax
4. Join Arize Slack: https://arize.com/slack
5. Email support: support@arize.com

## 🎉 Success!

When all items are checked, you have:
- ✅ Full observability into your multi-agent system
- ✅ Complete LLM call tracking
- ✅ Tool execution monitoring
- ✅ RAG pipeline visibility
- ✅ Session tracking for user journeys
- ✅ Error analysis with full context
- ✅ Performance metrics for optimization

**Now go explore your traces at [app.arize.com](https://app.arize.com)! 🏄‍♂️📊**

