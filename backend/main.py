from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Minimal observability via Arize/OpenInference (optional)
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template, using_metadata, using_attributes
    from opentelemetry import trace
    _TRACING = True
except Exception:
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_metadata(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_attributes(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
import httpx


class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    surf_preferences: Optional[str] = None  # reef breaks, point breaks, barrel hunting, mellow cruising
    skill_level: Optional[str] = None  # intermediate, advanced, expert
    travel_style: Optional[str] = None
    # Optional fields for enhanced session tracking and observability
    user_input: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    turn_index: Optional[int] = None


class TripResponse(BaseModel):
    result: str
    tool_calls: List[Dict[str, Any]] = []


def _init_llm():
    # Simple, test-friendly LLM init
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Test itinerary"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE", "0").lower() in {"1", "true", "yes"}:
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1500)
    elif os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter via OpenAI-compatible client
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.7,
        )
    else:
        # Require a key unless running tests
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# Feature flag for optional RAG demo (opt-in for learning)
ENABLE_RAG = os.getenv("ENABLE_RAG", "0").lower() not in {"0", "false", "no"}


# RAG helper: Load curated surf spots as LangChain documents
def _load_local_documents(path: Path) -> List[Document]:
    """Load surf spots JSON and convert to LangChain Documents."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return []

    docs: List[Document] = []
    for row in raw:
        description = row.get("description")
        destination = row.get("destination")
        break_name = row.get("break_name", "")
        if not description or not destination:
            continue
        skill_levels = row.get("skill_levels", []) or []
        wave_type = row.get("wave_type", "")
        metadata = {
            "destination": destination,
            "break_name": break_name,
            "skill_levels": skill_levels,
            "wave_type": wave_type,
            "best_season": row.get("best_season", ""),
            "source": row.get("source"),
        }
        # Prefix destination + wave type in content so embeddings capture location and surf context
        skill_text = ", ".join(skill_levels) if skill_levels else "all levels"
        content = f"Destination: {destination}\nBreak: {break_name}\nWave Type: {wave_type}\nSkill Levels: {skill_text}\nDescription: {description}"
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


class LocalGuideRetriever:
    """Retrieves curated surf spots using vector similarity search.
    
    This class demonstrates production RAG patterns for students:
    - Vector embeddings for semantic search
    - Fallback to keyword matching when embeddings unavailable
    - Graceful degradation with feature flags
    """
    
    def __init__(self, data_path: Path):
        """Initialize retriever with surf spots data.
        
        Args:
            data_path: Path to surf_spots.json file
        """
        self._docs = _load_local_documents(data_path)
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._vectorstore: Optional[InMemoryVectorStore] = None
        
        # Only create embeddings when RAG is enabled and we have an API key
        if ENABLE_RAG and self._docs and not os.getenv("TEST_MODE"):
            try:
                model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
                self._embeddings = OpenAIEmbeddings(model=model)
                store = InMemoryVectorStore(embedding=self._embeddings)
                store.add_documents(self._docs)
                self._vectorstore = store
            except Exception:
                # Gracefully degrade to keyword search if embeddings fail
                self._embeddings = None
                self._vectorstore = None

    @property
    def is_empty(self) -> bool:
        """Check if any documents were loaded."""
        return not self._docs

    def retrieve(self, destination: str, surf_preferences: Optional[str], *, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant surf spots for a destination.
        
        Args:
            destination: Surf destination name
            surf_preferences: Comma-separated preferences (e.g., "reef breaks, barrels")
            k: Number of results to return
            
        Returns:
            List of dicts with 'content', 'metadata', and 'score' keys
        """
        if not ENABLE_RAG or self.is_empty:
            return []

        # Use vector search if available, otherwise fall back to keywords
        if not self._vectorstore:
            return self._keyword_fallback(destination, surf_preferences, k=k)

        query = destination
        if surf_preferences:
            query = f"{destination} surfing {surf_preferences}"
        
        try:
            # LangChain retriever ensures embeddings + searches are traced
            retriever = self._vectorstore.as_retriever(search_kwargs={"k": max(k, 4)})
            docs = retriever.invoke(query)
        except Exception:
            return self._keyword_fallback(destination, surf_preferences, k=k)

        # Format results with metadata and scores
        top_docs = docs[:k]
        results = []
        for doc in top_docs:
            score_val: float = 0.0
            if isinstance(doc.metadata, dict):
                maybe_score = doc.metadata.get("score")
                if isinstance(maybe_score, (int, float)):
                    score_val = float(maybe_score)
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score_val,
            })

        if not results:
            return self._keyword_fallback(destination, surf_preferences, k=k)
        return results

    def _keyword_fallback(self, destination: str, surf_preferences: Optional[str], *, k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval when embeddings unavailable.
        
        This demonstrates graceful degradation for students learning about
        fallback strategies in production systems.
        """
        dest_lower = destination.lower()
        pref_terms = [part.strip().lower() for part in (surf_preferences or "").split(",") if part.strip()]

        def _score(doc: Document) -> int:
            score = 0
            dest_match = doc.metadata.get("destination", "").lower()
            # Match destination name
            if dest_lower and dest_lower.split(",")[0] in dest_match:
                score += 2
            # Match surf preferences
            for term in pref_terms:
                wave_type = doc.metadata.get("wave_type", "").lower()
                if term and (term in wave_type or term in " ".join(doc.metadata.get("skill_levels") or []).lower()):
                    score += 1
                if term and term in doc.page_content.lower():
                    score += 1
            return score

        scored_docs = [(_score(doc), doc) for doc in self._docs]
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        top_docs = scored_docs[:k]
        
        results = []
        for score, doc in top_docs:
            if score > 0:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                })
        return results


# Initialize retriever at module level (loads data once at startup)
_DATA_DIR = Path(__file__).parent / "data"
GUIDE_RETRIEVER = LocalGuideRetriever(_DATA_DIR / "surf_spots.json")


# Search API configuration and helpers
SEARCH_TIMEOUT = 10.0  # seconds


def _compact(text: str, limit: int = 200) -> str:
    """Compact text to a maximum length, truncating at word boundaries."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    truncated = cleaned[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(",.;- ")


def _search_api(query: str) -> Optional[str]:
    """Search the web using Tavily or SerpAPI if configured, return None otherwise.
    
    This demonstrates graceful degradation: tools work with or without API keys.
    Students can enable real search by adding TAVILY_API_KEY or SERPAPI_API_KEY.
    """
    query = query.strip()
    if not query:
        return None

    # Try Tavily first (recommended for AI apps)
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": query,
                        "max_results": 3,
                        "search_depth": "basic",
                        "include_answer": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer") or ""
                snippets = [
                    item.get("content") or item.get("snippet") or ""
                    for item in data.get("results", [])
                ]
                combined = " ".join([answer] + snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass  # Fail gracefully, try next option

    # Try SerpAPI as fallback
    serp_key = os.getenv("SERPAPI_API_KEY")
    if serp_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.get(
                    "https://serpapi.com/search",
                    params={
                        "api_key": serp_key,
                        "engine": "google",
                        "num": 5,
                        "q": query,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                organic = data.get("organic_results", [])
                snippets = [item.get("snippet", "") for item in organic]
                combined = " ".join(snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass  # Fail gracefully

    return None  # No search APIs configured


def _llm_fallback(instruction: str, context: Optional[str] = None) -> str:
    """Use the LLM to generate a response when search APIs aren't available.
    
    This ensures tools always return useful information, even without API keys.
    """
    prompt = "Respond with 200 characters or less.\n" + instruction.strip()
    if context:
        prompt += "\nContext:\n" + context.strip()
    response = llm.invoke([
        SystemMessage(content="You are a concise travel assistant."),
        HumanMessage(content=prompt),
    ])
    return _compact(response.content)


def _with_prefix(prefix: str, summary: str) -> str:
    """Add a prefix to a summary for clarity."""
    text = f"{prefix}: {summary}" if prefix else summary
    return _compact(text)


# Surf-specific tools with LLM fallback (graceful degradation pattern)
@tool
def surf_spot_info(destination: str) -> str:
    """Return essential surf spot information including break type, wave characteristics, and best swell direction."""
    query = f"{destination} surf break type wave characteristics swell direction best season surf conditions"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} surf spot info", summary)
    
    # LLM fallback when no search API is configured
    instruction = f"Summarize the break type (reef/point/beach), wave characteristics, best swell direction, ideal wave size, best season, and crowd levels for surfing at {destination}."
    return _llm_fallback(instruction)


@tool
def surf_trip_budget(destination: str, duration: str) -> str:
    """Return surf trip budget including board rentals, lessons, accommodation, and surf-specific costs."""
    query = f"{destination} surf trip budget board rental wetsuit costs surf lessons {duration}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} surf budget {duration}", summary)
    
    instruction = f"Outline accommodation near surf breaks, meals, board rentals, wetsuit costs, surf lessons/guides, wax and accessories, transport to breaks, and other surf-specific costs for a {duration} surf trip to {destination}."
    return _llm_fallback(instruction)


@tool
def local_surf_scene(destination: str, surf_preferences: Optional[str] = None) -> str:
    """Discover the local surf culture, scene, and community at the destination."""
    focus = surf_preferences or "local surf culture"
    query = f"{destination} surf culture local scene surf community {focus}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} surf scene", summary)
    
    instruction = f"Describe the local surf culture, community vibe, notable surf shops and shapers, local competitions or events, and surf scene that matches {focus} in {destination}."
    return _llm_fallback(instruction)


@tool
def surf_session_plan(destination: str, day: int) -> str:
    """Return a surf session plan for a specific day including tide timing and backup spots."""
    query = f"{destination} surf day {day} session plan tides best time"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"Day {day} surf sessions in {destination}", summary)
    
    instruction = f"Outline surf sessions for day {day} in {destination}, including dawn patrol timing, optimal tide windows, backup spots if conditions aren't ideal, and non-surf activities between sessions."
    return _llm_fallback(instruction)


# Additional simple tools per agent (to mirror original multi-tool behavior)
@tool
def surf_forecast_brief(destination: str) -> str:
    """Return surf forecast including swell size, direction, wind conditions, and tides."""
    query = f"{destination} surf forecast swell size direction wind conditions tides"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} surf forecast", summary)
    
    instruction = f"Provide a surf forecast for {destination} including expected swell size and direction, wind conditions (offshore/onshore), tide patterns, water temperature, and wetsuit recommendations."
    return _llm_fallback(instruction)


@tool
def visa_and_surf_gear_brief(destination: str) -> str:
    """Return visa guidance and surfboard/gear customs information for travel planning."""
    query = f"{destination} tourist visa requirements surfboard customs import rules"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} visa & surf gear", summary)
    
    instruction = f"Provide visa guidance and surfboard customs rules for {destination}, including any fees or restrictions on bringing surf equipment, and advice to confirm with the relevant embassy."
    return _llm_fallback(instruction)


@tool
def surf_services_pricing(destination: str, services: Optional[List[str]] = None) -> str:
    """Return pricing information for surf lessons, guides, boat trips, and rentals."""
    items = services or ["surf lessons", "board rentals", "guided sessions"]
    focus = ", ".join(items)
    query = f"{destination} surf lesson prices board rental costs {focus}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} surf services pricing", summary)
    
    instruction = f"Share typical costs for surf services such as {focus} in {destination}, including daily/weekly rental rates, lesson packages, and boat trip fees."
    return _llm_fallback(instruction)


@tool
def surf_etiquette(destination: str) -> str:
    """Return surf lineup etiquette, local rules, and respect protocols for the destination."""
    query = f"{destination} surf etiquette lineup rules localism respect protocols"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} surf etiquette", summary)
    
    instruction = f"Summarize surf lineup etiquette, local rules, how to respect locals, localism awareness, priority rules, and cultural surf customs that intermediate/advanced surfers should know before surfing {destination}."
    return _llm_fallback(instruction)


@tool
def secret_spots(destination: str) -> str:
    """Return lesser-known surf breaks and local spots for experienced surfers."""
    query = f"{destination} secret surf spots lesser known breaks local spots"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} secret surf spots", summary)
    
    instruction = f"Describe lesser-known surf breaks and local spots in {destination} that are suitable for intermediate/advanced surfers who respect local etiquette and want to avoid crowds."
    return _llm_fallback(instruction)


@tool
def travel_time(from_location: str, to_location: str, mode: str = "public") -> str:
    """Return travel time estimates between locations."""
    query = f"travel time {from_location} to {to_location} by {mode}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{from_location}→{to_location} {mode}", summary)
    
    instruction = f"Estimate travel time from {from_location} to {to_location} by {mode} transport."
    return _llm_fallback(instruction)


@tool
def surf_packing_list(destination: str, duration: str, skill_level: Optional[str] = None) -> str:
    """Return surf-specific packing recommendations including boards, wax, wetsuit, and gear."""
    level = skill_level or "intermediate"
    query = f"surf packing list {destination} {duration} wetsuit board wax {level}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} surf packing", summary)
    
    instruction = f"Suggest surf packing essentials for a {duration} surf trip to {destination} for {level} surfers, including board recommendations, wetsuit thickness, wax type, leash, repair kit, rash guards, booties, and other surf-specific gear."
    return _llm_fallback(instruction)


class TripState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    research: Optional[str]
    budget: Optional[str]
    local: Optional[str]
    final: Optional[str]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


def research_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    prompt_t = (
        "You are a surf spot research specialist.\n"
        "Gather essential surf information about {destination}.\n"
        "Use tools to get surf conditions, forecasts, break characteristics, and travel requirements for surfers, then summarize."
    )
    vars_ = {"destination": destination}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [surf_spot_info, surf_forecast_brief, visa_and_surf_gear_brief]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    tool_results = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["surf_research", "spot_intel"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "surf_research")
                current_span.set_attribute("metadata.agent_node", "research_agent")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    # Collect tool calls and execute them
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "research", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        tool_results = tr["messages"]
        
        # Add tool results to conversation and ask LLM to synthesize
        messages.append(res)
        messages.extend(tool_results)
        
        synthesis_prompt = "Based on the above information, provide a comprehensive summary for the traveler."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call with its own prompt template
        synthesis_vars = {"destination": destination, "context": "tool_results"}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "research": out, "tool_calls": calls}


def budget_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination, duration = req["destination"], req["duration"]
    budget = req.get("budget", "moderate")
    prompt_t = (
        "You are a surf trip budget specialist.\n"
        "Analyze costs for a surf trip to {destination} over {duration} with budget: {budget}.\n"
        "Use tools to get surf-specific pricing (board rentals, lessons, guides, accommodation near breaks), then provide a detailed breakdown."
    )
    vars_ = {"destination": destination, "duration": duration, "budget": budget}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [surf_trip_budget, surf_services_pricing]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["surf_budget", "cost_analysis"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "surf_budget")
                current_span.set_attribute("metadata.agent_node", "budget_agent")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "budget", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        
        synthesis_prompt = f"Create a detailed budget breakdown for {duration} in {destination} with a {budget} budget."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call
        synthesis_vars = {"duration": duration, "destination": destination, "budget": budget}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "budget": out, "tool_calls": calls}


def local_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    surf_preferences = req.get("surf_preferences", "quality waves")
    skill_level = req.get("skill_level", "intermediate")
    travel_style = req.get("travel_style", "standard")
    
    # RAG: Retrieve curated surf spots if enabled
    context_lines = []
    if ENABLE_RAG:
        retrieved = GUIDE_RETRIEVER.retrieve(destination, surf_preferences, k=3)
        if retrieved:
            context_lines.append("=== Curated Surf Spots (from database) ===")
            for idx, item in enumerate(retrieved, 1):
                content = item["content"]
                source = item["metadata"].get("source", "Unknown")
                context_lines.append(f"{idx}. {content}")
                context_lines.append(f"   Source: {source}")
            context_lines.append("=== End of Curated Surf Spots ===\n")
    
    context_text = "\n".join(context_lines) if context_lines else ""
    
    prompt_t = (
        "You are a local surf culture expert.\n"
        "Find authentic surf experiences in {destination} for {skill_level} surfers who prefer: {surf_preferences}.\n"
        "Travel style: {travel_style}. Use tools to gather local surf scene insights, etiquette, and secret spots.\n"
    )
    
    # Add retrieved context to prompt if available
    if context_text:
        prompt_t += "\nRelevant curated surf spots from our database:\n{context}\n"
    
    vars_ = {
        "destination": destination,
        "surf_preferences": surf_preferences,
        "skill_level": skill_level,
        "travel_style": travel_style,
        "context": context_text if context_text else "No curated context available.",
    }
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [local_surf_scene, surf_etiquette, secret_spots]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["surf_culture", "local_scene"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "surf_culture")
                current_span.set_attribute("metadata.agent_node", "local_agent")
                if ENABLE_RAG and context_text:
                    current_span.set_attribute("metadata.rag_enabled", "true")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "local", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        
        synthesis_prompt = f"Create a curated list of surf experiences and local scene insights for {skill_level} surfers interested in {surf_preferences} with a {travel_style} approach."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call
        synthesis_vars = {"surf_preferences": surf_preferences, "skill_level": skill_level, "travel_style": travel_style, "destination": destination}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "local": out, "tool_calls": calls}


def itinerary_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    duration = req["duration"]
    skill_level = req.get("skill_level", "intermediate")
    surf_preferences = req.get("surf_preferences", "quality waves")
    travel_style = req.get("travel_style", "standard")
    user_input = (req.get("user_input") or "").strip()
    
    prompt_parts = [
        "Create a {duration} surf trip itinerary for {destination} ({travel_style}).",
        "Target surfer: {skill_level} level, prefers {surf_preferences}.",
        "",
        "Focus on optimal surf session timing (dawn patrol, tide windows), backup spots, and surf-friendly activities.",
        "",
        "Inputs:",
        "Surf Research: {research}",
        "Surf Budget: {budget}",
        "Local Surf Scene: {local}",
    ]
    if user_input:
        prompt_parts.append("User input: {user_input}")
    
    prompt_t = "\n".join(prompt_parts)
    vars_ = {
        "duration": duration,
        "destination": destination,
        "skill_level": skill_level,
        "surf_preferences": surf_preferences,
        "travel_style": travel_style,
        "research": (state.get("research") or "")[:400],
        "budget": (state.get("budget") or "")[:400],
        "local": (state.get("local") or "")[:400],
        "user_input": user_input,
    }
    
    # Add span attributes for better observability in Arize
    # NOTE: using_attributes must be OUTER context for proper propagation
    with using_attributes(tags=["surf_itinerary", "session_planning"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.itinerary", "true")
                current_span.set_attribute("metadata.agent_type", "surf_itinerary")
                current_span.set_attribute("metadata.agent_node", "itinerary_agent")
                current_span.set_attribute("metadata.skill_level", skill_level)
                if user_input:
                    current_span.set_attribute("metadata.user_input", user_input)
        
        # Prompt template wrapper for Arize Playground integration
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])
    
    return {"messages": [SystemMessage(content=res.content)], "final": res.content}


def build_graph():
    g = StateGraph(TripState)
    g.add_node("research_node", research_agent)
    g.add_node("budget_node", budget_agent)
    g.add_node("local_node", local_agent)
    g.add_node("itinerary_node", itinerary_agent)

    # Run research, budget, and local agents in parallel
    g.add_edge(START, "research_node")
    g.add_edge(START, "budget_node")
    g.add_edge(START, "local_node")
    
    # All three agents feed into the itinerary agent
    g.add_edge("research_node", "itinerary_node")
    g.add_edge("budget_node", "itinerary_node")
    g.add_edge("local_node", "itinerary_node")
    
    g.add_edge("itinerary_node", END)

    # Compile without checkpointer to avoid state persistence issues
    return g.compile()


app = FastAPI(title="AI Surf Trip Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "frontend/index.html not found"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "ai-surf-trip-planner"}


# Initialize tracing once at startup, not per request
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            tp = register(space_id=space_id, api_key=api_key, project_name="ai-surf-trip-planner")
            LangChainInstrumentor().instrument(tracer_provider=tp, include_chains=True, include_agents=True, include_tools=True)
            LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
    except Exception:
        pass

@app.post("/plan-trip", response_model=TripResponse)
def plan_trip(req: TripRequest):
    graph = build_graph()
    
    # Only include necessary fields in initial state
    # Agent outputs (research, budget, local, final) will be added during execution
    state = {
        "messages": [],
        "trip_request": req.model_dump(),
        "tool_calls": [],
    }
    
    # Add session and user tracking attributes to the trace
    session_id = req.session_id
    user_id = req.user_id
    turn_idx = req.turn_index
    
    # Build attributes for session and user tracking
    attrs_kwargs = {}
    if session_id:
        attrs_kwargs["session_id"] = session_id
    if user_id:
        attrs_kwargs["user_id"] = user_id
    
    # Add turn_index as a custom span attribute if provided
    if turn_idx is not None and _TRACING:
        with using_attributes(**attrs_kwargs):
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("turn_index", turn_idx)
            out = graph.invoke(state)
    else:
        with using_attributes(**attrs_kwargs):
            out = graph.invoke(state)
    
    return TripResponse(result=out.get("final", ""), tool_calls=out.get("tool_calls", []))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
