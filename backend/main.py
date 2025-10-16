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
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import httpx


class PRDRequest(BaseModel):
    product_name: str
    version: Optional[str] = "1.0"
    product_manager: Optional[str] = None
    target_date: Optional[str] = None
    executive_summary: str
    problem_statement: str
    solution: Optional[str] = None
    primary_users: str
    secondary_users: Optional[str] = None
    features: List[Dict[str, Any]] = []
    platforms: List[str] = []
    tech_stack: Optional[str] = None
    performance_requirements: Optional[str] = None
    primary_kpis: Optional[str] = None
    secondary_kpis: Optional[str] = None
    timeline: Optional[str] = None
    milestones: Optional[str] = None
    risks: Optional[str] = None
    constraints: Optional[str] = None
    # Optional fields for enhanced session tracking and observability
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    turn_index: Optional[int] = None


class PRDResponse(BaseModel):
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
                content = "Test PRD"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE", "0").lower() not in {"0", "false", "no"}:
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=2000)
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


def _ensure_utf8(text: str) -> str:
    """Ensure text is properly UTF-8 encoded and handle any encoding issues."""
    if not text:
        return ""
    try:
        # If text is already a string, ensure it's properly encoded
        if isinstance(text, str):
            return text.encode('utf-8', errors='replace').decode('utf-8')
        return str(text)
    except Exception:
        # Fallback: replace problematic characters
        return text.encode('ascii', errors='replace').decode('ascii')


def _search_api(query: str) -> Optional[str]:
    """Search the web using Tavily or SerpAPI if configured, return None otherwise."""
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
    """Use the LLM to generate a response when search APIs aren't available."""
    prompt = "Respond with 200 characters or less.\n" + instruction.strip()
    if context:
        prompt += "\nContext:\n" + context.strip()
    response = llm.invoke([
        SystemMessage(content="You are a concise product management assistant."),
        HumanMessage(content=prompt),
    ])
    return _compact(response.content)


def _with_prefix(prefix: str, summary: str) -> str:
    """Add a prefix to a summary for clarity."""
    text = f"{prefix}: {summary}" if prefix else summary
    return _compact(text)


# Tools with real API calls + LLM fallback (graceful degradation pattern)
@tool
def market_research(product_name: str, problem_statement: str) -> str:
    """Research market trends and competitive landscape for the product."""
    query = f"{product_name} market research competitive analysis trends {problem_statement}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{product_name} market insights", summary)
    
    # LLM fallback when no search API is configured
    instruction = f"Provide market research insights for {product_name} addressing {problem_statement}."
    return _llm_fallback(instruction)


@tool
def user_research(primary_users: str, problem_statement: str) -> str:
    """Research user needs and pain points for the target audience."""
    query = f"{primary_users} user needs pain points {problem_statement}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{primary_users} user insights", summary)
    
    instruction = f"Provide user research insights for {primary_users} regarding {problem_statement}."
    return _llm_fallback(instruction)


@tool
def technical_research(tech_stack: str, platforms: str) -> str:
    """Research technical considerations and best practices."""
    query = f"{tech_stack} {platforms} technical requirements best practices"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{tech_stack} technical insights", summary)
    
    instruction = f"Provide technical insights for {tech_stack} on {platforms} platforms."
    return _llm_fallback(instruction)


@tool
def competitive_analysis(product_name: str, solution: str) -> str:
    """Analyze competitive landscape and positioning."""
    query = f"{product_name} competitors alternatives {solution}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{product_name} competitive analysis", summary)
    
    instruction = f"Provide competitive analysis for {product_name} with solution: {solution}."
    return _llm_fallback(instruction)


@tool
def risk_assessment(product_name: str, constraints: str) -> str:
    """Assess potential risks and mitigation strategies."""
    query = f"{product_name} risks challenges {constraints}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{product_name} risk assessment", summary)
    
    instruction = f"Provide risk assessment for {product_name} considering constraints: {constraints}."
    return _llm_fallback(instruction)


class PRDState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    prd_request: Dict[str, Any]
    market_research: Optional[str]
    user_research: Optional[str]
    technical_research: Optional[str]
    competitive_analysis: Optional[str]
    risk_assessment: Optional[str]
    final: Optional[str]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


def market_research_agent(state: PRDState) -> PRDState:
    req = state["prd_request"]
    product_name = req["product_name"]
    problem_statement = req["problem_statement"]
    
    prompt_t = (
        "You are a market research analyst.\n"
        "Research market trends and competitive landscape for {product_name}.\n"
        "Focus on the problem: {problem_statement}.\n"
        "Use tools to gather market insights, then synthesize findings."
    )
    vars_ = {"product_name": product_name, "problem_statement": problem_statement}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [market_research, competitive_analysis]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["market_research", "competitive_analysis"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "market_research")
                current_span.set_attribute("metadata.agent_node", "market_research_agent")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    # Collect tool calls and execute them
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "market_research", "tool": c["name"], "args": c.get("args", {})})
        
        # Execute tools manually
        tool_results = []
        for tool_call in res.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id", "")
            
            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        result = tool.invoke(tool_args)
                        tool_results.append(ToolMessage(content=f"Tool {tool_name} result: {result}", tool_call_id=tool_call_id))
                    except Exception as e:
                        tool_results.append(ToolMessage(content=f"Tool {tool_name} error: {str(e)}", tool_call_id=tool_call_id))
                    break
        
        # Add tool results to conversation and ask LLM to synthesize
        messages.append(res)
        messages.extend(tool_results)
        
        synthesis_prompt = "Based on the market research above, provide a comprehensive market analysis for the PRD."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call with its own prompt template
        synthesis_vars = {"product_name": product_name, "context": "market_research_results"}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    # Ensure proper UTF-8 encoding
    encoded_out = _ensure_utf8(out)
    return {"messages": [SystemMessage(content=encoded_out)], "market_research": encoded_out, "tool_calls": calls}


def user_research_agent(state: PRDState) -> PRDState:
    req = state["prd_request"]
    primary_users = req["primary_users"]
    problem_statement = req["problem_statement"]
    
    prompt_t = (
        "You are a user research specialist.\n"
        "Research user needs and pain points for {primary_users}.\n"
        "Focus on the problem: {problem_statement}.\n"
        "Use tools to gather user insights, then synthesize findings."
    )
    vars_ = {"primary_users": primary_users, "problem_statement": problem_statement}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [user_research]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["user_research", "user_needs"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "user_research")
                current_span.set_attribute("metadata.agent_node", "user_research_agent")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "user_research", "tool": c["name"], "args": c.get("args", {})})
        
        # Execute tools manually
        tool_results = []
        for tool_call in res.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id", "")
            
            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        result = tool.invoke(tool_args)
                        tool_results.append(ToolMessage(content=f"Tool {tool_name} result: {result}", tool_call_id=tool_call_id))
                    except Exception as e:
                        tool_results.append(ToolMessage(content=f"Tool {tool_name} error: {str(e)}", tool_call_id=tool_call_id))
                    break
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tool_results)
        
        synthesis_prompt = f"Based on the user research above, provide comprehensive user insights for {primary_users}."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call
        synthesis_vars = {"primary_users": primary_users, "context": "user_research_results"}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    # Ensure proper UTF-8 encoding
    encoded_out = _ensure_utf8(out)
    return {"messages": [SystemMessage(content=encoded_out)], "user_research": encoded_out, "tool_calls": calls}


def technical_research_agent(state: PRDState) -> PRDState:
    req = state["prd_request"]
    tech_stack = req.get("tech_stack", "")
    platforms = ", ".join(req.get("platforms", []))
    
    prompt_t = (
        "You are a technical architecture specialist.\n"
        "Research technical considerations for {tech_stack} on {platforms}.\n"
        "Use tools to gather technical insights, then synthesize findings."
    )
    vars_ = {"tech_stack": tech_stack, "platforms": platforms}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [technical_research]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["technical_research", "architecture"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "technical_research")
                current_span.set_attribute("metadata.agent_node", "technical_research_agent")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "technical_research", "tool": c["name"], "args": c.get("args", {})})
        
        # Execute tools manually
        tool_results = []
        for tool_call in res.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id", "")
            
            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        result = tool.invoke(tool_args)
                        tool_results.append(ToolMessage(content=f"Tool {tool_name} result: {result}", tool_call_id=tool_call_id))
                    except Exception as e:
                        tool_results.append(ToolMessage(content=f"Tool {tool_name} error: {str(e)}", tool_call_id=tool_call_id))
                    break
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tool_results)
        
        synthesis_prompt = f"Based on the technical research above, provide comprehensive technical insights for {tech_stack}."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call
        synthesis_vars = {"tech_stack": tech_stack, "context": "technical_research_results"}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    # Ensure proper UTF-8 encoding
    encoded_out = _ensure_utf8(out)
    return {"messages": [SystemMessage(content=encoded_out)], "technical_research": encoded_out, "tool_calls": calls}


def risk_assessment_agent(state: PRDState) -> PRDState:
    req = state["prd_request"]
    product_name = req["product_name"]
    constraints = req.get("constraints", "")
    
    prompt_t = (
        "You are a risk assessment specialist.\n"
        "Analyze potential risks for {product_name}.\n"
        "Consider constraints: {constraints}.\n"
        "Use tools to gather risk insights, then synthesize findings."
    )
    vars_ = {"product_name": product_name, "constraints": constraints}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [risk_assessment]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["risk_assessment", "risk_analysis"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "risk_assessment")
                current_span.set_attribute("metadata.agent_node", "risk_assessment_agent")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "risk_assessment", "tool": c["name"], "args": c.get("args", {})})
        
        # Execute tools manually
        tool_results = []
        for tool_call in res.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id", "")
            
            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        result = tool.invoke(tool_args)
                        tool_results.append(ToolMessage(content=f"Tool {tool_name} result: {result}", tool_call_id=tool_call_id))
                    except Exception as e:
                        tool_results.append(ToolMessage(content=f"Tool {tool_name} error: {str(e)}", tool_call_id=tool_call_id))
                    break
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tool_results)
        
        synthesis_prompt = f"Based on the risk assessment above, provide comprehensive risk analysis for {product_name}."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call
        synthesis_vars = {"product_name": product_name, "context": "risk_assessment_results"}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    # Ensure proper UTF-8 encoding
    encoded_out = _ensure_utf8(out)
    return {"messages": [SystemMessage(content=encoded_out)], "risk_assessment": encoded_out, "tool_calls": calls}


def prd_generation_agent(state: PRDState) -> PRDState:
    req = state["prd_request"]
    product_name = req["product_name"]
    version = req.get("version", "1.0")
    product_manager = req.get("product_manager", "TBD")
    executive_summary = req["executive_summary"]
    problem_statement = req["problem_statement"]
    solution = req.get("solution", "")
    primary_users = req["primary_users"]
    secondary_users = req.get("secondary_users", "")
    features = req.get("features", [])
    platforms = req.get("platforms", [])
    tech_stack = req.get("tech_stack", "")
    performance_requirements = req.get("performance_requirements", "")
    primary_kpis = req.get("primary_kpis", "")
    secondary_kpis = req.get("secondary_kpis", "")
    timeline = req.get("timeline", "")
    milestones = req.get("milestones", "")
    risks = req.get("risks", "")
    constraints = req.get("constraints", "")
    
    # Format features for the prompt
    features_text = ""
    for i, feature in enumerate(features, 1):
        if feature.get("name"):
            features_text += f"\n{i}. {feature['name']}"
            if feature.get("description"):
                features_text += f" - {feature['description']}"
            if feature.get("priority"):
                features_text += f" (Priority: {feature['priority']})"
            if feature.get("criteria"):
                features_text += f"\n   Acceptance Criteria: {feature['criteria']}"
    
    prompt_parts = [
        "Create a comprehensive, well-structured Product Requirements Document (PRD) for {product_name}.",
        "",
        "Format the response as professional markdown with:",
        "- Executive Summary with clear value proposition",
        "- Detailed Problem Statement and Solution",
        "- User personas and target audience analysis",
        "- Feature specifications with acceptance criteria",
        "- Technical requirements and architecture considerations",
        "- Success metrics and KPIs",
        "- Timeline and milestones",
        "- Risk assessment and mitigation strategies",
        "- Clear next steps and recommendations",
        "",
        "Make it comprehensive, actionable, and professional!",
        "",
        "Product Information:",
        "- Product Name: {product_name}",
        "- Version: {version}",
        "- Product Manager: {product_manager}",
        "- Executive Summary: {executive_summary}",
        "- Problem Statement: {problem_statement}",
        "- Solution: {solution}",
        "- Primary Users: {primary_users}",
        "- Secondary Users: {secondary_users}",
        "- Features: {features_text}",
        "- Platforms: {platforms}",
        "- Tech Stack: {tech_stack}",
        "- Performance Requirements: {performance_requirements}",
        "- Primary KPIs: {primary_kpis}",
        "- Secondary KPIs: {secondary_kpis}",
        "- Timeline: {timeline}",
        "- Milestones: {milestones}",
        "- Risks: {risks}",
        "- Constraints: {constraints}",
        "",
        "Research Context:",
        "- Market Research: {market_research}",
        "- User Research: {user_research}",
        "- Technical Research: {technical_research}",
        "- Risk Assessment: {risk_assessment}",
    ]
    
    prompt_t = "\n".join(prompt_parts)
    vars_ = {
        "product_name": product_name,
        "version": version,
        "product_manager": product_manager,
        "executive_summary": executive_summary,
        "problem_statement": problem_statement,
        "solution": solution,
        "primary_users": primary_users,
        "secondary_users": secondary_users,
        "features_text": features_text,
        "platforms": ", ".join(platforms) if platforms else "TBD",
        "tech_stack": tech_stack,
        "performance_requirements": performance_requirements,
        "primary_kpis": primary_kpis,
        "secondary_kpis": secondary_kpis,
        "timeline": timeline,
        "milestones": milestones,
        "risks": risks,
        "constraints": constraints,
        "market_research": (state.get("market_research") or "")[:500],
        "user_research": (state.get("user_research") or "")[:500],
        "technical_research": (state.get("technical_research") or "")[:500],
        "risk_assessment": (state.get("risk_assessment") or "")[:500],
    }
    
    # Add span attributes for better observability in Arize
    with using_attributes(tags=["prd_generation", "final_agent"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.prd_generation", "true")
                current_span.set_attribute("metadata.agent_type", "prd_generation")
                current_span.set_attribute("metadata.agent_node", "prd_generation_agent")
        
        # Prompt template wrapper for Arize Playground integration
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])
    
    # Ensure proper UTF-8 encoding of the final content
    final_content = _ensure_utf8(res.content)
    return {"messages": [SystemMessage(content=final_content)], "final": final_content}


def build_graph():
    g = StateGraph(PRDState)
    g.add_node("market_research_node", market_research_agent)
    g.add_node("user_research_node", user_research_agent)
    g.add_node("technical_research_node", technical_research_agent)
    g.add_node("risk_assessment_node", risk_assessment_agent)
    g.add_node("prd_generation_node", prd_generation_agent)

    # Run research agents in parallel
    g.add_edge(START, "market_research_node")
    g.add_edge(START, "user_research_node")
    g.add_edge(START, "technical_research_node")
    g.add_edge(START, "risk_assessment_node")
    
    # All research agents feed into the PRD generation agent
    g.add_edge("market_research_node", "prd_generation_node")
    g.add_edge("user_research_node", "prd_generation_node")
    g.add_edge("technical_research_node", "prd_generation_node")
    g.add_edge("risk_assessment_node", "prd_generation_node")
    
    g.add_edge("prd_generation_node", END)

    # Compile without checkpointer to avoid state persistence issues
    return g.compile()


app = FastAPI(title="AI PRD Generator")
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
        return FileResponse(path, media_type="text/html; charset=utf-8")
    return {"message": "frontend/index.html not found"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "ai-prd-generator"}

@app.post("/test-prd")
def test_prd():
    """Test endpoint with minimal data"""
    test_data = {
        "product_name": "Test Product",
        "executive_summary": "This is a test product",
        "problem_statement": "Testing the system",
        "primary_users": "Test users"
    }
    try:
        req = PRDRequest(**test_data)
        return {"message": "Test validation passed", "data": req.model_dump()}
    except Exception as e:
        return {"error": str(e), "message": "Test validation failed"}


# Initialize tracing once at startup, not per request
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            tp = register(space_id=space_id, api_key=api_key, project_name="ai-prd-generator")
            LangChainInstrumentor().instrument(tracer_provider=tp, include_chains=True, include_agents=True, include_tools=True)
            LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
    except Exception:
        pass

@app.post("/generate-prd", response_model=PRDResponse)
def generate_prd(req: PRDRequest):
    try:
        print(f"Received PRD request: {req.product_name}")
        graph = build_graph()
        print("Graph built successfully")
    except Exception as e:
        print(f"Error building graph: {e}")
        raise HTTPException(status_code=500, detail=f"Error building graph: {str(e)}")
    
    # Only include necessary fields in initial state
    state = {
        "messages": [],
        "prd_request": req.model_dump(),
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
    
    try:
        print("Starting graph execution...")
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
        
        print("Graph execution completed successfully")
        return PRDResponse(result=out.get("final", ""), tool_calls=out.get("tool_calls", []))
    except Exception as e:
        print(f"Error during graph execution: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during PRD generation: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

