"""
Workflow Nodes - All node functions in one place.

Simplified to 4 main nodes:
1. gather_context - Get RAG, web, existing outputs
2. plan_and_decide - Create plan, decide action
3. execute - Generate code, run it, fix errors
4. summarize - Analyze results, create summary
"""

import asyncio
import base64
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from loguru import logger

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime

from .state import State, Deps, Context, DEFAULTS


# =============================================================================
# HELPERS
# =============================================================================

def get_llm(deps: Deps, model_key: str = "llm") -> ChatOllama:
    """Get configured LLM."""
    model = deps.get(model_key, DEFAULTS.get(model_key, "qwen3:30b"))
    base_url = deps.get("base_url", DEFAULTS["base_url"])
    return ChatOllama(model=model, temperature=0, base_url=base_url)


def get_query(state: State) -> str:
    """Extract latest user query."""
    messages = state.get("messages", [])
    return messages[-1].content if messages else ""


# =============================================================================
# NODE 1: GATHER CONTEXT
# =============================================================================

async def gather_context(state: State, runtime: Runtime[Deps]) -> dict:
    """
    Gather all context in parallel: RAG, web search, existing outputs.
    
    Returns combined context for use in planning and code generation.
    """
    logger.info("📦 Gathering context...")
    deps = runtime.context
    query = get_query(state)
    workflow_id = state.get("workflow_id", "default")
    
    context = Context(rag="", web="", outputs="", plots=[], combined="")
    parts = []
    
    # 1. RAG Context
    rag = deps.get("rag")
    if rag and rag.enabled:
        try:
            rag_summary = rag.get_context_summary(
                query=query,
                workflow_id=workflow_id,
                max_tokens=1500
            )
            if rag_summary:
                context["rag"] = rag_summary
                parts.append(f"[Previous Analysis]\n{rag_summary}")
                logger.info(f"📚 RAG: {len(rag_summary)} chars")
        except Exception as e:
            logger.warning(f"RAG failed: {e}")
    
    # 2. Web Search (if enabled)
    if state.get("web_search_enabled"):
        try:
            from .web_search import search_analytics_methods
            result = await search_analytics_methods(query, max_results=2)
            if result.results:
                web_text = "\n".join(
                    f"- {r.title}: {r.content[:300]}" 
                    for r in result.results[:2]
                )
                context["web"] = web_text
                parts.append(f"[Web Research]\n{web_text}")
                logger.info(f"🌐 Web: {len(result.results)} results")
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
    
    # 3. Existing Outputs
    output_manager = deps.get("output_manager")
    if output_manager and state.get("stage_name"):
        try:
            stage_dir = output_manager.get_stage_dir(state["stage_name"])
            
            # Check for plots
            plots_dir = stage_dir / "plots"
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                if plot_files:
                    context["plots"] = [str(p) for p in plot_files]
                    parts.append(f"[Existing Plots: {len(plot_files)}]")
            
            # Check console output
            console_files = list(stage_dir.glob("console_output_*.txt"))
            if console_files:
                latest = sorted(console_files)[-1]
                with open(latest) as f:
                    output_text = f.read()[:1000]
                context["outputs"] = output_text
                parts.append(f"[Previous Output]\n{output_text}")
                
        except Exception as e:
            logger.warning(f"Output check failed: {e}")
    
    # Combine all context
    context["combined"] = "\n\n".join(parts) if parts else "No prior context."
    logger.info(f"📦 Context assembled: {len(context['combined'])} chars")
    
    return {"context": context}


# =============================================================================
# NODE 2: PLAN AND DECIDE
# =============================================================================

class PlanDecision(BaseModel):
    """Combined plan and action decision."""
    plan: str = Field(description="Step-by-step analysis plan")
    action: str = Field(description="'answer' if context sufficient, 'execute' if code needed")
    reasoning: str = Field(description="Why this action")


def plan_and_decide(state: State, runtime: Runtime[Deps]) -> dict:
    """
    Create analysis plan and decide: answer from context or execute code.
    
    Single LLM call combines planning and routing decision.
    """
    logger.info("🧠 Planning...")
    deps = runtime.context
    query = get_query(state)
    context = state.get("context", {}).get("combined", "")
    has_data = bool(state.get("data_path"))
    
    llm = get_llm(deps).with_structured_output(PlanDecision)
    
    prompt = f"""Create an analysis plan and decide the best action.

User Query: {query}

Available Context:
{context[:2000]}

Data File Available: {has_data}

Rules:
- action="answer" ONLY if context fully answers the query (no new analysis needed)
- action="execute" if ANY calculation, visualization, or new analysis is needed
- action="execute" if user asks to redo, repeat, or create something new
- When uncertain, choose "execute"

Create a specific, actionable plan."""

    result = llm.invoke([
        SystemMessage(content="You are a data scientist. Plan the analysis and decide the action."),
        HumanMessage(content=prompt)
    ])
    
    logger.info(f"🎯 Action: {result.action} ({result.reasoning[:50]}...)")
    
    return {
        "plan": result.plan,
        "action": result.action
    }


# =============================================================================
# NODE 3: EXECUTE (Code Gen + Run + Fix Loop)
# =============================================================================

class GeneratedCode(BaseModel):
    """Generated Python code."""
    code: str = Field(description="Complete Python code")
    reasoning: str = Field(description="Approach explanation")


async def execute(state: State, runtime: Runtime[Deps]) -> dict:
    """
    Generate code, execute it, and fix errors (up to max_retries).
    
    Combines code generation, execution, and fixing in one node.
    """
    logger.info("🚀 Executing...")
    deps = runtime.context
    query = get_query(state)
    data_path = Path(state.get("data_path", ""))
    
    if not data_path.exists():
        return {"error": "Data file not found", "code": "", "output": None}
    
    # Setup execution directory
    output_manager = deps.get("output_manager")
    stage_dir = output_manager.get_stage_dir(state.get("stage_name", "analysis"))
    exec_dir = stage_dir / "execution"
    exec_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(data_path, exec_dir / data_path.name)
    
    # Get context for code generation
    context = state.get("context", {})
    context_text = context.get("combined", "")[:2000]
    
    code_llm = get_llm(deps, "code_llm").with_structured_output(GeneratedCode)
    
    # Generate initial code
    code_prompt = f"""Write Python code to analyze '{data_path.name}'.

Plan:
{state.get("plan", "")[:800]}

Context:
{context_text}

Requirements:
- Load: pd.read_csv('{data_path.name}')
- Use matplotlib with 'Agg' backend
- Save plots with descriptive names (plt.savefig)
- Print key findings to stdout
- Handle errors gracefully"""

    result = code_llm.invoke([
        SystemMessage(content="Write complete, executable Python code."),
        HumanMessage(content=code_prompt)
    ])
    
    code = result.code
    logger.info(f"✍️ Code generated ({len(code)} chars)")
    
    # Execute with retry loop
    executor = deps.get("executor")
    max_retries = deps.get("max_retries", DEFAULTS["max_retries"])
    output = None
    error = ""
    
    for attempt in range(max_retries + 1):
        logger.info(f"⚡ Attempt {attempt + 1}/{max_retries + 1}")
        
        output = await executor.execute_with_output_manager(
            code=code,
            stage_name=state.get("stage_name", "analysis"),
            output_manager=output_manager,
            code_filename=f"code_v{attempt + 1}.py"
        )
        
        if output.success:
            logger.info("✅ Execution succeeded")
            error = ""
            break
        
        error = output.stderr or output.error or "Unknown error"
        logger.warning(f"❌ Failed: {error[:100]}...")
        
        if attempt < max_retries:
            # Try to fix
            fix_prompt = f"""Fix this code error:

Code:
```python
{code}
```

Error:
{error}

Output so far:
{output.stdout[:500] if output.stdout else 'None'}

Return the complete fixed code."""

            try:
                fix_result = code_llm.invoke([
                    SystemMessage(content="Fix the Python code error."),
                    HumanMessage(content=fix_prompt)
                ])
                code = fix_result.code
                logger.info("🔧 Code fixed, retrying...")
            except Exception as e:
                logger.error(f"Fix failed: {e}")
                break
    
    # Store in RAG if successful
    rag = deps.get("rag")
    if output and output.success and rag and rag.enabled:
        rag.add_code_execution(
            code=code,
            stdout=output.stdout,
            stderr=output.stderr or "",
            stage_name=state.get("stage_name", "analysis"),
            workflow_id=state.get("workflow_id", "default"),
            success=True
        )
    
    return {
        "code": code,
        "output": output,
        "error": error
    }


# =============================================================================
# NODE 4: SUMMARIZE (Analyze Plots + Create Summary)
# =============================================================================

async def summarize(state: State, runtime: Runtime[Deps]) -> dict:
    """
    Analyze any plots with vision LLM, then create comprehensive summary.
    """
    logger.info("📝 Summarizing...")
    deps = runtime.context
    query = get_query(state)
    
    # Gather results
    output = state.get("output")
    stdout = output.stdout if output and hasattr(output, 'stdout') else ""
    context = state.get("context", {})
    
    # Analyze new plots with vision LLM
    plot_analyses = []
    output_manager = deps.get("output_manager")
    plot_cache = deps.get("plot_cache")
    
    if output_manager and state.get("stage_name"):
        try:
            stage_dir = output_manager.get_stage_dir(state["stage_name"])
            plots_dir = stage_dir / "plots"
            
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))[-5:]  # Last 5 plots
                
                if plot_files:
                    vision_llm = get_llm(deps, "vision_llm")
                    
                    for plot_path in plot_files:
                        # Check cache first
                        if plot_cache:
                            cached = plot_cache.get(str(plot_path))
                            if cached:
                                plot_analyses.append(cached)
                                continue
                        
                        try:
                            with open(plot_path, "rb") as f:
                                img_data = base64.b64encode(f.read()).decode()
                            
                            response = vision_llm.invoke([
                                SystemMessage(content="Describe key insights from this visualization concisely."),
                                HumanMessage(content=[
                                    {"type": "image_url", "image_url": f"data:image/png;base64,{img_data}"},
                                    {"type": "text", "text": "Analyze this plot."}
                                ])
                            ])
                            
                            analysis = {
                                "plot": plot_path.name,
                                "analysis": response.content
                            }
                            plot_analyses.append(analysis)
                            
                            if plot_cache:
                                plot_cache.set(str(plot_path), analysis)
                            
                            logger.info(f"🔍 Analyzed: {plot_path.name}")
                            
                        except Exception as e:
                            logger.warning(f"Plot analysis failed for {plot_path}: {e}")
                            
        except Exception as e:
            logger.warning(f"Plot gathering failed: {e}")
    
    # Build summary context
    summary_parts = []
    
    if stdout:
        summary_parts.append(f"Code Output:\n{stdout[:2000]}")
    
    if plot_analyses:
        
        plots_text = "\n".join(f"- {p['plot']}: {p['analysis'][:300]}" for p in plot_analyses if 'plot' in p and 'analysis' in p)
        logger.info(f"🖼️ Plot analyses: {len(plot_analyses)}")
        logger.debug(f"Plot analyses details: {plots_text}")
        summary_parts.append(f"Plot Analyses:\n{plots_text}")
    
    if context.get("rag"):
        summary_parts.append(f"Previous Context:\n{context['rag'][:500]}")
    
    if context.get("web"):
        summary_parts.append(f"Research:\n{context['web'][:300]}")
    
    results_context = "\n\n".join(summary_parts) if summary_parts else "No results available."
    
    # Generate summary
    llm = get_llm(deps)
    
    summary_prompt = f"""Provide a comprehensive answer to the user's query.

Query: {query}

Plan:
{state.get("plan", "")[:500]}

Results:
{results_context}

Be thorough, specific, and actionable. Reference specific findings from the results."""

    response = llm.invoke([
        SystemMessage(content="You are a data scientist providing analysis results."),
        HumanMessage(content=summary_prompt)
    ])
    
    # Store summary in RAG
    rag = deps.get("rag")
    if rag and rag.enabled:
        rag.add_summary(
            summary=response.content,
            stage_name=state.get("stage_name", "analysis"),
            workflow_id=state.get("workflow_id", "default")
        )
    
    logger.info("✅ Summary complete")
    
    return {"summary": response.content}


# =============================================================================
# SIMPLE ANSWER NODE (when no execution needed)
# =============================================================================

def answer_from_context(state: State, runtime: Runtime[Deps]) -> dict:
    """
    Answer directly from gathered context (no code execution).
    """
    logger.info("💬 Answering from context...")
    deps = runtime.context
    query = get_query(state)
    context = state.get("context", {}).get("combined", "")
    
    llm = get_llm(deps)
    
    prompt = f"""Answer the user's query using the available context.

Query: {query}

Context:
{context[:3000]}

Plan:
{state.get("plan", "")}

Provide a clear, helpful answer based on the context."""

    response = llm.invoke([
        SystemMessage(content="Answer based on the provided context."),
        HumanMessage(content=prompt)
    ])
    
    return {"summary": response.content}


# =============================================================================
# ROUTING
# =============================================================================

def route_action(state: State, runtime: Runtime[Deps]) -> str:
    """Route based on plan decision."""
    action = state.get("action", "execute")
    
    if action == "answer":
        logger.info("➡️ Routing to answer")
        return "answer"
    
    logger.info("➡️ Routing to execute")
    return "execute"