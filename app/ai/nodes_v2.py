"""
Workflow Nodes - Modular nodes using factory pattern.

Nodes:
- gather_context: RAG, web, outputs, documents
- plan: Create analysis plan with routing decisions
- execute: Generate and run code with file tools
- web_search: Search for methodology guidance
- rag_lookup: Query previous analyses
- plot_analysis: Analyze plots with vision model
- summarize: Create comprehensive summary
- verify: Check output completeness
- answer: Answer from context (no execution)
"""

import asyncio
import base64
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage

from .state import State, Deps, Context, DEFAULTS
from .factory import NodeFactory, NodeConfig, CodeOutput
from .tools import WorkflowFileTools, create_file_tools_for_code
from .routing import DynamicRouter, RouteDecision


# =============================================================================
# FACTORY INSTANCE
# =============================================================================

_factory: Optional[NodeFactory] = None


def get_factory(deps: Optional[Deps] = None) -> NodeFactory:
    """Get or create the node factory."""
    global _factory
    if _factory is None:
        base_url = deps.get("base_url", DEFAULTS["base_url"]) if deps else DEFAULTS["base_url"]
        _factory = NodeFactory(base_url=base_url)
    return _factory


# =============================================================================
# HELPERS
# =============================================================================

def get_llm(deps: Deps, model_key: str = "llm") -> ChatOllama:
    """Get configured LLM."""
    factory = get_factory(deps)
    return factory.get_llm(model_key, deps)


def get_query(state: State) -> str:
    """Extract latest user query."""
    messages = state.get("messages", [])
    return messages[-1].content if messages else ""


def get_emitter(deps: Deps):
    """Get progress emitter."""
    return deps.get("progress_emitter")


def emit(deps: Deps, stage: str, message: str):
    """Emit progress event."""
    emitter = get_emitter(deps)
    if emitter:
        from utils.progress_events import EventType
        emitter.emit(EventType.PROGRESS, stage, message)


# =============================================================================
# NODE: GATHER CONTEXT
# =============================================================================

async def gather_context(state: State, runtime) -> dict:
    """
    Gather all context: RAG, web search, existing outputs, documents.
    
    Stores web results in RAG for future retrieval.
    """
    deps = runtime.context
    emitter = get_emitter(deps)
    
    if emitter:
        emitter.stage_start("gather_context", "Gathering context")
    
    query = get_query(state)
    workflow_id = state.get("workflow_id", "default")
    stage_name = state.get("stage_name", "analysis")
    
    context = Context(rag="", web="", outputs="", plots=[], combined="")
    parts = []
    
    # 1. RAG Context
    rag = deps.get("rag")
    if rag and rag.enabled:
        try:
            emit(deps, "gather_context", "📚 Querying RAG...")
            
            rag_summary = rag.get_context_summary(
                query=query,
                workflow_id=workflow_id,
                max_tokens=1500
            )
            if rag_summary:
                context["rag"] = rag_summary
                parts.append(f"[Previous Analysis]\n{rag_summary}")
                
                if emitter:
                    emitter.rag_query("gather_context", len(rag_summary.split('\n')))
                
                logger.info(f"📚 RAG: {len(rag_summary)} chars")
                
            # Query documents separately
            doc_results = rag.query_documents(
                query=query,
                workflow_id=workflow_id,
                n_results=5
            )
            if doc_results:
                doc_texts = [
                    f"[{d['metadata'].get('title', 'Doc')}]: {d['document'][:500]}"
                    for d in doc_results
                ]
                parts.append(f"[Reference Documents]\n" + "\n\n".join(doc_texts))
                emit(deps, "gather_context", f"📄 Found {len(doc_results)} document chunks")
                
        except Exception as e:
            logger.warning(f"RAG failed: {e}")
    
    # 2. Web Search (if enabled)
    if state.get("web_search_enabled"):
        try:
            from .web_search import search_and_synthesize
            
            emit(deps, "gather_context", "🌐 Searching web...")
            
            llm_model = deps.get("llm", DEFAULTS["llm"])
            base_url = deps.get("base_url", DEFAULTS["base_url"])
            
            def on_progress(msg):
                emit(deps, "gather_context", f"🌐 {msg}")
            
            search_result = await search_and_synthesize(
                query=query,
                context=context.get("rag", ""),
                llm_model=llm_model,
                base_url=base_url,
                on_progress=on_progress
            )
            
            if search_result["results"]:
                context["web"] = search_result["formatted_text"]
                parts.append(f"[Web Research]\n{search_result['formatted_text']}")
                
                # Store in RAG
                if rag and rag.enabled:
                    web_results = [
                        {
                            "url": r.url,
                            "title": r.title,
                            "content": r.content,
                            "score": r.score,
                            "source": r.source,
                            "enriched": "crawl4ai" in r.source.lower(),
                            "query_used": r.query_used
                        }
                        for r in search_result["results"]
                    ]
                    rag.add_web_search_batch(
                        query=query,
                        results=web_results,
                        stage_name=stage_name,
                        workflow_id=workflow_id,
                    )
                    
                logger.info(f"🌐 Web: {len(search_result['results'])} results")
                
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
    
    # 3. Existing Outputs
    output_manager = deps.get("output_manager")
    if output_manager and stage_name:
        try:
            stage_dir = output_manager.get_stage_dir(stage_name)
            
            # List available files
            file_tools = WorkflowFileTools(output_manager.workflow_dir)
            stage_files = file_tools.get_stage_files(stage_name)
            
            if stage_files["plots"]:
                context["plots"] = [f.path for f in stage_files["plots"]]
                parts.append(f"[Existing Plots: {len(stage_files['plots'])}]")
                emit(deps, "gather_context", f"📊 Found {len(stage_files['plots'])} plots")
            
            # Get latest console output
            latest_output = file_tools.get_latest_output(stage_name)
            if latest_output:
                context["outputs"] = latest_output[:1500]
                parts.append(f"[Previous Output]\n{latest_output[:1000]}")
                
        except Exception as e:
            logger.warning(f"Output check failed: {e}")
    
    context["combined"] = "\n\n".join(parts) if parts else "No prior context."
    
    if emitter:
        emitter.stage_end("gather_context", success=True)
    
    logger.info(f"📦 Context: {len(context['combined'])} chars")
    
    return {"context": context}


# =============================================================================
# NODE: PLAN
# =============================================================================

class PlanOutput(BaseModel):
    """Planning output with routing decisions."""
    plan: str = Field(description="Step-by-step analysis plan")
    steps: List[str] = Field(description="Ordered list of steps")
    action: str = Field(description="Primary action: answer, execute, web_search, plot_analysis")
    requires_code: bool = Field(default=True)
    requires_web: bool = Field(default=False)
    requires_plots: bool = Field(default=False)
    reasoning: str = Field(description="Why this approach")


def plan(state: State, runtime) -> dict:
    """
    Create analysis plan and decide primary action.
    """
    deps = runtime.context
    emitter = get_emitter(deps)
    
    if emitter:
        emitter.stage_start("plan", "Creating analysis plan")
    
    query = get_query(state)
    context = state.get("context", {}).get("combined", "")
    has_data = bool(state.get("data_path"))
    existing_plots = state.get("context", {}).get("plots", [])
    
    llm = get_llm(deps).with_structured_output(PlanOutput)
    
    prompt = f"""Create an analysis plan for this request.

Query: {query}

Context Available:
{context[:2000]}

Data File: {has_data}
Existing Plots: {len(existing_plots)}

Decide the primary action:
- "answer" ONLY if context fully answers the query (no new analysis)
- "execute" if calculations, visualizations, or data processing needed
- "web_search" if methodology guidance needed first
- "plot_analysis" if existing plots need interpretation

Create specific, actionable steps. Be conservative - prefer execute over answer."""

    emit(deps, "plan", "🧠 Analyzing query...")
    
    try:
        result = llm.invoke([
            SystemMessage(content="You are a data science planner. Create actionable analysis plans."),
            HumanMessage(content=prompt),
        ])
        
        emit(deps, "plan", f"📋 Action: {result.action}")
        
        if emitter:
            emitter.stage_end("plan", success=True)
        
        logger.info(f"🎯 Plan: {result.action} ({len(result.steps)} steps)")
        
        return {
            "plan": result.plan,
            "action": result.action,
            "plan_steps": result.steps,
        }
        
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        if emitter:
            emitter.stage_end("plan", success=False)
        return {
            "plan": "Execute data analysis",
            "action": "execute",
            "plan_steps": ["Analyze data"],
        }


# =============================================================================
# NODE: EXECUTE (with file tools)
# =============================================================================

async def execute(state: State, runtime) -> dict:
    """
    Generate code with file tools, execute, and fix errors.
    """
    deps = runtime.context
    emitter = get_emitter(deps)
    
    if emitter:
        emitter.stage_start("execute", "Generating and executing code")
    
    query = get_query(state)
    data_path = Path(state.get("data_path", ""))
    
    if not data_path.exists():
        if emitter:
            emitter.stage_end("execute", success=False)
        return {"error": "Data file not found", "code": "", "output": None}
    
    output_manager = deps.get("output_manager")
    stage_name = state.get("stage_name", "analysis")
    stage_dir = output_manager.get_stage_dir(stage_name)
    exec_dir = stage_dir / "execution"
    exec_dir.mkdir(exist_ok=True, parents=True)
    
    # Copy data file
    shutil.copy(data_path, exec_dir / data_path.name)
    
    # Get file tools info for the code agent
    file_tools = WorkflowFileTools(output_manager.workflow_dir)
    available_files = file_tools.list_files()
    files_info = file_tools.format_file_listing(available_files, max_files=15)
    
    # Inject file tools into code
    file_tools_code = create_file_tools_for_code(output_manager.workflow_dir)
    
    context = state.get("context", {})
    context_text = context.get("combined", "")[:2000]
    plan = state.get("plan", "")
    
    # Code generation
    emit(deps, "execute", "✍️ Generating code...")
    
    code_llm = get_llm(deps, "code_llm").with_structured_output(CodeOutput)
    
    code_prompt = f"""Write Python code to analyze '{data_path.name}'.

PLAN:
{plan[:800]}

QUERY:
{query}

CONTEXT:
{context_text}

AVAILABLE FILES IN WORKFLOW:
{files_info}

REQUIREMENTS:
- Load data: pd.read_csv('{data_path.name}')
- Use matplotlib with 'Agg' backend
- Save plots with descriptive names (plt.savefig) at 300 dpi
- Print key findings to stdout
- Use the file tools (list_workflow_files, read_workflow_file) to access previous outputs if needed
- Handle errors with clear messages
- Write modular, documented code

The following file tools are auto-injected:
- list_workflow_files(pattern, stage): List files in workflow
- read_workflow_file(path): Read file content
- get_data_files(stage): Get CSV/JSON files
- get_previous_output(stage): Get console output from a stage"""

    try:
        result = code_llm.invoke([
            SystemMessage(content="Write complete, executable Python code. Use the available file tools to access previous workflow outputs when relevant."),
            HumanMessage(content=code_prompt),
        ])
        
        # Prepend file tools to code
        code = file_tools_code + "\n" + result.code
        
        if emitter:
            emitter.code_generated("execute", result.code[:300])
        
        logger.info(f"✍️ Code generated ({len(result.code)} chars)")
        
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        if emitter:
            emitter.stage_end("execute", success=False)
        return {"error": str(e), "code": "", "output": None}
    
    # Execute with retries
    executor = deps.get("executor")
    max_retries = deps.get("max_retries", DEFAULTS["max_retries"])
    output = None
    error = ""
    
    for attempt in range(max_retries + 1):
        emit(deps, "execute", f"⚡ Attempt {attempt + 1}/{max_retries + 1}")
        
        output = await executor.execute_with_output_manager(
            code=code,
            stage_name=stage_name,
            output_manager=output_manager,
            code_filename=f"code_v{attempt + 1}.py"
        )
        
        if output.success:
            emit(deps, "execute", "✅ Execution succeeded")
            logger.info("✅ Code executed successfully")
            error = ""
            break
        
        error = output.stderr or output.error or "Unknown error"
        
        if emitter:
            emitter.code_output("execute", error[:300], is_error=True)
        
        logger.warning(f"❌ Attempt {attempt + 1} failed: {error[:100]}")
        
        if attempt < max_retries:
            emit(deps, "execute", "🔧 Fixing code...")
            
            fix_prompt = f"""Fix this code error:

CODE:
```python
{result.code}
```

ERROR:
{error}

OUTPUT:
{output.stdout[:500] if output.stdout else 'None'}

Return the complete fixed code."""

            try:
                fix_result = code_llm.invoke([
                    SystemMessage(content="Fix the Python code error."),
                    HumanMessage(content=fix_prompt),
                ])
                code = file_tools_code + "\n" + fix_result.code
                logger.info("🔧 Code fixed")
            except Exception as e:
                logger.error(f"Fix failed: {e}")
                break
    
    # Store in RAG
    rag = deps.get("rag")
    if output and output.success and rag and rag.enabled:
        rag.add_code_execution(
            code=result.code,  # Original code without tools
            stdout=output.stdout,
            stderr=output.stderr or "",
            stage_name=stage_name,
            workflow_id=state.get("workflow_id", "default"),
            success=True
        )
    
    if emitter:
        emitter.stage_end("execute", success=(output and output.success))
    
    return {
        "code": result.code,
        "output": output,
        "error": error,
    }


# =============================================================================
# NODE: PLOT ANALYSIS
# =============================================================================

async def analyze_plots(state: State, runtime) -> dict:
    """
    Analyze plots with vision LLM.
    """
    deps = runtime.context
    emitter = get_emitter(deps)
    
    if emitter:
        emitter.stage_start("plot_analysis", "Analyzing visualizations")
    
    output_manager = deps.get("output_manager")
    plot_cache = deps.get("plot_cache")
    stage_name = state.get("stage_name", "analysis")
    
    analyses = []
    
    if output_manager:
        try:
            stage_dir = output_manager.get_stage_dir(stage_name)
            plots_dir = stage_dir / "plots"
            
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                
                if plot_files:
                    emit(deps, "plot_analysis", f"🔍 Analyzing {len(plot_files)} plots")
                    
                    vision_llm = get_llm(deps, "vision_llm")
                    
                    for plot_path in plot_files:
                        # Check cache
                        if plot_cache:
                            cached = plot_cache.get(str(plot_path))
                            if cached:
                                analyses.append(cached)
                                emit(deps, "plot_analysis", f"⚡ Cached: {plot_path.name}")
                                continue
                        
                        try:
                            emit(deps, "plot_analysis", f"🔍 {plot_path.name}")
                            
                            with open(plot_path, "rb") as f:
                                img_data = base64.b64encode(f.read()).decode()
                            
                            response = vision_llm.invoke([
                                SystemMessage(content="Describe key insights from this visualization. Be specific about trends, patterns, and notable values."),
                                HumanMessage(content=[
                                    {"type": "image_url", "image_url": f"data:image/png;base64,{img_data}"},
                                    {"type": "text", "text": "Analyze this plot. Focus on actionable insights."}
                                ])
                            ])
                            
                            analysis = {
                                "plot": plot_path.name,
                                "path": str(plot_path),
                                "analysis": response.content
                            }
                            analyses.append(analysis)
                            
                            # Cache and store in RAG
                            if plot_cache:
                                plot_cache.set(str(plot_path), analysis)
                            
                            rag = deps.get("rag")
                            if rag and rag.enabled:
                                rag.add_plot_analysis(
                                    plot_name=plot_path.name,
                                    plot_path=str(plot_path),
                                    analysis=response.content,
                                    stage_name=stage_name,
                                    workflow_id=state.get("workflow_id", "default")
                                )
                            
                            if emitter:
                                emitter.plot_analyzed("plot_analysis", plot_path.name, response.content[:100])
                            
                        except Exception as e:
                            logger.warning(f"Plot analysis failed for {plot_path}: {e}")
                            
        except Exception as e:
            logger.warning(f"Plot gathering failed: {e}")
    
    if emitter:
        emitter.stage_end("plot_analysis", success=True)
    
    logger.info(f"📊 Analyzed {len(analyses)} plots")
    
    return {"plot_analyses": analyses}


# =============================================================================
# NODE: SUMMARIZE
# =============================================================================

async def summarize(state: State, runtime) -> dict:
    """
    Create comprehensive summary from all results.
    """
    deps = runtime.context
    emitter = get_emitter(deps)
    
    if emitter:
        emitter.stage_start("summarize", "Creating summary")
    
    query = get_query(state)
    
    # Gather all results
    output = state.get("output")
    stdout = output.stdout if output and hasattr(output, "stdout") else ""
    
    context = state.get("context", {})
    plot_analyses = state.get("plot_analyses", [])
    
    # Build summary input
    parts = []
    
    if stdout:
        parts.append(f"Code Output:\n{stdout[:2000]}")
    
    if plot_analyses:
        plots_text = "\n".join(
            f"- {p['plot']}: {p['analysis'][:300]}"
            for p in plot_analyses if 'plot' in p and 'analysis' in p
        )
        parts.append(f"Plot Analyses:\n{plots_text}")
    
    if context.get("rag"):
        parts.append(f"Previous Context:\n{context['rag'][:500]}")
    
    if context.get("web"):
        parts.append(f"Research:\n{context['web'][:300]}")
    
    results_text = "\n\n".join(parts) if parts else "No results available."
    
    emit(deps, "summarize", "📝 Generating summary...")
    
    llm = get_llm(deps)
    
    prompt = f"""Create a comprehensive answer to the user's query.

QUERY: {query}

PLAN: {state.get("plan", "")[:500]}

RESULTS:
{results_text}

Be thorough, specific, and actionable. Reference actual findings from the results.
Include specific numbers and metrics where available."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a data scientist providing analysis results. Be specific and data-driven."),
            HumanMessage(content=prompt),
        ])
        
        summary = response.content
        
        # Store in RAG
        rag = deps.get("rag")
        if rag and rag.enabled:
            rag.add_summary(
                summary=summary,
                stage_name=state.get("stage_name", "analysis"),
                workflow_id=state.get("workflow_id", "default")
            )
        
        if emitter:
            emitter.stage_end("summarize", success=True)
        
        logger.info("✅ Summary complete")
        
        return {"summary": summary}
        
    except Exception as e:
        logger.error(f"Summary failed: {e}")
        if emitter:
            emitter.stage_end("summarize", success=False)
        return {"summary": f"Summary generation failed: {e}"}


# =============================================================================
# NODE: ANSWER (from context only)
# =============================================================================

def answer_from_context(state: State, runtime) -> dict:
    """
    Answer directly from gathered context.
    """
    deps = runtime.context
    emitter = get_emitter(deps)
    
    if emitter:
        emitter.stage_start("answer", "Answering from context")
    
    query = get_query(state)
    context = state.get("context", {}).get("combined", "")
    
    llm = get_llm(deps)
    
    emit(deps, "answer", "💬 Generating answer...")
    
    prompt = f"""Answer the user's query using the available context.

Query: {query}

Context:
{context[:3000]}

Plan: {state.get("plan", "")}

Provide a clear, helpful answer based on the context."""

    try:
        response = llm.invoke([
            SystemMessage(content="Answer based on the provided context."),
            HumanMessage(content=prompt),
        ])
        
        if emitter:
            emitter.stage_end("answer", success=True)
        
        return {"summary": response.content}
        
    except Exception as e:
        logger.error(f"Answer failed: {e}")
        if emitter:
            emitter.stage_end("answer", success=False)
        return {"summary": f"Failed to generate answer: {e}"}


# =============================================================================
# NODE: VERIFY
# =============================================================================

async def verify(state: State, runtime) -> dict:
    """
    Verify output completeness.
    """
    from .verification import verify_output
    return await verify_output(state, runtime)


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_action(state: State, runtime) -> str:
    """Route based on plan decision."""
    action = state.get("action", "execute")
    
    route_map = {
        "answer": "answer",
        "execute": "execute",
        "web_search": "web_search",
        "plot_analysis": "analyze_plots",
    }
    
    result = route_map.get(action, "execute")
    logger.info(f"➡️ Routing to: {result}")
    return result


def route_after_verify(state: State, runtime) -> str:
    """Route based on verification results."""
    verification = state.get("verification", {})
    
    is_complete = verification.get("is_complete", True)
    quality = verification.get("quality_score", 1.0)
    action = verification.get("suggested_action", "done")
    
    # Check retry count to prevent infinite loops
    retry_count = state.get("retry_count", 0)
    max_retries = 2
    
    if retry_count >= max_retries:
        logger.info("➡️ Max retries reached, completing")
        return "done"
    
    if is_complete and quality >= 0.7:
        logger.info("➡️ Verification passed")
        return "done"
    
    if action == "retry_code":
        logger.info("➡️ Retrying code execution")
        return "execute"
    elif action == "add_context":
        logger.info("➡️ Adding more context")
        return "gather_context"
    elif action == "refine":
        logger.info("➡️ Refining summary")
        return "summarize"
    
    return "done"