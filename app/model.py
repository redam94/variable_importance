import os
from typing import TypedDict, Annotated, List, Any, Union
import operator
from pathlib import Path
from loguru import logger
import base64

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END

from variable_importance.utils.code_executer import OutputCapturingExecutor, ExecutionResult
from variable_importance.utils.output_manager import OutputManager

# Import caching and RAG systems
from plot_analysis_cache import PlotAnalysisCache
from context_rag import ContextRAG


def cp_data_to_folder(input_path: str, output_path: str) -> None:
    import shutil
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    shutil.copy(input_path, output_path)

class Code(BaseModel):
    code: str = Field(..., description="Python code to be executed")
    task: str = Field(..., description="Description of the task to be performed")

class State(TypedDict):
    code: str
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], operator.add]
    input_data_path: str
    stage_name: str
    code_output: ExecutionResult
    summary: str
    graph_summaries: Annotated[List[str], operator.add]
    existing_outputs: dict
    plot_analyses: Annotated[List[dict], operator.add]
    skip_execution: bool
    workflow_id: str  # NEW: Track workflow ID for RAG

class GraphState(TypedDict):
    graph_summary: str
    graph_summaries: Annotated[List[str], operator.add]

class ExecutionDeps(TypedDict):
    executor: OutputCapturingExecutor
    output_manager: OutputManager
    plot_cache: PlotAnalysisCache  # NEW: Plot analysis cache
    rag: ContextRAG  # NEW: RAG system


def check_existing_outputs(state: State, runtime: Runtime[ExecutionDeps]) -> dict:
    """
    Check for existing outputs and artifacts from previous executions.
    Enhanced with cache checking.
    """
    output_manager = runtime.context["output_manager"]
    stage_name = state.get("stage_name", "")
    
    existing = {
        "has_plots": False,
        "plots": [],
        "has_data": False,
        "data_files": [],
        "has_console_output": False,
        "console_output": None,
        "has_previous_code": False,
        "previous_code": None
    }
    
    if not stage_name:
        return {"existing_outputs": existing, "skip_execution": False}
    
    try:
        stage_dir = output_manager.get_stage_dir(stage_name)
        
        # Check for plots
        plots_dir = stage_dir / "plots"
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg"))
            if plot_files:
                existing["has_plots"] = True
                existing["plots"] = [str(p) for p in plot_files]
                logger.info(f"📊 Found {len(plot_files)} existing plots")
        
        # Check for data files
        data_dir = stage_dir / "data"
        if data_dir.exists():
            data_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet"))
            if data_files:
                existing["has_data"] = True
                existing["data_files"] = [str(d) for d in data_files]
                logger.info(f"📁 Found {len(data_files)} existing data files")
        
        # Check for console output
        console_output_file = stage_dir / "console_output.txt"
        if console_output_file.exists():
            existing["has_console_output"] = True
            with open(console_output_file, 'r') as f:
                existing["console_output"] = f.read()
            logger.info(f"📄 Found existing console output")
        
        # Check for previous code
        for code_file in stage_dir.glob("*.py"):
            if not code_file.name.startswith("_"):
                existing["has_previous_code"] = True
                with open(code_file, 'r') as f:
                    existing["previous_code"] = f.read()
                logger.info(f"💾 Found existing code: {code_file.name}")
                break
                
    except Exception as e:
        logger.warning(f"⚠️ Error checking existing outputs: {e}")
    
    return {"existing_outputs": existing, "skip_execution": False}


def analyze_plots_with_vision_llm(state: State, runtime: Runtime[ExecutionDeps]) -> dict:
    """
    Analyze plots using vision LLM with caching and RAG integration.
    """
    existing_outputs = state.get("existing_outputs", {})
    plots = existing_outputs.get("plots", [])
    
    if not plots:
        logger.info("No plots to analyze")
        return {"plot_analyses": []}
    
    logger.info(f"🔍 Analyzing {len(plots)} plots with vision LLM (with caching)...")
    
    # Get cache and RAG from context
    plot_cache = runtime.context.get("plot_cache")
    rag = runtime.context.get("rag")
    workflow_id = state.get("workflow_id", "default")
    stage_name = state.get("stage_name", "")
    
    # Initialize vision LLM
    vision_llm = ChatOllama(
        model="qwen3-vl:30b",
        temperature=0,
        base_url="http://100.91.155.118:11434"
    )
    
    analyses = []
    
    for plot_path in plots[:5]:  # Limit to 5 plots
        try:
            plot_name = Path(plot_path).name
            
            # Check cache first
            if plot_cache:
                cached_analysis = plot_cache.get(plot_path)
                if cached_analysis:
                    analyses.append(cached_analysis)
                    logger.info(f"⚡ Used cached analysis: {plot_name}")
                    continue
            
            # Not in cache - analyze with vision LLM
            with open(plot_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = vision_llm.invoke([
                SystemMessage(
                    content="""You are an expert data scientist analyzing visualization plots. 
                    Describe what you see in the plot including:
                    - Type of plot (scatter, line, bar, histogram, etc.)
                    - Key patterns, trends, or insights
                    - Axis labels and scales
                    - Any notable features or anomalies
                    Be concise but thorough."""
                ),
                HumanMessage(
                    content=[
                        {"type": "image_url", "image_url": f"data:image/png;base64,{image_data}"},
                        {"type": "text", "text": "Analyze this plot and describe what insights it provides."}
                    ]
                )
            ])
            
            analysis = {
                "plot_path": plot_path,
                "plot_name": plot_name,
                "analysis": response.content
            }
            analyses.append(analysis)
            
            # Cache the analysis
            if plot_cache:
                plot_cache.set(plot_path, analysis)
            
            # Add to RAG
            if rag and rag.enabled:
                rag.add_plot_analysis(
                    plot_name=plot_name,
                    plot_path=plot_path,
                    analysis=response.content,
                    stage_name=stage_name,
                    workflow_id=workflow_id
                )
            
            logger.info(f"✅ Analyzed and cached: {plot_name}")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing plot {plot_path}: {e}")
            analyses.append({
                "plot_path": plot_path,
                "plot_name": Path(plot_path).name,
                "analysis": f"Error analyzing plot: {str(e)}"
            })
    
    return {"plot_analyses": analyses}


def determine_if_code_needed(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    """
    Enhanced routing with RAG-based context retrieval.
    Uses RAG to get only relevant context, reducing token usage.
    """
    if state.get("input_data_path") is None:
        return "no_code"
    
    user_query = state["messages"]
    existing_outputs = state.get("existing_outputs", {})
    plot_analyses = state.get("plot_analyses", [])
    
    # NEW: Use RAG to get relevant context
    rag = runtime.context.get("rag")
    workflow_id = state.get("workflow_id", "default")
    stage_name = state.get("stage_name", "")
    
    # Get latest user message
    latest_message = user_query[-1].content if user_query else ""
    
    # Query RAG for relevant context (token-efficient)
    rag_context = ""
    if rag and rag.enabled:
        rag_context = rag.get_context_summary(
            query=latest_message,
            workflow_id=workflow_id,
            stage_name=stage_name,
            max_tokens=1000  # Limit context size
        )
    
    class DecisionReasoning(BaseModel):
        can_answer_from_existing: bool = Field(...)
        needs_new_code: bool = Field(...)
        reasoning: str = Field(...)
    
    llm = ChatOllama(
        model="gpt-oss:20b",
        temperature=0,
        base_url="http://100.91.155.118:11434"
    )
    structured_llm = llm.with_structured_output(DecisionReasoning)
    
    # Build concise context
    context_parts = []
    
    if existing_outputs.get("has_plots"):
        context_parts.append(f"📊 Available: {len(existing_outputs['plots'])} plots")
    
    if plot_analyses:
        context_parts.append(f"🔍 Analyzed: {len(plot_analyses)} plots")
        # Add brief summaries (not full analyses)
        for pa in plot_analyses[:2]:
            brief = pa['analysis'][:150] + "..." if len(pa['analysis']) > 150 else pa['analysis']
            context_parts.append(f"  • {pa['plot_name']}: {brief}")
    
    if rag_context:
        context_parts.append("\n📚 Relevant context from RAG:")
        context_parts.append(rag_context[:500])  # Limit RAG context
    
    context = "\n".join(context_parts) if context_parts else "No existing outputs"
    
    result = structured_llm.invoke([
        SystemMessage(
            content=f"""Determine if new code execution is needed.
            
            Available context:
            {context}
            
            Prefer using existing outputs when they sufficiently answer the query."""
        ),
        HumanMessage(content=latest_message)
    ])
    
    logger.info(f"🤔 Decision: can_answer={result.can_answer_from_existing}, needs_code={result.needs_new_code}")
    logger.info(f"   Reasoning: {result.reasoning}")
    
    if result.can_answer_from_existing and not result.needs_new_code:
        return "use_existing"
    elif result.needs_new_code:
        return "write_code"
    else:
        return "use_existing"


def write_code_for_task(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    """
    Enhanced code generation with RAG context.
    """
    code_llm = ChatOllama(
        model="qwen3-coder:30b",
        temperature=0,
        base_url="http://100.91.155.118:11434")

    structured_code_llm = code_llm.with_structured_output(Code)

    input_data_path = Path(state["input_data_path"])
    user_query = state["messages"]
    existing_outputs = state.get("existing_outputs", {})
    
    # Get relevant context from RAG
    rag = runtime.context.get("rag")
    workflow_id = state.get("workflow_id", "default")
    latest_message = user_query[-1].content if user_query else ""
    
    rag_context = ""
    if rag and rag.enabled:
        rag_context = rag.get_context_summary(
            query=latest_message,
            workflow_id=workflow_id,
            max_tokens=800
        )
    
    context_info = []
    if existing_outputs.get("has_plots"):
        context_info.append(f"Note: {len(existing_outputs['plots'])} plots exist")
    if existing_outputs.get("previous_code"):
        context_info.append("Note: Previous code exists")
    if rag_context:
        context_info.append(f"Relevant previous work:\n{rag_context[:300]}")
    
    context_str = "\n".join(context_info) if context_info else ""
    
    logger.info(f"Input data path for code generation: {input_data_path}")
    
    system_prompt = f"""
    Write Python code to load data from `{input_data_path.name}` and solve the user's query. 
    Save plots with descriptive names.
    Use pandas for data manipulation.
    Always load the data from `{input_data_path.name}`.
    
    {context_str}
    """
    
    code = structured_code_llm.invoke([
        SystemMessage(content=system_prompt),
        *user_query[-3:],  # Only last 3 messages to save tokens
    ])
    
    return {'code': code.code}


def move_data_to_execution_folder(state: State, runtime: Runtime[ExecutionDeps]) -> None:
    stage_name = state["stage_name"]
    path = runtime.context["output_manager"].get_stage_dir(stage_name)
    execution_path = path / "execution"/ Path(state['input_data_path']).name
    cp_data_to_folder(state["input_data_path"], execution_path)
    return {}


async def execute_code(state: State, runtime: Runtime[ExecutionDeps]) -> Any:
    """
    Execute code and store results in RAG.
    """
    executor = runtime.context["executor"]
    output_manager = runtime.context["output_manager"]
    rag = runtime.context.get("rag")
    
    stage_name = state["stage_name"]
    workflow_id = state.get("workflow_id", "default")
    code = state["code"]

    result = await executor.execute_with_output_manager(
        code=code,
        stage_name=stage_name,
        output_manager=output_manager,
        code_filename="code_with_data.py"
    )
    
    # Add execution results to RAG
    if rag and rag.enabled:
        rag.add_code_execution(
            code=code,
            stdout=result.stdout,
            stderr=result.stderr,
            stage_name=stage_name,
            workflow_id=workflow_id,
            success=result.success
        )

    return {'code_output': result}


def code_fix(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    code_llm = ChatOllama(
        model="qwen3-coder:30b",
        temperature=0,
        base_url="http://100.91.155.118:11434")
    structured_code_llm = code_llm.with_structured_output(Code)
    code = state["code"]
    error = state["code_output"].error
    output = state['code_output'].stdout
    fixed_code = structured_code_llm.invoke([
        SystemMessage(
            content="Fix the following code error."),
        AIMessage(content=code),
        HumanMessage(content=f"Error: {error}\nOutput: {output}"),
    ])
    return {'code': fixed_code.code}


def check_execution_success(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    if state["code_output"].success:
        return "SUCCESS"
    else:
        return "FAILURE"


def summarize_results(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    """
    Enhanced summarization with RAG context (token-efficient).
    """
    code_output = state.get("code_output", None)
    plot_analyses = state.get("plot_analyses", [])
    existing_outputs = state.get("existing_outputs", {})
    
    # Get RAG context
    rag = runtime.context.get("rag")
    workflow_id = state.get("workflow_id", "default")
    stage_name = state.get("stage_name", "")
    user_messages = state.get("messages", [])
    latest_query = user_messages[-1].content if user_messages else ""
    
    stdout = ""
    error = ""
    if code_output is not None:
        stdout = code_output.stdout
        error = code_output.error

    llm = ChatOllama(
        model="gemma3:27b",
        temperature=0,
        base_url="http://100.91.155.118:11434/"
    )
    
    # Build concise context using RAG
    context_parts = []
    
    if stdout or error:
        context_parts.append("=== Execution Output ===")
        context_parts.append(f"Output: {stdout[:500]}")  # Truncate
        if error:
            context_parts.append(f"Errors: {error[:200]}")
    
    # Add plot analyses (brief)
    if plot_analyses:
        context_parts.append("\n=== Visual Analysis ===")
        for i, pa in enumerate(plot_analyses[:3], 1):  # Max 3
            brief = pa['analysis'][:200] + "..." if len(pa['analysis']) > 200 else pa['analysis']
            context_parts.append(f"Plot {i} ({pa['plot_name']}): {brief}")
    
    # Add RAG context (most token-efficient)
    if rag and rag.enabled:
        rag_summary = rag.get_context_summary(
            query=latest_query,
            workflow_id=workflow_id,
            stage_name=stage_name,
            max_tokens=500  # Limit to save tokens
        )
        if rag_summary:
            context_parts.append("\n=== Relevant Previous Work ===")
            context_parts.append(rag_summary)
    
    context = "\n".join(context_parts)
    
    summary = llm.invoke([
        SystemMessage(
            content="""
            Answer the user's query using the provided context.
            Focus on insights and findings, not code details.
            Reference specific visualizations when relevant.
            Be concise but comprehensive.
            """
        ),
        HumanMessage(content=latest_query),
        HumanMessage(content=f"Context:\n\n{context}")
    ])
    
    # Store summary in RAG
    if rag and rag.enabled:
        rag.add_summary(
            summary=summary.content,
            stage_name=stage_name,
            workflow_id=workflow_id
        )
    
    return {'summary': summary.content}


# ============================================================================
# BUILD ENHANCED WORKFLOW GRAPH
# ============================================================================

def route_after_check_outputs(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    """Route based on whether plots exist to analyze."""
    if state["existing_outputs"].get("has_plots"):
        return "analyze_plots"
    else:
        return "determine_code_needed"


def determine_code_decision_node(state: State, runtime: Runtime[ExecutionDeps]) -> dict:
    """
    Decision node for code execution when no plots exist.
    """
    decision = determine_if_code_needed(state, runtime)
    return {"skip_execution": decision == "use_existing"}


def route_after_decision(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    """Route based on the skip_execution decision."""
    if state.get("skip_execution", False):
        return "summarize_results"
    else:
        return "write_code"


code_graph = StateGraph(State, ExecutionDeps)

# Add all nodes
code_graph.add_node("check_existing_outputs", check_existing_outputs)
code_graph.add_node("analyze_plots", analyze_plots_with_vision_llm)
code_graph.add_node("determine_code_needed", determine_code_decision_node)
code_graph.add_node("write_code", write_code_for_task)
code_graph.add_node("move_data", move_data_to_execution_folder)
code_graph.add_node("execute_code", execute_code)
code_graph.add_node("fix_code", code_fix)
code_graph.add_node("summarize_results", summarize_results)

# Routing
code_graph.add_edge(START, "check_existing_outputs")

code_graph.add_conditional_edges(
    "check_existing_outputs",
    route_after_check_outputs,
    {"analyze_plots": "analyze_plots", "determine_code_needed": "determine_code_needed"}
)

code_graph.add_conditional_edges(
    "analyze_plots",
    determine_if_code_needed,
    {"use_existing": "summarize_results", "write_code": "write_code", "no_code": "summarize_results"}
)

code_graph.add_conditional_edges(
    "determine_code_needed",
    route_after_decision,
    {"summarize_results": "summarize_results", "write_code": "write_code"}
)

code_graph.add_edge("write_code", "move_data")
code_graph.add_edge("move_data", "execute_code")
code_graph.add_conditional_edges(
    "execute_code",
    check_execution_success,
    {"SUCCESS": "summarize_results", "FAILURE": "fix_code"}
)
code_graph.add_edge("fix_code", "execute_code")
code_graph.add_edge("summarize_results", END)

# Compile the workflow
code_workflow = code_graph.compile()

logger.info("✅ Enhanced code workflow compiled successfully (with caching and RAG)")