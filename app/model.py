import os
from typing import TypedDict, Annotated, List, Any, Union, Optional, Dict
import operator
from pathlib import Path
from loguru import logger
import base64
from datetime import datetime

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
    workflow_id: str  # Track workflow ID for RAG
    rag_context: dict  # Store RAG query results
    can_answer_from_rag: bool  # Flag if RAG has sufficient info
    stage_metadata: dict  # Metadata for stage tracking


class GraphState(TypedDict):
    graph_summary: str
    graph_summaries: Annotated[List[str], operator.add]


class ExecutionDeps(TypedDict):
    executor: OutputCapturingExecutor
    output_manager: OutputManager
    plot_cache: PlotAnalysisCache
    rag: ContextRAG


def check_rag_for_answer(state: State, runtime: Runtime[ExecutionDeps]) -> dict:
    """
    NEW FIRST STEP: Check RAG for relevant context to answer user's query.
    This is the primary entry point to check if we can answer without new execution.
    """
    rag = runtime.context.get("rag")
    workflow_id = state.get("workflow_id", "default")
    stage_name = state.get("stage_name", "")
    
    # Get user's latest query
    user_messages = state.get("messages", [])
    latest_query = user_messages[-1].content if user_messages else ""
    
    logger.info(f"🔍 Checking RAG for existing context to answer: '{latest_query[:100]}...'")
    
    # Initialize return values
    rag_context = {
        "has_relevant_context": False,
        "contexts": [],
        "summary": "",
        "confidence": 0.0
    }
    can_answer_from_rag = False
    
    if not rag or not rag.enabled:
        logger.info("❌ RAG not available, proceeding with standard workflow")
        return {
            "rag_context": rag_context,
            "can_answer_from_rag": can_answer_from_rag
        }
    
    # Query RAG for all relevant context types
    try:
        # Get various types of relevant information
        plot_contexts = rag.query_relevant_context(
            query=latest_query,
            workflow_id=workflow_id,
            doc_types=["plot_analysis"],
            n_results=5
        )
        
        code_contexts = rag.query_relevant_context(
            query=latest_query,
            workflow_id=workflow_id,
            doc_types=["code_execution"],
            n_results=3
        )
        
        summary_contexts = rag.query_relevant_context(
            query=latest_query,
            workflow_id=workflow_id,
            doc_types=["summary"],
            n_results=3
        )
        
        all_contexts = plot_contexts + code_contexts + summary_contexts
        
        if all_contexts:
            rag_context["has_relevant_context"] = True
            rag_context["contexts"] = all_contexts
            
            # Get a comprehensive summary
            rag_summary = rag.get_context_summary(
                query=latest_query,
                workflow_id=workflow_id,
                max_tokens=1500
            )
            rag_context["summary"] = rag_summary
            
            # Determine if we have enough context to answer directly
            # Use LLM to assess if RAG context is sufficient
            assessment_llm = ChatOllama(
                model="gpt-oss:20b",
                temperature=0,
                base_url="http://100.91.155.118:11434"
            )
            
            class RAGAssessment(BaseModel):
                sufficient_to_answer: bool = Field(
                    description="Whether the RAG context is sufficient to answer the user's query"
                )
                confidence: float = Field(
                    description="Confidence level (0-1) that the context is sufficient",
                    ge=0.0, le=1.0
                )
                reasoning: str = Field(
                    description="Brief explanation of the assessment"
                )
            
            structured_llm = assessment_llm.with_structured_output(RAGAssessment)
            
            assessment_prompt = f"""
            User Query: {latest_query}
            
            Available Context from Previous Work:
            {rag_summary[:2000]}
            
            Number of relevant documents found: {len(all_contexts)}
            - Plot analyses: {len(plot_contexts)}
            - Code executions: {len(code_contexts)}
            - Summaries: {len(summary_contexts)}
            
            Assess whether this context is sufficient to answer the user's query without running new code.
            Consider:
            1. Does the context directly address the user's question?
            2. Is the information complete and current?
            3. Would running new code likely provide significantly better information?
            """
            
            assessment = structured_llm.invoke([
                SystemMessage(
                    content="Assess if the provided context is sufficient to answer the user's query."
                ),
                HumanMessage(content=assessment_prompt)
            ])
            
            rag_context["confidence"] = assessment.confidence
            can_answer_from_rag = assessment.sufficient_to_answer
            
            logger.info(f"✅ RAG Assessment: sufficient={assessment.sufficient_to_answer}, "
                       f"confidence={assessment.confidence:.2f}")
            logger.info(f"   Reasoning: {assessment.reasoning}")
            
            # Add stage metadata for tracking
            stage_metadata = {
                "rag_documents_found": len(all_contexts),
                "rag_assessment_sufficient": assessment.sufficient_to_answer,
                "rag_assessment_confidence": assessment.confidence,
                "rag_assessment_reasoning": assessment.reasoning,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store this assessment in RAG for learning
            if assessment.sufficient_to_answer:
                rag.add_summary(
                    summary=f"Query answered from RAG context: {latest_query[:100]}",
                    stage_name=f"{stage_name}_rag_answer",
                    workflow_id=workflow_id,
                    metadata=stage_metadata
                )
        else:
            logger.info("❌ No relevant context found in RAG")
            stage_metadata = {
                "rag_documents_found": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"❌ Error querying RAG: {e}")
        stage_metadata = {"error": str(e)}
    
    return {
        "rag_context": rag_context,
        "can_answer_from_rag": can_answer_from_rag,
        "stage_metadata": stage_metadata
    }


def route_after_rag_check(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    """
    Route based on RAG assessment.
    If RAG has sufficient info, go straight to summary.
    Otherwise, continue with normal workflow.
    """
    if state.get("can_answer_from_rag", False):
        logger.info("➡️ Routing to summarize (RAG has sufficient context)")
        return "summarize_from_rag"
    else:
        logger.info("➡️ Routing to check existing outputs (need more analysis)")
        return "check_existing_outputs"


def summarize_from_rag(state: State, runtime: Runtime[ExecutionDeps]) -> dict:
    """
    Generate summary directly from RAG context without code execution.
    """
    logger.info("📝 Generating summary from RAG context...")
    
    rag_context = state.get("rag_context", {})
    user_messages = state.get("messages", [])
    latest_query = user_messages[-1].content if user_messages else ""
    
    llm = ChatOllama(
        model="gemma3:27b",
        temperature=0,
        base_url="http://100.91.155.118:11434"
    )
    
    # Build comprehensive context from RAG
    context_parts = [
        "=== Answer based on previous analysis ===",
        "",
        rag_context.get("summary", ""),
        "",
        f"Confidence: {rag_context.get('confidence', 0):.1%}",
        f"Sources: {len(rag_context.get('contexts', []))} relevant documents"
    ]
    
    context = "\n".join(context_parts)
    
    response = llm.invoke([
        SystemMessage(
            content="""
            Answer the user's query using the provided context from previous analyses.
            Be specific and reference the relevant findings.
            Make it clear this is based on existing analysis results.
            Do not mention technical details about RAG or context retrieval.
            """
        ),
        HumanMessage(content=f"Query: {latest_query}"),
        HumanMessage(content=f"Context:\n{context}")
    ])
    
    # Store this interaction in RAG
    rag = runtime.context.get("rag")
    workflow_id = state.get("workflow_id", "default")
    stage_name = state.get("stage_name", "")
    
    if rag and rag.enabled:
        rag.add_summary(
            summary=f"RAG-based answer: {response.content[:500]}",
            stage_name=f"{stage_name}_rag_response",
            workflow_id=workflow_id,
            metadata={
                "answered_from_rag": True,
                "query": latest_query[:200],
                "timestamp": datetime.now().isoformat()
            }
        )
    
    logger.info("✅ Generated summary from RAG context")
    
    return {"summary": response.content}


def check_existing_outputs(state: State, runtime: Runtime[ExecutionDeps]) -> dict:
    """
    Check for existing outputs and artifacts from previous executions.
    Enhanced with cache checking and RAG integration.
    """
    output_manager = runtime.context["output_manager"]
    rag = runtime.context.get("rag")
    stage_name = state.get("stage_name", "")
    workflow_id = state.get("workflow_id", "default")
    
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
            
            # Add console output to RAG if not already there
            if rag and rag.enabled and existing["console_output"]:
                rag.add_code_execution(
                    code="# Previous execution",
                    stdout=existing["console_output"][:1000],
                    stderr="",
                    stage_name=stage_name,
                    workflow_id=workflow_id,
                    success=True,
                    metadata={"from_existing": True, "timestamp": datetime.now().isoformat()}
                )
        
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
                    
                    # Still add to RAG for this workflow if not present
                    if rag and rag.enabled:
                        rag.add_plot_analysis(
                            plot_name=plot_name,
                            plot_path=plot_path,
                            analysis=cached_analysis['analysis'],
                            stage_name=stage_name,
                            workflow_id=workflow_id,
                            metadata={"from_cache": True, "timestamp": datetime.now().isoformat()}
                        )
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
                    workflow_id=workflow_id,
                    metadata={"timestamp": datetime.now().isoformat()}
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
    Uses both existing outputs and RAG context to make decision.
    """
    logger.info("🤔 Determining if new code execution is needed...")
    if state.get("input_data_path") is None:
        return "no_code"
    
    user_query = state["messages"]
    existing_outputs = state.get("existing_outputs", {})
    plot_analyses = state.get("plot_analyses", [])
    rag_context = state.get("rag_context", {})
    
    # Get latest user message
    latest_message = user_query[-1].content if user_query else ""
    
    class DecisionReasoning(BaseModel):
        can_answer_from_existing: bool = Field(...)
        needs_new_code: bool = Field(...)
        reasoning: str = Field(...)
    
    llm = ChatOllama(
        model="qwen3:30b",
        temperature=0,
        base_url="http://100.91.155.118:11434"
    )
    structured_llm = llm.with_structured_output(DecisionReasoning)
    
    # Build concise context including RAG info
    context_parts = []
    
    # Include RAG context summary if available
    if rag_context.get("has_relevant_context"):
        context_parts.append(f"📚 RAG Context Available: {len(rag_context.get('contexts', []))} documents")
        if rag_context.get("summary"):
            brief_rag = rag_context["summary"][:300] + "..."
            context_parts.append(f"RAG Summary: {brief_rag}")
    
    if existing_outputs.get("has_plots"):
        context_parts.append(f"📊 Available: {len(existing_outputs['plots'])} plots")
    
    if plot_analyses:
        context_parts.append(f"🔍 Analyzed: {len(plot_analyses)} plots")
        # Add brief summaries (not full analyses)
        for pa in plot_analyses[:2]:
            brief = pa['analysis'][:150] + "..." if len(pa['analysis']) > 150 else pa['analysis']
            context_parts.append(f"  • {pa['plot_name']}: {brief}")
    
    context = "\n".join(context_parts) if context_parts else "No existing outputs"
    
    result = structured_llm.invoke([
        SystemMessage(
            content=f"""Determine if new code execution is needed.
            
            Available context:
            {context}

            Data provided at: `{Path(state['input_data_path']).name}`
            
            Consider both existing outputs and RAG context.
            Prefer using existing information when sufficient."""
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


def write_code_for_task(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    """
    Enhanced code generation with comprehensive RAG context.
    """
    code_llm = ChatOllama(
        model="qwen3-coder:30b",
        temperature=0,
        base_url="http://100.91.155.118:11434")

    structured_code_llm = code_llm.with_structured_output(Code)

    input_data_path = Path(state["input_data_path"])
    user_query = state["messages"]
    existing_outputs = state.get("existing_outputs", {})
    rag_context = state.get("rag_context", {})
    
    # Get relevant context from RAG
    rag = runtime.context.get("rag")
    workflow_id = state.get("workflow_id", "default")
    latest_message = user_query[-1].content if user_query else ""
    
    # Get code-specific context from RAG
    code_examples = []
    if rag and rag.enabled:
        code_contexts = rag.query_relevant_context(
            query=latest_message,
            workflow_id=workflow_id,
            doc_types=["code_execution"],
            n_results=2
        )
        for ctx in code_contexts:
            if "Code:" in ctx.get("document", ""):
                code_examples.append(ctx["document"][:500])
    
    context_info = []
    if existing_outputs.get("has_plots"):
        context_info.append(f"Note: {len(existing_outputs['plots'])} plots already exist from previous analysis")
    
    if code_examples:
        context_info.append("Previous relevant code approaches:")
        for example in code_examples[:2]:
            context_info.append(f"```python\n{example}\n```")
    
    if rag_context.get("summary"):
        context_info.append(f"Context from previous work: {rag_context['summary'][:300]}")
    
    context_str = "\n".join(context_info) if context_info else ""
    
    logger.info(f"Input data path for code generation: {input_data_path}")
    
    system_prompt = f"""
    Write Python code to load data from `{input_data_path.name}` and solve the user's query. 
    Save plots with descriptive names.
    Use pandas for data manipulation.
    Always load the data from `{input_data_path.name}`.
    
    {context_str}
    
    Build upon previous work if relevant, but create new analysis as requested.
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
    Execute code and store results in RAG for future queries.
    """
    executor = runtime.context["executor"]
    output_manager = runtime.context["output_manager"]
    rag = runtime.context.get("rag")
    
    stage_name = state["stage_name"]
    workflow_id = state.get("workflow_id", "default")
    code = state["code"]
    stage_metadata = state.get("stage_metadata", {})

    result = await executor.execute_with_output_manager(
        code=code,
        stage_name=stage_name,
        output_manager=output_manager,
        code_filename="code_with_data.py"
    )
    
    # Add execution results to RAG with enhanced metadata
    if rag and rag.enabled:
        # Prepare metadata - ensure no lists or complex types
        execution_metadata = {
            "execution_time": result.execution_time_seconds,
            "file_count": len(result.generated_files) if result.generated_files else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add stage metadata if present (already cleaned)
        if stage_metadata:
            execution_metadata.update(stage_metadata)
        
        # Add code execution
        rag.add_code_execution(
            code=code,
            stdout=result.stdout,
            stderr=result.stderr,
            stage_name=stage_name,
            workflow_id=workflow_id,
            success=result.success,
            metadata=execution_metadata
        )
        
        # If there are generated files, add summary about them
        if result.generated_files:
            file_summary = f"Generated {len(result.generated_files)} files: {', '.join(result.generated_files[:5])}"
            rag.add_summary(
                summary=file_summary,
                stage_name=f"{stage_name}_files",
                workflow_id=workflow_id,
                metadata={
                    "file_count": len(result.generated_files),
                    "file_names": ", ".join(result.generated_files[:10])  # Convert list to string
                }
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


def route_after_check_outputs(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    """Route based on whether plots exist to analyze."""
    if state["existing_outputs"].get("has_plots"):
        return "analyze_plots"
    else:
        return "determine_code_needed"


def summarize_results(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    """
    Enhanced summarization that combines all sources:
    - RAG context from previous work
    - Current execution results
    - Plot analyses
    Stores comprehensive summary in RAG for future use.
    """
    code_output = state.get("code_output", None)
    plot_analyses = state.get("plot_analyses", [])
    existing_outputs = state.get("existing_outputs", {})
    rag_context = state.get("rag_context", {})
    
    # Get RAG and workflow info
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
    
    # Build comprehensive context
    context_parts = []
    
    # Include RAG context if available
    if rag_context.get("has_relevant_context") and rag_context.get("summary"):
        context_parts.append("=== Previous Related Work ===")
        context_parts.append(rag_context["summary"][:800])
        context_parts.append("")
    
    # Include current execution results
    if stdout or error:
        context_parts.append("=== Current Analysis Results ===")
        context_parts.append(f"Output: {stdout[:500]}")
        if error:
            context_parts.append(f"Errors: {error[:200]}")
        context_parts.append("")
    
    # Include plot analyses
    if plot_analyses:
        context_parts.append("=== Visual Analysis ===")
        for i, pa in enumerate(plot_analyses[:3], 1):
            brief = pa['analysis'][:200] + "..." if len(pa['analysis']) > 200 else pa['analysis']
            context_parts.append(f"Plot {i} ({pa['plot_name']}): {brief}")
        context_parts.append("")
    
    context = "\n".join(context_parts)
    
    # Generate comprehensive summary
    summary_prompt = """
    Provide a comprehensive answer to the user's query using ALL available context.
    
    Structure your response to:
    1. Directly answer the user's question
    2. Reference specific findings from the analysis
    3. Mention relevant visualizations when applicable
    4. Build upon previous work if relevant
    5. Be concise but thorough
    
    Do not mention technical implementation details.
    """
    
    summary = llm.invoke([
        SystemMessage(content=summary_prompt),
        HumanMessage(content=f"User Query: {latest_query}"),
        HumanMessage(content=f"Context:\n\n{context}")
    ])
    
    # Store comprehensive summary in RAG
    if rag and rag.enabled:
        # Store the main summary
        rag.add_summary(
            summary=summary.content,
            stage_name=stage_name,
            workflow_id=workflow_id,
            metadata={
                "query": latest_query[:200],
                "had_rag_context": rag_context.get("has_relevant_context", False),
                "had_code_execution": code_output is not None,
                "had_plot_analysis": len(plot_analyses) > 0,
                "plot_analysis_count": len(plot_analyses),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Store query-response pair for better future retrieval
        rag.add_summary(
            summary=f"Q: {latest_query}\nA: {summary.content[:500]}",
            stage_name=f"{stage_name}_qa",
            workflow_id=workflow_id,
            metadata={"type": "question_answer_pair", "timestamp": datetime.now().isoformat()}
        )
    
    logger.info("✅ Generated comprehensive summary")
    
    return {'summary': summary.content}


# ============================================================================
# BUILD ENHANCED WORKFLOW GRAPH WITH RAG AS FIRST STEP
# ============================================================================

# Create the enhanced workflow graph
code_graph = StateGraph(State, ExecutionDeps)

# Add all nodes (including new RAG-first nodes)
code_graph.add_node("check_rag", check_rag_for_answer)  # NEW: First step
code_graph.add_node("summarize_from_rag", summarize_from_rag)  # NEW: Direct RAG answer
code_graph.add_node("check_existing_outputs", check_existing_outputs)
code_graph.add_node("analyze_plots", analyze_plots_with_vision_llm)
code_graph.add_node("determine_code_needed", determine_code_decision_node)
code_graph.add_node("write_code", write_code_for_task)
code_graph.add_node("move_data", move_data_to_execution_folder)
code_graph.add_node("execute_code", execute_code)
code_graph.add_node("fix_code", code_fix)
code_graph.add_node("summarize_results", summarize_results)

# Define edges - START WITH RAG CHECK
code_graph.add_edge(START, "check_rag")  # CHANGED: Start with RAG check

# Route after RAG check
code_graph.add_conditional_edges(
    "check_rag",
    route_after_rag_check,
    {
        "summarize_from_rag": "summarize_from_rag",  # Can answer directly from RAG
        "check_existing_outputs": "check_existing_outputs"  # Need more analysis
    }
)

# Direct RAG answer ends the workflow
code_graph.add_edge("summarize_from_rag", END)

# Continue with existing workflow if RAG isn't sufficient
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

# Compile the enhanced workflow
code_workflow = code_graph.compile()

logger.info("✅ Enhanced RAG-first workflow compiled successfully!")
logger.info("   Workflow now starts by checking RAG for existing context")
logger.info("   All stages persist data to RAG for future queries within the workflow")