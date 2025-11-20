"""
Complete Enhanced Workflow Model - Ready for Streamlit Integration

This is a complete, standalone model file that includes:
- All node functions (enhanced and original)
- Complete StateGraph construction
- Compiled workflow ready to use
- RAG context integration
- User feedback mechanism

Usage in Streamlit:
    from model_complete import code_workflow, State, ExecutionDeps
    
    result = await code_workflow.ainvoke(
        initial_state,
        context=execution_context
    )
"""

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
from langgraph.graph import StateGraph, START, END


# ============================================================================
# DATA MODELS
# ============================================================================

class Code(BaseModel):
    code: str = Field(..., description="Python code to be executed")
    task: str = Field(..., description="Description of the task to be performed")
    reasoning: str = Field(default="", description="Explanation of the code approach")


class State(TypedDict):
    code: str
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], operator.add]
    input_data_path: str
    stage_name: str
    code_output: Any  # ExecutionResult
    summary: str
    graph_summaries: Annotated[List[str], operator.add]
    existing_outputs: dict
    plot_analyses: Annotated[List[dict], operator.add]
    skip_execution: bool
    workflow_id: str
    rag_context: dict
    can_answer_from_rag: bool
    stage_metadata: dict
    fix_attempt_count: int
    last_error: str
    needs_user_feedback: bool
    user_feedback: str


class ExecutionDeps(TypedDict):
    executor: Any  # OutputCapturingExecutor
    output_manager: Any  # OutputManager
    plot_cache: Any  # PlotAnalysisCache
    rag: Any  # ContextRAG


# ============================================================================
# NODE FUNCTIONS - RAG & EXISTING OUTPUT CHECKS
# ============================================================================

def check_rag_for_answer(state: State, runtime) -> dict:
    """Check RAG for relevant context to answer user's query."""
    rag = runtime.context.get("rag")
    workflow_id = state.get("workflow_id", "default")
    
    user_messages = state.get("messages", [])
    latest_query = user_messages[-1].content if user_messages else ""
    
    logger.info(f"🔍 Checking RAG for existing context...")
    
    rag_context = {
        "has_relevant_context": False,
        "contexts": [],
        "summary": "",
        "confidence": 0.0
    }
    can_answer_from_rag = False
    
    if not rag or not rag.enabled:
        logger.info("❌ RAG not available")
        return {
            "rag_context": rag_context,
            "can_answer_from_rag": can_answer_from_rag
        }
    
    try:
        # Query RAG for relevant context
        contexts = rag.query_relevant_context(
            query=latest_query,
            workflow_id=workflow_id,
            n_results=10
        )
        
        if contexts:
            rag_context["has_relevant_context"] = True
            rag_context["contexts"] = contexts
            
            rag_summary = rag.get_context_summary(
                query=latest_query,
                workflow_id=workflow_id,
                max_tokens=1500
            )
            rag_context["summary"] = rag_summary
            
            llm_answer = ChatOllama(
                model='gemma3:27b',
                temperature=0,
                base_url="http://100.91.155.118:11434"
            )
            class RAGDecision(BaseModel):
                can_answer: bool = Field(
                    description="Whether the RAG context is sufficient to answer the query"
                )
                confidence: float = Field(
                    description="Confidence level (0-1) in this decision",
                    ge=0.0, le=1.0
                )
                reasoning: str = Field(
                    description="Brief explanation of the decision"
                )
            structured_llm = llm_answer.with_structured_output(RAGDecision)
            decision_prompt = f"""
Using the provided context, determine if it is sufficient to answer the user's query.
User Query: {latest_query}

RAG Context Summary:
{rag_summary}
Guidelines:
1. If the context directly addresses the query, can_answer = True
2. If the context is only tangentially related or insufficient, can_answer = False
3. If the user asks for specific calculations or analyses not covered by the context, can_answer = False
4. If the user asks for interpretations that can be drawn from the context, can_answer = True
5. If in doubt, err on the side of can_answer = False

Be decisive in your assessment.

IF USER ASKS FOR ANALYSIS TO BE REPEATED OR NEW CALCULATIONS, SET can_answer = FALSE.
"""
            decision = structured_llm.invoke(decision_prompt)
            rag_context["confidence"] = decision.confidence
            can_answer_from_rag = decision.can_answer

            logger.info(f"✅ Found {len(contexts)} relevant chunks (confidence: {rag_context['confidence']:.2f})")
            logger.info(f"🎯 Can answer from RAG: {can_answer_from_rag} (reasoning: {decision.reasoning})")
    
    except Exception as e:
        logger.error(f"❌ Error querying RAG: {e}")
    
    return {
        "rag_context": rag_context,
        "can_answer_from_rag": can_answer_from_rag
    }


def route_after_rag_check(state: State, runtime) -> str:
    """Route based on RAG assessment."""
    if state.get("can_answer_from_rag", False):
        logger.info("➡️ Routing to summarize (RAG has sufficient context)")
        return "summarize_from_rag"
    else:
        logger.info("➡️ Routing to check existing outputs")
        return "check_existing_outputs"


def summarize_from_rag(state: State, runtime) -> dict:
    """Generate summary directly from RAG context."""
    logger.info("📝 Generating summary from RAG context...")
    
    rag_context = state.get("rag_context", {})
    user_messages = state.get("messages", [])
    latest_query = user_messages[-1].content if user_messages else ""
    
    llm = ChatOllama(
        model="gemma3:27b",
        temperature=0,
        base_url="http://100.91.155.118:11434"
    )
    
    context = rag_context.get("summary", "")
    
    response = llm.invoke([
        SystemMessage(
            content="Answer the user's query using the provided context from previous analyses."
        ),
        HumanMessage(content=f"Query: {latest_query}"),
        HumanMessage(content=f"Context:\n{context}")
    ])
    
    logger.info("✅ Generated summary from RAG context")
    
    return {"summary": response.content}


def check_existing_outputs(state: State, runtime) -> dict:
    """Check for existing outputs from previous executions."""
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
            data_files = list(data_dir.glob("*.csv"))
            if data_files:
                existing["has_data"] = True
                existing["data_files"] = [str(d) for d in data_files]
        
        # Check console output
        console_file = stage_dir / "console_output.txt"
        if console_file.exists():
            existing["has_console_output"] = True
            with open(console_file, 'r') as f:
                existing["console_output"] = f.read()
                
    except Exception as e:
        logger.warning(f"⚠️ Error checking existing outputs: {e}")
    
    return {"existing_outputs": existing, "skip_execution": False}


def analyze_plots_with_vision_llm(state: State, runtime) -> dict:
    """Analyze plots using vision LLM."""
    output_manager = runtime.context["output_manager"]
    stage_name = state.get("stage_name", "")
    rag = runtime.context.get("rag")
    try:
        stage_dir = output_manager.get_stage_dir(stage_name)
        
        # Check for plots
        plots_dir = stage_dir / "plots"
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg"))
    except Exception as e:
        logger.warning(f"⚠️ Error accessing plots for analysis: {e}")
        plot_files = []
                
    if not plot_files:
        logger.info("No plots to analyze")
        return {"plot_analyses": []}
    
    logger.info(f"🔍 Analyzing {len(plot_files)} plots...")
    plot_cache = runtime.context.get("plot_cache")
    vision_llm = ChatOllama(
        model="qwen3-vl:30b",
        temperature=0,
        base_url="http://100.91.155.118:11434"
    )
    
    analyses = []
    
    for plot_path in plot_files:
        try:
            plot_name = Path(plot_path).name
            
            # Check cache
            if plot_cache:
                cached = plot_cache.get(plot_path)
                
                if cached:
                    logger.info(f"✅ Cache hit for plot: {cached}")
                    analyses.append(cached)
                    logger.info(f"⚡ Used cached analysis: {plot_name}")
                    # if rag and rag.enabled:
                    #     rag.add_plot_analysis(
                    #         plot_name=plot_name,
                    #         plot_path=str(plot_path),
                    #         analysis=cached['analysis'],
                    #         stage_name=stage_name,
                    #         workflow_id=state.get("workflow_id", "default")
                    #     )
                    continue
            
            # Analyze with vision LLM
            with open(plot_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            response = vision_llm.invoke([
                SystemMessage(content="Analyze this visualization plot and describe key insights."),
                HumanMessage(content=[
                    {"type": "image_url", "image_url": f"data:image/png;base64,{image_data}"},
                    {"type": "text", "text": "Analyze this plot."}
                ])
            ])
            
            analysis = {
                "plot_path": plot_path,
                "plot_name": plot_name,
                "analysis": response.content
            }
            analyses.append(analysis)
            
            if plot_cache:
                plot_cache.set(plot_path, analysis)

            if rag and rag.enabled:
                rag.add_plot_analysis(
                    plot_name=plot_name,
                    plot_path=str(plot_path),
                    analysis=response.content,
                    stage_name=stage_name,
                    workflow_id=state.get("workflow_id", "default")
                )
            

            logger.info(f"✅ Analyzed: {plot_name}")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {plot_path}: {e}")
    
    return {"plot_analyses": analyses}


def route_after_check_outputs(state: State, runtime) -> str:
    """Route based on whether plots exist."""
    if state["existing_outputs"].get("has_plots"):
        return "analyze_plots"
    else:
        return "determine_code_needed"


def determine_if_code_needed(state: State, runtime) -> str:
    """Determine if new code execution is needed."""
    logger.info("🤔 Determining if new code execution is needed...")
    
    # No data path = can't execute code
    if state.get("input_data_path") is None:
        logger.info("❌ No data path - cannot execute code")
        return "no_code"
    
    # Gather all available context
    user_messages = state.get("messages", [])
    latest_query = user_messages[-1].content if user_messages else ""
    
    existing_outputs = state.get("existing_outputs", {})
    plot_analyses = state.get("plot_analyses", [])
    rag_context = state.get("rag_context", {})
    
    # Build comprehensive context for LLM
    context_parts = []
    
    # RAG context
    if rag_context.get("has_relevant_context"):
        context_parts.append(f"📚 RAG Context: {len(rag_context.get('contexts', []))} relevant documents found")
        if rag_context.get("summary"):
            summary_preview = rag_context["summary"][:400] + "..." if len(rag_context["summary"]) > 400 else rag_context["summary"]
            context_parts.append(f"Summary: {summary_preview}")
    
    # Existing outputs
    if existing_outputs.get("has_plots"):
        context_parts.append(f"📊 Existing: {len(existing_outputs['plots'])} plots available")
    
    if existing_outputs.get("has_console_output"):
        console_preview = existing_outputs["console_output"][:300] + "..." if len(existing_outputs.get("console_output", "")) > 300 else existing_outputs.get("console_output", "")
        context_parts.append(f"Console Output: {console_preview}")
    
    # Plot analyses
    if plot_analyses:
        context_parts.append(f"🔍 Analyses: {len(plot_analyses)} plots analyzed")
        for i, pa in enumerate(plot_analyses[:2], 1):
            analysis_preview = pa['analysis'][:200] + "..." if len(pa['analysis']) > 200 else pa['analysis']
            context_parts.append(f"  Plot {i} ({pa['plot_name']}): {analysis_preview}")
    
    if not context_parts:
        context_parts.append("No existing outputs or context available")
    
    context_summary = "\n".join(context_parts)
    
    # Define structured decision model
    class CodeDecision(BaseModel):
        needs_code_execution: bool = Field(
            description="Whether new code execution is needed to answer the query"
        )
        confidence: float = Field(
            description="Confidence level (0-1) in this decision",
            ge=0.0, le=1.0
        )
        reasoning: str = Field(
            description="Brief explanation of the decision"
        )
        can_answer_with_existing: bool = Field(
            description="Whether existing outputs alone can answer the query"
        )
    
    # Use LLM for decision
    llm = ChatOllama(
        model="qwen3:30b",
        temperature=0,
        base_url="http://100.91.155.118:11434"
    )
    structured_llm = llm.with_structured_output(CodeDecision)
    
    decision_prompt = f"""
Determine if new code execution is needed to answer the user's query.

User Query: {latest_query}

Available Context:
{context_summary}

Data File: {Path(state['input_data_path']).name}

Guidelines:
1. ERR ON THE SIDE OF CODE EXECUTION - When in doubt, execute code
2. Execute code if:
   - Query asks for NEW analysis not covered by existing outputs
   - Query requests specific calculations, statistics, or transformations
   - Query asks to create new visualizations
   - Existing outputs are incomplete or don't fully address the query
   - Query mentions specific variables/columns that need analysis
3. Use existing outputs ONLY if:
   - Query is purely about interpreting existing plots/results
   - Existing analyses completely and directly answer the query
   - No new computation or visualization is requested

Be decisive and prefer code execution for thorough analysis.
"""
    
    try:
        decision = structured_llm.invoke([
            SystemMessage(
                content="You are an expert at determining if code execution is needed for data analysis queries. Bias toward code execution for comprehensive analysis."
            ),
            HumanMessage(content=decision_prompt)
        ])
        
        logger.info(f"🎯 Decision: needs_code={decision.needs_code_execution}, confidence={decision.confidence:.2f}")
        logger.info(f"   Reasoning: {decision.reasoning}")
        
        # Route based on decision with bias toward code execution
        if decision.needs_code_execution:
            logger.info("➡️ Route: WRITE_CODE (new analysis needed)")
            return "write_code"
        elif decision.can_answer_with_existing and decision.confidence >= 0.7:
            logger.info("➡️ Route: USE_EXISTING (high confidence in existing outputs)")
            return "use_existing"
        else:
            # Default to code execution if uncertain
            logger.info("➡️ Route: WRITE_CODE (default to execution when uncertain)")
            return "write_code"
            
    except Exception as e:
        logger.error(f"❌ Error in LLM decision: {e}")
        # Fallback: prefer code execution
        logger.info("➡️ Route: WRITE_CODE (fallback due to error)")
        return "write_code"


def determine_code_decision_node(state: State, runtime) -> dict:
    """Decision node for code execution."""
    decision = determine_if_code_needed(state, runtime)
    return {"skip_execution": decision == "use_existing"}


def route_after_decision(state: State, runtime) -> str:
    """Route after code decision."""
    if state.get("skip_execution", False):
        return "summarize_results"
    else:
        return "write_code"


# ============================================================================
# NODE FUNCTIONS - CODE GENERATION & EXECUTION
# ============================================================================

def write_code_for_task(state: State, runtime) -> dict:
    """Enhanced code generation with RAG context."""
    code_llm = ChatOllama(
        model="qwen3-coder:30b",
        temperature=0,
        base_url="http://100.91.155.118:11434"
    )
    structured_code_llm = code_llm.with_structured_output(Code)

    input_data_path = Path(state["input_data_path"])
    user_query = state["messages"]
    
    # Get RAG context
    rag = runtime.context.get("rag")
    workflow_id = state.get("workflow_id", "default")
    latest_message = user_query[-1].content if user_query else ""
    
    relevant_context = ""
    if rag and rag.enabled:
        contexts = rag.query_relevant_context(
            query=f"{latest_message} code examples",
            workflow_id=workflow_id,
            doc_types=["code_execution"],
            n_results=5
        )
        
        if contexts:
            context_parts = ["=== Relevant Code Examples ==="]
            for ctx in contexts[:3]:
                context_parts.append(ctx["document"][:300])
            relevant_context = "\n".join(context_parts)
            logger.info(f"📚 Using {len(contexts)} context chunks")
    
    system_prompt = f"""
Write Python code to load data from `{input_data_path.name}` and solve the user's query.

Requirements:
- Load data using pandas: pd.read_csv('{input_data_path.name}')
- Save plots with descriptive names
- Use matplotlib with 'Agg' backend

{relevant_context}

Provide reasoning for your approach.
"""
    
    code = structured_code_llm.invoke([
        SystemMessage(content=system_prompt),
        *user_query[-3:],
    ])
    
    logger.info(f"✅ Generated code")
    
    return {
        'code': code.code,
        'fix_attempt_count': 0,
        'needs_user_feedback': False
    }


def move_data_to_execution_folder(state: State, runtime) -> dict:
    """Move data file to execution folder."""
    stage_name = state["stage_name"]
    path = runtime.context["output_manager"].get_stage_dir(stage_name)
    execution_path = path / "execution" / Path(state['input_data_path']).name
    
    import shutil
    if not execution_path.parent.exists():
        execution_path.parent.mkdir(parents=True)
    shutil.copy(state["input_data_path"], execution_path)
    
    return {}


async def execute_code(state: State, runtime) -> dict:
    """Execute code and store results."""
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
        code_filename="code.py"
    )
    
    # Add to RAG
    if rag and rag.enabled:
        rag.add_code_execution(
            code=code,
            stdout=result.stdout,
            stderr=result.stderr,
            stage_name=stage_name,
            workflow_id=workflow_id,
            success=result.success,
            metadata={"execution_time": result.execution_time_seconds}
        )

    return {'code_output': result}


def check_execution_success(state: State, runtime) -> str:
    """Check execution success."""
    if state["code_output"].success:
        return "SUCCESS"
    else:
        return "FAILURE"


# ============================================================================
# NODE FUNCTIONS - CODE FIXING & USER FEEDBACK
# ============================================================================

def code_fix(state: State, runtime) -> dict:
    """Enhanced code fixing with RAG context and retry tracking."""
    code_llm = ChatOllama(
        model="qwen3-coder:30b",
        temperature=0,
        base_url="http://100.91.155.118:11434"
    )
    class CodeFix(BaseModel):
        fixed_code: str = Field(..., description="Fixed Python code after addressing the error")
        reasoning: str = Field(..., description="Explanation of the fix")
        
    structured_code_llm = code_llm.with_structured_output(CodeFix)
    
    code = state["code"]
    error = state["code_output"].stderr
    output = state['code_output'].stdout
    prompt = state["messages"][-1].content if state["messages"] else ""
    current_attempt = state.get("fix_attempt_count", 0) + 1
    logger.warning(f"🔧 Attempting code fix #{current_attempt}")
    logger.info(f"Traceback: {error}")
    logger.info(f"Output: {output[:200] if output else 'None'}")
    # Get error context from RAG
    rag = runtime.context.get("rag")
    workflow_id = state.get("workflow_id", "default")
    error_context = ""
    
    if rag and rag.enabled and error:
        contexts = rag.query_relevant_context(
            query=f"error fix: {error[:200]}",
            workflow_id=workflow_id,
            doc_types=["code_execution"],
            n_results=3
        )
        
        if contexts:
            error_context = "=== Similar Patterns ===\n" + "\n".join([
                ctx['document'][:150] for ctx in contexts
            ])
        logger.info(f"📚 Retrieved {len(contexts)} similar error contexts from RAG")
        logger.debug(f"Error Context: {error_context}")
    fix_prompt = f"""
Fix the following code error (Attempt {current_attempt}/5).

Here is the broken code:
<code snippet>
{code}
</code snippet>

Here is the traceback:
<traceback>
{error}
</traceback>

Here is the current output of the code execution:
<output>
{output[:500] if output else 'None'}
</output>

Here is some additional context that may help:
<error_context>
{error_context}
</error_context>
"""
    
    fixed_code = structured_code_llm.invoke([
        SystemMessage(content="""You are an expert Python Debugger. You are provided with a snippet of BROKEN CODE and a TRACEBACK.

Your task is to output the fixed code. To do this correctly, you must think step-by-step.

**Guidelines:**
-   **Contextualize:** Read the traceback to find the exact file and line number of the crash.
-   **Isolate:** Identify the specific variable or logic causing the crash.
-   **Correction:** specific fix for that error. If the error suggests a missing import, add it. If it suggests a type mismatch, cast the variable or change the logic.
-   **Completeness:** Return the FULL corrected code, not just the snippets that changed.

**Response Structure:**
1.  **Reasoning:** Briefly describe your thought process. Identify the line causing the error and the reason for the fix.
2.  **Fixed Code:** Provide the fully corrected code."""),
        HumanMessage(content=fix_prompt),
    ])
    
    needs_feedback = current_attempt >= 5
    logger.info(f"✅ Generated fixed code on attempt #{current_attempt}")
    logger.info(f"Reasoning: {fixed_code.reasoning}")
    
    if needs_feedback:
        logger.error(f"❌ Failed after {current_attempt} attempts. Requesting user feedback.")
    
    return {
        'code': fixed_code.fixed_code,
        'fix_attempt_count': current_attempt,
        'last_error': error,
        'needs_user_feedback': needs_feedback
    }


def check_needs_feedback(state: State, runtime) -> str:
    """Route based on feedback need."""
    if state.get("needs_user_feedback", False):
        return "request_feedback"
    else:
        return "execute_code"


def request_user_feedback(state: State, runtime) -> dict:
    """Request user feedback after failures."""
    code = state["code"]
    error = state.get("last_error", "Unknown error")
    attempt_count = state.get("fix_attempt_count", 0)
    
    logger.error(f"🛑 Code fixing failed after {attempt_count} attempts")
    
    feedback_request = f"""
⚠️ CODE EXECUTION FAILED AFTER {attempt_count} ATTEMPTS

Last Error: {error}

Current Code:
```python
{code}
```

PLEASE PROVIDE FEEDBACK:
1. What's causing the error?
2. How should I fix it?

Your feedback will help me resolve this issue.
"""
    
    return {
        'messages': [AIMessage(content=feedback_request)],
        'needs_user_feedback': True
    }


def process_user_feedback(state: State, runtime) -> dict:
    """Process user feedback and revise code."""
    user_feedback = state.get("user_feedback", "")
    
    if not user_feedback:
        logger.warning("⚠️ No user feedback provided")
        return {
            'fix_attempt_count': 0,
            'needs_user_feedback': False
        }
    
    logger.info(f"💬 Processing user feedback...")
    
    code_llm = ChatOllama(
        model="qwen3-coder:30b",
        temperature=0,
        base_url="http://100.91.155.118:11434"
    )
    structured_code_llm = code_llm.with_structured_output(Code)
    
    code = state["code"]
    error = state.get("last_error", "")
    
    feedback_prompt = f"""
USER FEEDBACK: {user_feedback}

Current Code:
```python
{code}
```

Last Error: {error}

Revise the code based on user feedback.
"""
    
    revised_code = structured_code_llm.invoke([
        SystemMessage(content="Revise the code based on user feedback."),
        HumanMessage(content=feedback_prompt),
    ])
    
    logger.info(f"✅ Revised code based on feedback")
    
    return {
        'code': revised_code.code,
        'fix_attempt_count': 0,
        'needs_user_feedback': False,
        'user_feedback': ""
    }


# ============================================================================
# NODE FUNCTIONS - SUMMARIZATION
# ============================================================================

def summarize_results(state: State, runtime) -> dict:
    """Enhanced summarization with RAG context."""
    code_output = state.get("code_output", None)
    plot_analyses = state.get("plot_analyses", [])
    rag_context = state.get("rag_context", {})
    
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
    
    # Build context
    context_parts = []
    
    if rag and rag.enabled:
        rag_summary = rag.get_context_summary(
            query=latest_query,
            workflow_id=workflow_id,
            max_tokens=1000
        )
        if rag_summary:
            context_parts.append("=== Previous Work ===")
            context_parts.append(rag_summary[:800])
    
    if stdout or error:
        context_parts.append("=== Current Results ===")
        context_parts.append(f"Output: {stdout[:500]}")
    
    if plot_analyses:
        context_parts.append("=== Visual Analysis ===")
        for pa in plot_analyses:
            context_parts.append(f"{pa['plot_name']}: {pa['analysis'][:400]}...")
    
    context = "\n".join(context_parts)
    
    summary = llm.invoke([
        SystemMessage(content="Provide a comprehensive answer using all available context."),
        HumanMessage(content=f"Query: {latest_query}"),
        HumanMessage(content=f"Context:\n{context}")
    ])
    
    # Store in RAG
    if rag and rag.enabled:
        rag.add_summary(
            summary=summary.content,
            stage_name=stage_name,
            workflow_id=workflow_id,
            metadata={"query": latest_query[:200], "timestamp": datetime.now().isoformat()}
        )
    
    logger.info("✅ Generated comprehensive summary")
    
    return {'summary': summary.content}


# ============================================================================
# BUILD COMPLETE WORKFLOW GRAPH
# ============================================================================

logger.info("🔧 Building complete enhanced workflow graph...")

code_graph = StateGraph(State, ExecutionDeps)

# Add all nodes
code_graph.add_node("check_rag", check_rag_for_answer)
code_graph.add_node("summarize_from_rag", summarize_from_rag)
code_graph.add_node("check_existing_outputs", check_existing_outputs)
code_graph.add_node("analyze_plots", analyze_plots_with_vision_llm)
code_graph.add_node("determine_code_needed", determine_code_decision_node)
code_graph.add_node("write_code", write_code_for_task)
code_graph.add_node("move_data", move_data_to_execution_folder)
code_graph.add_node("execute_code", execute_code)
code_graph.add_node("fix_code", code_fix)
code_graph.add_node("request_feedback", request_user_feedback)
code_graph.add_node("process_feedback", process_user_feedback)
code_graph.add_node("summarize_results", summarize_results)

# Define edges - START WITH RAG CHECK
code_graph.add_edge(START, "check_rag")

# Route after RAG check
code_graph.add_conditional_edges(
    "check_rag",
    route_after_rag_check,
    {
        "summarize_from_rag": "summarize_from_rag",
        "check_existing_outputs": "check_existing_outputs"
    }
)

# RAG answer ends workflow
code_graph.add_edge("summarize_from_rag", END)

# Continue workflow if RAG not sufficient
code_graph.add_conditional_edges(
    "check_existing_outputs",
    route_after_check_outputs,
    {
        "analyze_plots": "determine_code_needed",
        "determine_code_needed": "determine_code_needed"
    }
)

# code_graph.add_conditional_edges(
#     "analyze_plots",
#     determine_if_code_needed,
#     {
#         "use_existing": "summarize_results",
#         "write_code": "write_code",
#         "no_code": "summarize_results"
#     }
# )

code_graph.add_conditional_edges(
    "determine_code_needed",
    route_after_decision,
    {
        "summarize_results": "summarize_results",
        "write_code": "write_code"
    }
)

code_graph.add_edge("write_code", "move_data")
code_graph.add_edge("move_data", "execute_code")

# Execution success/failure routing
code_graph.add_conditional_edges(
    "execute_code",
    check_execution_success,
    {
        "SUCCESS": "analyze_plots",
        "FAILURE": "fix_code"
    }
)

# Fix code with feedback mechanism
code_graph.add_conditional_edges(
    "fix_code",
    check_needs_feedback,
    {
        "request_feedback": "analyze_plots",
        "execute_code": "execute_code"
    }
)

code_graph.add_edge("request_feedback", "process_feedback")
code_graph.add_edge("process_feedback", "execute_code")
code_graph.add_edge("analyze_plots", "summarize_results")
code_graph.add_edge("summarize_results", END)

# Compile the workflow
code_workflow = code_graph.compile()

logger.info("✅ Complete enhanced workflow compiled successfully!")
logger.info("   Features:")
logger.info("   - RAG-first context checking")
logger.info("   - Intelligent code generation with past examples")
logger.info("   - Error pattern learning in fixes")
logger.info("   - User feedback after 5 failures")
logger.info("   - Complete workflow ready for Streamlit")

# Export everything needed
__all__ = [
    'State',
    'ExecutionDeps',
    'Code',
    'code_workflow',  # ← Main export for Streamlit
]