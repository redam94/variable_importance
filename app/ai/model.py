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
from dataclasses import dataclass

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

from .config import DefaultConfig
from .state import State, ExecutionDeps
from .rag_functions import get_context_from_rag, summarize_from_rag, route_after_rag_check
from .file_context import check_existing_outputs, analyze_plots_with_vision_llm


DEFAULT_CONFIG = DefaultConfig()
# ============================================================================
# DATA MODELS
# ============================================================================    
class Code(BaseModel):
    code: str = Field(..., description="Python code to be executed")
    task: str = Field(..., description="Description of the task to be performed")
    reasoning: str = Field(default="", description="Explanation of the code approach")


def planning_agent(state: State, runtime) -> dict:
    """ Agent to plan the workflow based on user query and context. """
    
    logger.info("🧠 Planning agent generating workflow plan...")
    latest_query = state["messages"][-1].content if state["messages"] else ""
    reasoning_llm = ChatOllama(
        model="gemma3:27b",
        temperature=0,
        base_url=DEFAULT_CONFIG.base_url
    )

    system_prompt = f"""
    You are an expert data scientist. You have been given a user query and must plan the best approach to answer it.
    Create a step-by-step plan considering what information might be needed, what analyses to perform, and how to structure the best answer.
    This plan will be passed to other components to generate code, execute analyses and have the ability to refer to existing outputs.
    Your goal is to understand the user's intent and create a comprehensive plan to fulfill it.
    Only provide the plan, do not write any code yourself.
    """
    plan = reasoning_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=latest_query)
    ])

    logger.info(f"🧠 Planning agent created plan: {plan.content[:100]}")

    return {'plan': plan.content}




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
        console_preview = '\n'.join(existing_outputs["console_output"])[:500]
        context_parts.append(f"Console Output:\n{console_preview}")
    
    # Plot analyses
    if plot_analyses:
        context_parts.append(f"🔍 Analyses: {len(plot_analyses)} plots analyzed")
        for i, pa in enumerate(plot_analyses, 1):
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
        base_url=DEFAULT_CONFIG.base_url
    )
    structured_llm = llm.with_structured_output(CodeDecision)
    
    decision_prompt = f"""
Determine if new code execution is needed to answer the user's query and complete the provided plan.

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

Available Context:
{context_summary}

Plan:
<plan>
{state.get("plan", "")}
</plan>

Data File: {Path(state['input_data_path']).name}

Be decisive and prefer code execution for thorough analysis.
"""
    
    try:
        decision = structured_llm.invoke([
            SystemMessage(
                content=decision_prompt
            ),
            HumanMessage(content=latest_query)
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
        base_url=DEFAULT_CONFIG.base_url
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
Use the provided plan to guide your approach. Focus on the parts of the plan that require data loading, analysis, and visualization.

Plan:
<plan>
{state.get("plan", "")}
</plan>

Requirements:
- Load data using pandas: pd.read_csv('{input_data_path.name}')
- Save plots with descriptive names
- Use matplotlib with 'Agg' backend
- Ensure code is self-contained and executable
- Wrap sections in try-except for error handling

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
        base_url=DEFAULT_CONFIG.base_url
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
        base_url=DEFAULT_CONFIG.base_url
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
        base_url=DEFAULT_CONFIG.base_url
    )
    
    # Build context
    context_parts = []
    
    if rag and rag.enabled:
        rag_summary = rag.get_context_summary(
            query=latest_query,
            workflow_id=workflow_id,
            stage_name=stage_name,
            max_tokens=1000
        )
        if rag_summary:
            context_parts.append("=== Previous Work ===")
            context_parts.append(rag_summary[:800])
    
    if stdout or error:
        context_parts.append("=== Current Results ===")
        context_parts.append(f"Output: {stdout}")
    
    if plot_analyses:
        context_parts.append("=== Visual Analysis ===")
        for pa in plot_analyses:
            context_parts.append(f"{pa['plot_name']}: {pa['analysis'][:400]}...")
    
    context = "\n".join(context_parts)
    system_prompt = f"""
You are an expert data scientist. Provide a comprehensive summary answering the user's query.
Use all available context including previous work, current results, and visual analyses.
Be clear, concise, and thorough. Do not omit important details. Use the provided plan to guide your summary.
Plan:
<plan>
{state.get("plan", "")}
</plan>
Context:
{context}
"""

    summary = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=latest_query),

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
code_graph.add_node("planning_agent", planning_agent)
code_graph.add_node("check_rag", get_context_from_rag)
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
code_graph.add_edge(START, "planning_agent")
code_graph.add_edge("planning_agent", "check_rag")

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