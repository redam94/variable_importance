"""
Workflow Graph V2 - Modular workflow with dynamic routing and verification.

Graph Structure:

    ┌─────────────────┐
    │  gather_context │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │      plan       │
    └────────┬────────┘
             │
      ┌──────┴──────────────┐
      │                     │
      ▼                     ▼
  [execute]            [answer]
      │                     │
      ▼                     │
 [plot_analysis]            │
      │                     │
      └──────┬──────────────┘
             │
    ┌────────▼────────┐
    │    summarize    │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │     verify      │
    └────────┬────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
   [done]      [retry loop]
"""

from loguru import logger
from langgraph.graph import StateGraph, START, END

from .state import State, Deps
from .nodes_v2 import (
    gather_context,
    plan,
    execute,
    analyze_plots,
    summarize,
    answer_from_context,
    verify,
    route_action,
    route_after_verify,
)


def build_workflow_v2() -> StateGraph:
    """Build workflow with verification loop."""
    
    logger.info("🔧 Building workflow v2...")
    
    graph = StateGraph(State, Deps)
    
    # Add nodes
    graph.add_node("gather_context", gather_context)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("analyze_plots", analyze_plots)
    graph.add_node("answer", answer_from_context)
    graph.add_node("summarize", summarize)
    graph.add_node("verify", verify)
    
    # Increment retry counter node
    def increment_retry(state: State, runtime) -> dict:
        return {"retry_count": state.get("retry_count", 0) + 1}
    
    graph.add_node("increment_retry", increment_retry)
    
    # Main flow
    graph.add_edge(START, "gather_context")
    graph.add_edge("gather_context", "plan")
    
    # Route based on plan
    graph.add_conditional_edges(
        "plan",
        route_action,
        {
            "execute": "execute",
            "answer": "answer",
            "analyze_plots": "analyze_plots",
        }
    )
    
    # Execute -> optionally analyze plots -> summarize
    graph.add_edge("execute", "analyze_plots")
    graph.add_edge("analyze_plots", "summarize")
    
    # Answer -> summarize (for consistency)
    graph.add_edge("answer", "summarize")
    
    # Summarize -> verify
    graph.add_edge("summarize", "verify")
    
    # Verify -> done or retry
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "done": END,
            "execute": "increment_retry",
            "gather_context": "increment_retry",
            "summarize": "increment_retry",
        }
    )
    
    # Retry increments counter then routes back
    def route_retry(state: State, runtime) -> str:
        action = state.get("verification", {}).get("suggested_action", "done")
        if action == "retry_code":
            return "execute"
        elif action == "add_context":
            return "gather_context"
        else:
            return "summarize"
    
    graph.add_conditional_edges(
        "increment_retry",
        route_retry,
        {
            "execute": "execute",
            "gather_context": "gather_context",
            "summarize": "summarize",
        }
    )
    
    logger.info("✅ Workflow v2 built with verification loop")
    
    return graph.compile()


def build_simple_workflow() -> StateGraph:
    """Build simplified workflow without verification (for faster iteration)."""
    
    logger.info("🔧 Building simple workflow...")
    
    graph = StateGraph(State, Deps)
    
    # Add nodes
    graph.add_node("gather_context", gather_context)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("analyze_plots", analyze_plots)
    graph.add_node("answer", answer_from_context)
    graph.add_node("summarize", summarize)
    
    # Flow
    graph.add_edge(START, "gather_context")
    graph.add_edge("gather_context", "plan")
    
    graph.add_conditional_edges(
        "plan",
        route_action,
        {
            "execute": "execute",
            "answer": "answer",
            "analyze_plots": "analyze_plots",
        }
    )
    
    graph.add_edge("execute", "analyze_plots")
    graph.add_edge("analyze_plots", "summarize")
    graph.add_edge("answer", "summarize")
    graph.add_edge("summarize", END)
    
    logger.info("✅ Simple workflow built")
    
    return graph.compile()


# Default to v2 workflow
workflow = build_workflow_v2()
simple_workflow = build_simple_workflow()