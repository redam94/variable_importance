"""
Workflow Graph Definition

Simple 4-node workflow:

    ┌─────────────────┐
    │  gather_context │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  plan_and_decide│
    └────────┬────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
  [answer]      [execute]
      │             │
      └──────┬──────┘
             │
    ┌────────▼────────┐
    │    summarize    │
    └─────────────────┘
"""

from loguru import logger
from langgraph.graph import StateGraph, START, END

from .state import State, Deps
from .nodes import (
    gather_context,
    plan_and_decide,
    execute,
    summarize,
    answer_from_context,
    route_action,
)


def build_workflow() -> StateGraph:
    """Build and return the compiled workflow."""
    
    logger.info("🔧 Building workflow...")
    
    graph = StateGraph(State, Deps)
    
    # Add nodes
    graph.add_node("gather_context", gather_context)
    graph.add_node("plan_and_decide", plan_and_decide)
    graph.add_node("execute", execute)
    graph.add_node("answer", answer_from_context)
    graph.add_node("summarize", summarize)
    
    # Define flow
    graph.add_edge(START, "gather_context")
    graph.add_edge("gather_context", "plan_and_decide")
    
    # Route based on decision
    graph.add_conditional_edges(
        "plan_and_decide",
        route_action,
        {"execute": "execute", "answer": "answer"}
    )
    
    # Both paths lead to summarize
    graph.add_edge("execute", "summarize")
    graph.add_edge("answer", "summarize")
    
    graph.add_edge("summarize", END)
    
    logger.info("✅ Workflow built: gather → plan → (execute|answer) → summarize")
    
    return graph.compile()


# Build on import
workflow = build_workflow()