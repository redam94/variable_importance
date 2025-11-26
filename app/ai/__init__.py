"""
AI Workflow Module

Simple 4-node workflow for data analysis:
    gather_context → plan_and_decide → (execute|answer) → summarize

Usage:
    from ai import workflow, State, Deps
    
    result = await workflow.ainvoke(
        {"messages": [...], "data_path": "...", ...},
        context=Deps(executor=..., output_manager=...)
    )
"""

from .state import State, Deps, Context
from .workflow import workflow

__all__ = ["workflow", "State", "Deps", "Context"]