"""
AI Workflow Module

Two workflow versions available:

1. workflow (v2): Full workflow with verification loop
   gather_context → plan → (execute|answer) → plot_analysis → summarize → verify → (done|retry)

2. simple_workflow: Faster workflow without verification
   gather_context → plan → (execute|answer) → summarize

Usage:
    from ai import workflow, State, Deps
    
    result = await workflow.ainvoke(
        {"messages": [...], "data_path": "...", ...},
        context=Deps(executor=..., output_manager=...)
    )
    
Components:
    - factory: Node factory for custom nodes
    - tools: File search tools for code agent
    - routing: Dynamic routing decisions
    - verification: Output completeness checking
"""

# V2 (recommended)
from .state_v2 import State, Deps, Context, DEFAULTS
from .workflow_v2 import workflow, simple_workflow, build_workflow_v2, build_simple_workflow

# Factory and components
from .factory import NodeFactory, NodeConfig

# Legacy support - import conditionally to avoid circular imports
try:
    from .workflow import workflow as workflow_v1
except ImportError:
    workflow_v1 = None

__all__ = [
    # Workflows
    "workflow",
    "simple_workflow",
    "workflow_v1",
    "build_workflow_v2",
    "build_simple_workflow",
    
    # State
    "State",
    "Deps",
    "Context",
    "DEFAULTS",
    
    # Factory
    "NodeFactory",
    "NodeConfig",
]