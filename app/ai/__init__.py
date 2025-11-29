"""
AI Workflow Module

Workflows:
- workflow (v2): Full workflow with verification loop
- simple_workflow: Faster workflow without verification

Components:
- RAGSearchAgent: Intelligent RAG search with query refinement
- NodeFactory: Custom node creation
- DynamicRouter: Workflow routing decisions
"""

# State and config
from .state_v2 import State, Deps, Context, DEFAULTS

# Workflows
from .workflow_v2 import workflow, simple_workflow, build_workflow_v2, build_simple_workflow

# Factory
from .factory import NodeFactory, NodeConfig

# RAG Agent
from .rag_agent import RAGSearchAgent, RAGAgentConfig, RAGSearchResult, search_rag_with_agent

# Integration helpers
from .rag_agent_integration import gather_context_with_agent, gather_rag_context

__all__ = [
    # Workflows
    "workflow",
    "simple_workflow",
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
    
    # RAG Agent
    "RAGSearchAgent",
    "RAGAgentConfig",
    "RAGSearchResult",
    "search_rag_with_agent",
    "gather_context_with_agent",
    "gather_rag_context",
]