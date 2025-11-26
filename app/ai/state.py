"""
Workflow State - Simplified

Minimal state with clear purpose for each field.
"""

from typing import TypedDict, Any, List, Union, Annotated, Optional
import operator
from langchain.messages import HumanMessage, AIMessage, SystemMessage


class Context(TypedDict, total=False):
    """All gathered context in one place."""
    rag: str
    web: str
    outputs: str
    plots: List[str]
    combined: str


class State(TypedDict, total=False):
    """Minimal workflow state."""
    
    # Input
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], operator.add]
    data_path: str
    stage_name: str
    workflow_id: str
    web_search_enabled: bool
    
    # Context (gathered once)
    context: Context
    
    # Plan & Decision
    plan: str
    action: str  # "answer" | "execute"
    
    # Execution
    code: str
    output: Any
    error: str
    
    # Result
    summary: str


class Deps(TypedDict, total=False):
    """Runtime dependencies - passed as context to workflow."""
    executor: Any
    output_manager: Any
    rag: Any
    plot_cache: Any
    
    # Progress tracking
    progress_emitter: Any  # ProgressEmitter for real-time updates
    
    # Model names
    llm: str
    code_llm: str
    vision_llm: str
    
    # Config
    base_url: str
    max_retries: int


# Default configuration
DEFAULTS = {
    "llm": "qwen3:30b",
    "code_llm": "qwen3-coder:30b",
    "vision_llm": "qwen3-vl:30b",
    "base_url": "http://100.91.155.118:11434",
    "max_retries": 3,
}