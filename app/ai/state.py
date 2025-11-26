from typing import TypedDict, Any, List, Union, Annotated
import operator

from langchain.messages import HumanMessage, AIMessage, SystemMessage

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
    plan: str
    context_summary: str


class ExecutionDeps(TypedDict):
    executor: Any  # OutputCapturingExecutor
    output_manager: Any  # OutputManager
    plot_cache: Any  # PlotAnalysisCache
    rag: Any  # ContextRAG
    rag_llm: str
    rag_summary_llm: str
    code_llm: str