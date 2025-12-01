"""
Workflow Emitter Protocol - Dependency injection for workflow progress events.

Provides a protocol that workflow nodes use to emit events,
with WebSocket implementation for real-time updates.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Protocol, runtime_checkable


class EventType(str, Enum):
    """Event types emitted during workflow execution."""
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    PROGRESS = "progress"
    RAG_SEARCH_START = "rag_search_start"
    RAG_QUERY = "rag_query"
    RAG_RESULT = "rag_result"
    RAG_JUDGMENT = "rag_judgment"
    RAG_SEARCH_END = "rag_search_end"
    CODE_GENERATED = "code_generated"
    EXECUTION_START = "execution_start"
    EXECUTION_RESULT = "execution_result"
    ERROR = "error"


@dataclass
class RAGQueryEvent:
    """Details of a RAG query iteration."""
    query: str
    iteration: int
    chunks_found: int
    relevance_score: Optional[float] = None


@dataclass
class RAGSearchEvent:
    """Aggregate RAG search event for collapsible display."""
    event_id: str
    status: str  # "searching", "complete", "failed"
    total_iterations: int = 0
    total_chunks: int = 0
    final_relevance: float = 0.0
    accepted: bool = False
    queries: List[RAGQueryEvent] = field(default_factory=list)
    message: str = ""


@runtime_checkable
class WorkflowEmitter(Protocol):
    """Protocol for emitting workflow progress events."""
    
    async def emit(
        self,
        event_type: EventType,
        stage: str,
        message: str,
        data: Optional[dict] = None,
    ) -> None:
        """Emit a progress event."""
        ...

    async def stage_start(self, stage: str, description: str = "") -> None:
        """Signal stage start."""
        ...

    async def stage_end(self, stage: str, success: bool = True) -> None:
        """Signal stage end."""
        ...

    async def rag_search_start(self, stage: str) -> None:
        """Signal RAG search beginning."""
        ...

    async def rag_query(
        self,
        stage: str,
        query: str,
        iteration: int,
        chunks_found: int,
        relevance: Optional[float] = None,
    ) -> None:
        """Emit RAG query iteration event."""
        ...

    async def rag_search_end(
        self,
        stage: str,
        total_iterations: int,
        total_chunks: int,
        final_relevance: float,
        accepted: bool,
    ) -> None:
        """Signal RAG search complete."""
        ...

    async def code_generated(self, stage: str, code: str, line_count: int) -> None:
        """Code generation event."""
        ...

    async def execution_result(
        self, stage: str, success: bool, output: str = ""
    ) -> None:
        """Code execution result."""
        ...

    async def error(self, stage: str, error: str) -> None:
        """Error event."""
        ...


class WebSocketEmitter:
    """
    WebSocket-backed emitter implementation.
    
    Sends events through the ConnectionManager to connected clients.
    """

    def __init__(self, workflow_id: str, task_id: str, manager):
        self.workflow_id = workflow_id
        self.task_id = task_id
        self.manager = manager
        self._current_stage: Optional[str] = None
        self._rag_event_id: Optional[str] = None

    async def emit(
        self,
        event_type: EventType,
        stage: str,
        message: str,
        data: Optional[dict] = None,
    ) -> None:
        """Emit a progress event via WebSocket."""
        payload = {
            "type": event_type.value,
            "task_id": self.task_id,
            "stage": stage,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        if data:
            payload["data"] = data

        await self.manager.send_to_workflow(self.workflow_id, payload)

    async def stage_start(self, stage: str, description: str = "") -> None:
        self._current_stage = stage
        await self.emit(EventType.STAGE_START, stage, description)

    async def stage_end(self, stage: str, success: bool = True) -> None:
        status = "completed" if success else "failed"
        await self.emit(EventType.STAGE_END, stage, status)
        self._current_stage = None

    async def rag_search_start(self, stage: str) -> None:
        import uuid
        self._rag_event_id = f"rag_{uuid.uuid4().hex[:8]}"
        await self.emit(
            EventType.RAG_SEARCH_START,
            stage,
            "Starting intelligent RAG search...",
            {"event_id": self._rag_event_id},
        )

    async def rag_query(
        self,
        stage: str,
        query: str,
        iteration: int,
        chunks_found: int,
        relevance: Optional[float] = None,
    ) -> None:
        await self.emit(
            EventType.RAG_QUERY,
            stage,
            f"Query {iteration}: {query[:50]}...",
            {
                "event_id": self._rag_event_id,
                "query": query,
                "iteration": iteration,
                "chunks_found": chunks_found,
                "relevance": relevance,
            },
        )

    async def rag_search_end(
        self,
        stage: str,
        total_iterations: int,
        total_chunks: int,
        final_relevance: float,
        accepted: bool,
    ) -> None:
        status = "✅ Accepted" if accepted else "⚠️ Low relevance"
        await self.emit(
            EventType.RAG_SEARCH_END,
            stage,
            f"{status}: {total_chunks} chunks, {final_relevance:.0%} relevance",
            {
                "event_id": self._rag_event_id,
                "total_iterations": total_iterations,
                "total_chunks": total_chunks,
                "final_relevance": final_relevance,
                "accepted": accepted,
            },
        )
        self._rag_event_id = None

    async def code_generated(self, stage: str, code: str, line_count: int) -> None:
        await self.emit(
            EventType.CODE_GENERATED,
            stage,
            f"Generated {line_count} lines",
            {"code": code, "line_count": line_count},
        )

    async def execution_result(
        self, stage: str, success: bool, output: str = ""
    ) -> None:
        status = "success" if success else "error"
        await self.emit(
            EventType.EXECUTION_RESULT,
            stage,
            status,
            {"success": success, "output": output[:1000]},
        )

    async def error(self, stage: str, error: str) -> None:
        await self.emit(EventType.ERROR, stage, error, {"error": error})


class NullEmitter:
    """No-op emitter for when progress tracking is not needed."""

    async def emit(self, *args, **kwargs) -> None:
        pass

    async def stage_start(self, *args, **kwargs) -> None:
        pass

    async def stage_end(self, *args, **kwargs) -> None:
        pass

    async def rag_search_start(self, *args, **kwargs) -> None:
        pass

    async def rag_query(self, *args, **kwargs) -> None:
        pass

    async def rag_search_end(self, *args, **kwargs) -> None:
        pass

    async def code_generated(self, *args, **kwargs) -> None:
        pass

    async def execution_result(self, *args, **kwargs) -> None:
        pass

    async def error(self, *args, **kwargs) -> None:
        pass


def create_emitter(
    workflow_id: str,
    task_id: str,
    manager=None,
) -> WorkflowEmitter:
    """Factory function for creating appropriate emitter."""
    if manager is not None:
        return WebSocketEmitter(workflow_id, task_id, manager)
    return NullEmitter()