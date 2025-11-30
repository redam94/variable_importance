"""
WebSocket Router - Real-time communication.

Provides:
- /ws/workflow/{workflow_id} - Real-time workflow progress
- /ws/chat/{workflow_id} - Streaming chat via WebSocket
- Connection management with authentication
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Set, Any
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
from loguru import logger

from auth import get_user_from_token, User


router = APIRouter(tags=["WebSocket"])


# =============================================================================
# CONNECTION MANAGER
# =============================================================================

@dataclass
class Connection:
    """Represents a WebSocket connection."""
    websocket: WebSocket
    user: Optional[User]
    workflow_id: str
    connected_at: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash((self.websocket, self.user.username if self.user else None, self.workflow_id))



class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.
    
    Supports:
    - Per-workflow connection groups
    - Authenticated and anonymous connections
    - Broadcast to specific workflows
    """
    
    def __init__(self):
        # workflow_id -> set of Connection objects
        self._connections: Dict[str, Set[Connection]] = {}
        self._lock = asyncio.Lock()
    
    async def connect(
        self,
        websocket: WebSocket,
        workflow_id: str,
        user: Optional[User] = None
    ) -> Connection:
        """Accept and register a new connection."""
        await websocket.accept()
        
        conn = Connection(
            websocket=websocket,
            user=user,
            workflow_id=workflow_id,
        )
        
        async with self._lock:
            if workflow_id not in self._connections:
                self._connections[workflow_id] = set()
            self._connections[workflow_id].add(conn)
        
        username = user.username if user else "anonymous"
        logger.info(f"🔌 WebSocket connected: {username} -> {workflow_id}")
        
        return conn
    
    async def disconnect(self, conn: Connection):
        """Remove a connection."""
        async with self._lock:
            workflow_id = conn.workflow_id
            if workflow_id in self._connections:
                self._connections[workflow_id].discard(conn)
                if not self._connections[workflow_id]:
                    del self._connections[workflow_id]
        
        username = conn.user.username if conn.user else "anonymous"
        logger.info(f"🔌 WebSocket disconnected: {username} <- {conn.workflow_id}")
    
    async def send_to_workflow(self, workflow_id: str, message: dict):
        """Send message to all connections in a workflow."""
        async with self._lock:
            connections = self._connections.get(workflow_id, set()).copy()
        
        if not connections:
            return
        
        data = json.dumps(message)
        
        # Send to all connections
        disconnected = []
        for conn in connections:
            try:
                await conn.websocket.send_text(data)
            except Exception:
                disconnected.append(conn)
        
        # Clean up disconnected
        for conn in disconnected:
            await self.disconnect(conn)
    
    async def send_to_user(self, username: str, message: dict):
        """Send message to all connections for a specific user."""
        data = json.dumps(message)
        disconnected = []
        
        async with self._lock:
            all_connections = [
                conn 
                for conns in self._connections.values() 
                for conn in conns
                if conn.user and conn.user.username == username
            ]
        
        for conn in all_connections:
            try:
                await conn.websocket.send_text(data)
            except Exception:
                disconnected.append(conn)
        
        for conn in disconnected:
            await self.disconnect(conn)
    
    async def broadcast(self, message: dict):
        """Send message to all connections."""
        data = json.dumps(message)
        disconnected = []
        
        async with self._lock:
            all_connections = [
                conn 
                for conns in self._connections.values() 
                for conn in conns
            ]
        
        for conn in all_connections:
            try:
                await conn.websocket.send_text(data)
            except Exception:
                disconnected.append(conn)
        
        for conn in disconnected:
            await self.disconnect(conn)
    
    def get_connection_count(self, workflow_id: Optional[str] = None) -> int:
        """Get number of active connections."""
        if workflow_id:
            return len(self._connections.get(workflow_id, set()))
        return sum(len(conns) for conns in self._connections.values())


# Global connection manager
manager = ConnectionManager()


# =============================================================================
# PROGRESS EMITTER (for workflow integration)
# =============================================================================

class WebSocketProgressEmitter:
    """
    Progress emitter that sends updates via WebSocket.
    
    Drop-in replacement for the Streamlit progress emitter.
    """
    
    def __init__(self, workflow_id: str, task_id: str):
        self.workflow_id = workflow_id
        self.task_id = task_id
        self._current_stage: Optional[str] = None
    
    async def emit(self, event_type: str, stage: str, message: str, data: Optional[dict] = None):
        """Emit a progress event."""
        payload = {
            "type": "progress",
            "task_id": self.task_id,
            "event": event_type,
            "stage": stage,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        if data:
            payload["data"] = data
        
        await manager.send_to_workflow(self.workflow_id, payload)
    
    async def stage_start(self, stage: str, description: str = ""):
        """Signal stage start."""
        self._current_stage = stage
        await self.emit("stage_start", stage, description)
    
    async def stage_end(self, stage: str, success: bool = True):
        """Signal stage completion."""
        status = "completed" if success else "failed"
        await self.emit("stage_end", stage, status)
        self._current_stage = None
    
    async def rag_query(self, stage: str, result_count: int):
        """RAG query event."""
        await self.emit("rag_query", stage, f"Found {result_count} results")
    
    async def code_generated(self, stage: str, line_count: int):
        """Code generation event."""
        await self.emit("code_generated", stage, f"Generated {line_count} lines")
    
    async def execution_result(self, stage: str, success: bool, output: str = ""):
        """Code execution result."""
        status = "success" if success else "error"
        await self.emit("execution", stage, status, {"output": output[:500]})


def get_ws_emitter(workflow_id: str, task_id: str) -> WebSocketProgressEmitter:
    """Factory function for creating WebSocket emitters."""
    return WebSocketProgressEmitter(workflow_id, task_id)


# =============================================================================
# WEBSOCKET ENDPOINTS
# =============================================================================

@router.websocket("/ws/workflow/{workflow_id}")
async def workflow_websocket(
    websocket: WebSocket,
    workflow_id: str,
    token: Optional[str] = Query(None, description="JWT token or API key"),
):
    """
    WebSocket for real-time workflow progress updates.
    
    Connect to receive:
    - Stage start/end events
    - Progress messages
    - Code generation updates
    - Execution results
    - Error notifications
    
    Authentication:
    - Pass `token` query parameter with JWT or API key
    - Anonymous connections allowed but may have limited access
    
    Message format:
    ```json
    {
        "type": "progress",
        "task_id": "task_123",
        "event": "stage_start",
        "stage": "gather_context",
        "message": "Gathering context...",
        "timestamp": "2024-01-15T10:30:00Z"
    }
    ```
    """
    # Authenticate
    user = None
    if token:
        user = await get_user_from_token(token)
    
    # Connect
    conn = await manager.connect(websocket, workflow_id, user)
    
    # Send connection confirmation
    await websocket.send_json({
        "type": "connected",
        "workflow_id": workflow_id,
        "authenticated": user is not None,
        "username": user.username if user else None,
    })
    
    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                # Handle ping/pong for keepalive
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
                # Handle subscription changes
                elif message.get("type") == "subscribe":
                    new_workflow = message.get("workflow_id")
                    if new_workflow and new_workflow != workflow_id:
                        # Move to new workflow
                        await manager.disconnect(conn)
                        workflow_id = new_workflow
                        conn = await manager.connect(websocket, workflow_id, user)
                        await websocket.send_json({
                            "type": "subscribed",
                            "workflow_id": workflow_id,
                        })
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                })
                
    except WebSocketDisconnect:
        await manager.disconnect(conn)


@router.websocket("/ws/chat/{workflow_id}")
async def chat_websocket(
    websocket: WebSocket,
    workflow_id: str,
    token: Optional[str] = Query(None),
):
    """
    WebSocket for streaming chat.
    
    Send messages:
    ```json
    {
        "type": "message",
        "content": "What patterns are in the data?",
        "model": "qwen3:30b"
    }
    ```
    
    Receive:
    ```json
    {"type": "token", "content": "The"}
    {"type": "token", "content": " data"}
    {"type": "done"}
    ```
    """
    from langchain_ollama import ChatOllama
    from langchain.messages import HumanMessage, AIMessage, SystemMessage
    from api.dependencies import RAGManager, settings
    
    # Authenticate
    user = None
    if token:
        user = await get_user_from_token(token)
    
    # Connect
    conn = await manager.connect(websocket, f"chat_{workflow_id}", user)
    
    await websocket.send_json({
        "type": "connected",
        "workflow_id": workflow_id,
        "authenticated": user is not None,
    })
    
    # Message history for this session
    history: list = []
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue
                
                if message.get("type") == "clear":
                    history.clear()
                    await websocket.send_json({"type": "cleared"})
                    continue
                
                if message.get("type") != "message":
                    continue
                
                content = message.get("content", "")
                model = message.get("model", settings.DEFAULT_MODEL)
                
                if not content:
                    continue
                
                # Get RAG context
                rag = await RAGManager.get_rag(workflow_id)
                rag_context = ""
                
                if rag and rag.enabled:
                    await websocket.send_json({
                        "type": "status",
                        "content": "Retrieving context...",
                    })
                    
                    try:
                        rag_context = rag.get_context_summary(
                            query=content,
                            workflow_id=workflow_id,
                            max_tokens=2000,
                        )
                    except Exception as e:
                        logger.warning(f"RAG retrieval failed: {e}")
                
                # Build messages
                system_content = (
                    "You are a helpful data science assistant. "
                    "Answer based on the provided context and your knowledge."
                )
                if rag_context:
                    system_content += f"\n\nContext:\n{rag_context}"
                
                messages = [SystemMessage(content=system_content)]
                messages.extend(history)
                messages.append(HumanMessage(content=content))
                
                # Stream response
                llm = ChatOllama(
                    model=model,
                    base_url=settings.OLLAMA_BASE_URL,
                    streaming=True,
                )
                
                await websocket.send_json({"type": "start"})
                
                full_response = ""
                async for chunk in llm.astream(messages):
                    if chunk.content:
                        full_response += chunk.content
                        await websocket.send_json({
                            "type": "token",
                            "content": chunk.content,
                        })
                
                await websocket.send_json({"type": "done"})
                
                # Update history
                history.append(HumanMessage(content=content))
                history.append(AIMessage(content=full_response))
                
                # Trim history if too long
                if len(history) > 20:
                    history = history[-20:]
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                })
            except Exception as e:
                logger.error(f"Chat error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
                
    except WebSocketDisconnect:
        await manager.disconnect(conn)


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@router.get("/ws/connections", tags=["System"])
async def get_connections() -> dict:
    """Get WebSocket connection statistics."""
    return {
        "total_connections": manager.get_connection_count(),
        "workflows": {
            wf: len(conns) 
            for wf, conns in manager._connections.items()
        },
    }