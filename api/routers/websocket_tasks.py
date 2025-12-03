"""
Task WebSocket Router - Real-time task progress via Redis pub/sub.

The worker publishes progress to Redis.
This endpoint subscribes and forwards to connected clients.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from loguru import logger

from auth import get_user_from_token
from pubsub import TaskSubscriber
from task_queue import get_task_queue


router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/task/{task_id}")
async def task_progress_websocket(
    websocket: WebSocket,
    task_id: str,
    token: Optional[str] = Query(None),
):
    """
    WebSocket for real-time task progress.
    
    Subscribes to Redis pub/sub and forwards events from the worker.
    
    Events:
    - stage_start: Node starting
    - stage_end: Node completed
    - rag_query: RAG search iteration
    - rag_search_end: RAG search complete
    - code_generated: Code was generated
    - execution_result: Code execution result
    - done: Workflow completed
    - error: Error occurred
    - progress: General progress update
    
    Example:
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/task/task_abc?token=...');
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === 'done') {
        console.log('Complete:', data.data.summary);
      }
    };
    ```
    """
    # Authenticate
    user = None
    if token:
        user = await get_user_from_token(token)
    
    await websocket.accept()
    
    # Verify task exists
    task_queue = await get_task_queue()
    task = await task_queue.get_task(task_id)
    
    if not task:
        await websocket.send_json({
            "type": "error",
            "message": f"Task not found: {task_id}",
        })
        await websocket.close()
        return
    
    # Send initial status
    await websocket.send_json({
        "type": "connected",
        "task_id": task_id,
        "workflow_id": task.workflow_id,
        "status": task.status.value,
        "progress": task.progress,
    })
    
    # If already complete, send result and close
    if task.status.value in ("completed", "failed", "cancelled"):
        await websocket.send_json({
            "type": "done",
            "task_id": task_id,
            "status": task.status.value,
            "result_summary": task.result_summary,
            "error": task.error,
        })
        await websocket.close()
        return
    
    # Subscribe to progress updates
    subscriber = TaskSubscriber(task_id, websocket)
    
    try:
        # Run subscriber and handle incoming messages concurrently
        listen_task = asyncio.create_task(subscriber.listen())
        
        # Handle client messages (ping/pong, cancel requests)
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    import json
                    msg = json.loads(data)
                    
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    
                    elif msg.get("type") == "cancel":
                        # Handle cancel request
                        await task_queue.request_cancel(task_id)
                        await websocket.send_json({
                            "type": "cancelling",
                            "message": "Cancel requested",
                        })
                        
                except json.JSONDecodeError:
                    pass
                    
        except WebSocketDisconnect:
            pass
        finally:
            subscriber.stop()
            listen_task.cancel()
            
    except Exception as e:
        logger.error(f"Task WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })


@router.websocket("/ws/workflow/{workflow_id}/tasks")
async def workflow_tasks_websocket(
    websocket: WebSocket,
    workflow_id: str,
    token: Optional[str] = Query(None),
):
    """
    WebSocket for all tasks in a workflow.
    
    Automatically subscribes to new tasks as they're created.
    """
    user = None
    if token:
        user = await get_user_from_token(token)
    
    await websocket.accept()
    
    task_queue = await get_task_queue()
    
    # Get current tasks
    tasks = await task_queue.get_workflow_tasks(workflow_id)
    active_task_ids = {t.task_id for t in tasks if t.status.value in ("pending", "queued", "running")}
    
    await websocket.send_json({
        "type": "connected",
        "workflow_id": workflow_id,
        "active_tasks": list(active_task_ids),
    })
    
    from pubsub import MultiTaskSubscriber
    subscriber = MultiTaskSubscriber(workflow_id, websocket)
    
    # Subscribe to active tasks
    for task_id in active_task_ids:
        await subscriber.add_task(task_id)
    
    try:
        listen_task = asyncio.create_task(subscriber.listen())
        
        # Periodically check for new tasks
        while True:
            await asyncio.sleep(2)
            
            current_tasks = await task_queue.get_workflow_tasks(workflow_id)
            current_ids = {t.task_id for t in current_tasks if t.status.value in ("pending", "queued", "running")}
            
            # Subscribe to new tasks
            for task_id in current_ids - active_task_ids:
                await subscriber.add_task(task_id)
                await websocket.send_json({
                    "type": "task_added",
                    "task_id": task_id,
                })
            
            active_task_ids = current_ids
            
    except WebSocketDisconnect:
        pass
    finally:
        subscriber.stop()
        listen_task.cancel()