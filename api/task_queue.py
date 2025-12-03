"""
Background Task Queue using arq (async Redis queue).

Runs workflows in a SEPARATE WORKER PROCESS, not blocking FastAPI.

Architecture:
- FastAPI enqueues jobs to Redis via arq
- Separate worker process (python -m worker) executes jobs
- Status/progress stored in Redis
- WebSocket updates via Redis pub/sub

Usage:
    # Start worker (separate terminal):
    arq worker.WorkerSettings

    # Or with custom settings:
    python -m worker
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import redis.asyncio as redis
from arq import create_pool, ArqRedis
from arq.connections import RedisSettings
from loguru import logger


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"      # In arq queue, waiting for worker
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CANCELLING = "cancelling"


@dataclass
class TaskInfo:
    """Task metadata stored in Redis."""
    task_id: str
    workflow_id: str
    status: TaskStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: int = 0
    current_node: Optional[str] = None
    error: Optional[str] = None
    result_summary: Optional[str] = None
    username: str = ""
    arq_job_id: Optional[str] = None
    
    # Request params (needed by worker)
    query: str = ""
    stage_name: str = "main"
    data_path: Optional[str] = None
    web_search_enabled: bool = False
    rag_enabled: bool = True
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "TaskInfo":
        data["status"] = TaskStatus(data["status"])
        return cls(**data)


class TaskQueue:
    """
    Redis + arq task queue for background workflow execution.
    
    Jobs run in a separate worker process, not blocking FastAPI.
    """
    
    TASK_KEY = "task:{task_id}"
    CANCEL_KEY = "task:{task_id}:cancel"
    WORKFLOW_TASKS_KEY = "workflow:{workflow_id}:tasks"
    PROGRESS_CHANNEL = "task:{task_id}:progress"
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        task_ttl: int = 86400 * 7,  # 7 days
    ):
        self.redis_url = redis_url
        self.task_ttl = task_ttl
        self._redis: Optional[redis.Redis] = None
        self._arq_pool: Optional[ArqRedis] = None
    
    @property
    def redis_settings(self) -> RedisSettings:
        """Parse redis URL into arq RedisSettings."""
        # Parse redis://host:port/db
        url = self.redis_url.replace("redis://", "")
        if "/" in url:
            host_port, db = url.rsplit("/", 1)
            db = int(db)
        else:
            host_port = url
            db = 0
        
        if ":" in host_port:
            host, port = host_port.split(":")
            port = int(port)
        else:
            host = host_port
            port = 6379
        
        return RedisSettings(host=host, port=port, database=db)
    
    async def connect(self) -> None:
        """Initialize Redis and arq connections."""
        if self._redis is None:
            self._redis = redis.from_url(
                self.redis_url,
                decode_responses=True,
            )
            await self._redis.ping()
            logger.info(f"✅ Task queue connected to Redis")
        
        if self._arq_pool is None:
            self._arq_pool = await create_pool(self.redis_settings)
            logger.info("✅ arq pool created")
    
    async def disconnect(self) -> None:
        """Close connections."""
        if self._arq_pool:
            await self._arq_pool.close()
            self._arq_pool = None
        if self._redis:
            await self._redis.close()
            self._redis = None
    
    @property
    def redis(self) -> redis.Redis:
        if self._redis is None:
            raise RuntimeError("TaskQueue not connected")
        return self._redis
    
    @property
    def arq(self) -> ArqRedis:
        if self._arq_pool is None:
            raise RuntimeError("arq pool not connected")
        return self._arq_pool
    
    # -------------------------------------------------------------------------
    # Task Creation & Enqueueing
    # -------------------------------------------------------------------------
    
    async def enqueue_workflow(
        self,
        task_id: str,
        workflow_id: str,
        query: str,
        username: str = "",
        stage_name: str = "main",
        data_path: Optional[str] = None,
        web_search_enabled: bool = False,
        rag_enabled: bool = True,
    ) -> TaskInfo:
        """
        Create task and enqueue for worker processing.
        
        Returns immediately - worker picks up job asynchronously.
        """
        task = TaskInfo(
            task_id=task_id,
            workflow_id=workflow_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now().isoformat(),
            username=username,
            query=query,
            stage_name=stage_name,
            data_path=data_path,
            web_search_enabled=web_search_enabled,
            rag_enabled=rag_enabled,
        )
        
        # Save to Redis
        await self._save_task(task)
        
        # Enqueue job for worker
        job = await self.arq.enqueue_job(
            "execute_workflow",  # Function name in worker
            task_id=task_id,
        )
        
        # Update with job ID
        task.arq_job_id = job.job_id
        task.status = TaskStatus.QUEUED
        await self._save_task(task)
        
        logger.info(f"📋 Enqueued workflow task: {task_id} (job: {job.job_id})")
        return task
    
    async def _save_task(self, task: TaskInfo) -> None:
        """Save task to Redis."""
        key = self.TASK_KEY.format(task_id=task.task_id)
        await self.redis.set(key, json.dumps(task.to_dict()), ex=self.task_ttl)
        
        # Track under workflow
        wf_key = self.WORKFLOW_TASKS_KEY.format(workflow_id=task.workflow_id)
        await self.redis.sadd(wf_key, task.task_id)
        await self.redis.expire(wf_key, self.task_ttl)
    
    # -------------------------------------------------------------------------
    # Task Status (called by FastAPI)
    # -------------------------------------------------------------------------
    
    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task info."""
        key = self.TASK_KEY.format(task_id=task_id)
        data = await self.redis.get(key)
        if data:
            return TaskInfo.from_dict(json.loads(data))
        return None
    
    async def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        current_node: Optional[str] = None,
        error: Optional[str] = None,
        result_summary: Optional[str] = None,
    ) -> Optional[TaskInfo]:
        """Update task fields."""
        task = await self.get_task(task_id)
        if not task:
            return None
        
        if status:
            task.status = status
            if status == TaskStatus.RUNNING and not task.started_at:
                task.started_at = datetime.now().isoformat()
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                task.completed_at = datetime.now().isoformat()
        
        if progress is not None:
            task.progress = progress
        if current_node is not None:
            task.current_node = current_node
        if error is not None:
            task.error = error
        if result_summary is not None:
            task.result_summary = result_summary
        
        await self._save_task(task)
        return task
    
    # -------------------------------------------------------------------------
    # Cancellation
    # -------------------------------------------------------------------------
    
    async def request_cancel(self, task_id: str) -> bool:
        """Request task cancellation."""
        task = await self.get_task(task_id)
        if not task:
            return False
        
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False
        
        # Set cancel flag (worker checks this)
        cancel_key = self.CANCEL_KEY.format(task_id=task_id)
        await self.redis.set(cancel_key, "1", ex=self.task_ttl)
        
        await self.update_task(task_id, status=TaskStatus.CANCELLING)
        
        # Try to abort the arq job if still queued
        if task.arq_job_id and task.status == TaskStatus.QUEUED:
            try:
                job = await self.arq.job(task.arq_job_id)
                if job:
                    await job.abort()
            except Exception as e:
                logger.warning(f"Could not abort arq job: {e}")
        
        logger.info(f"🛑 Cancel requested: {task_id}")
        return True
    
    async def is_cancelled(self, task_id: str) -> bool:
        """Check if cancellation requested (called by worker)."""
        cancel_key = self.CANCEL_KEY.format(task_id=task_id)
        return await self.redis.exists(cancel_key) > 0
    
    async def acknowledge_cancel(self, task_id: str) -> None:
        """Mark task as cancelled (called by worker after cleanup)."""
        cancel_key = self.CANCEL_KEY.format(task_id=task_id)
        await self.redis.delete(cancel_key)
        await self.update_task(task_id, status=TaskStatus.CANCELLED)
    
    # -------------------------------------------------------------------------
    # Progress Publishing (called by worker)
    # -------------------------------------------------------------------------
    
    async def publish_progress(
        self,
        task_id: str,
        event_type: str,
        stage: str,
        message: str,
        data: Optional[dict] = None,
    ) -> None:
        """
        Publish progress event to Redis pub/sub.
        
        FastAPI subscribes to forward to WebSocket clients.
        """
        channel = self.PROGRESS_CHANNEL.format(task_id=task_id)
        payload = {
            "type": event_type,
            "task_id": task_id,
            "stage": stage,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        if data:
            payload["data"] = data
        
        await self.redis.publish(channel, json.dumps(payload))
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    async def get_workflow_tasks(self, workflow_id: str) -> list[TaskInfo]:
        """Get all tasks for a workflow."""
        wf_key = self.WORKFLOW_TASKS_KEY.format(workflow_id=workflow_id)
        task_ids = await self.redis.smembers(wf_key)
        
        tasks = []
        for task_id in task_ids:
            task = await self.get_task(task_id)
            if task:
                tasks.append(task)
        
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    async def get_active_tasks(self) -> list[TaskInfo]:
        """Get all running/pending/queued tasks."""
        tasks = []
        async for key in self.redis.scan_iter("task:*"):
            if ":" in key.split("task:")[1]:  # Skip sub-keys like :cancel
                continue
            data = await self.redis.get(key)
            if data:
                task = TaskInfo.from_dict(json.loads(data))
                if task.status in (TaskStatus.PENDING, TaskStatus.QUEUED, 
                                   TaskStatus.RUNNING, TaskStatus.CANCELLING):
                    tasks.append(task)
        return tasks


# -----------------------------------------------------------------------------
# Global Instance
# -----------------------------------------------------------------------------

_task_queue: Optional[TaskQueue] = None


async def get_task_queue() -> TaskQueue:
    """Get or create global task queue."""
    global _task_queue
    if _task_queue is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _task_queue = TaskQueue(redis_url=redis_url)
        await _task_queue.connect()
    return _task_queue


async def shutdown_task_queue() -> None:
    """Shutdown task queue."""
    global _task_queue
    if _task_queue:
        await _task_queue.disconnect()
        _task_queue = None