"""
Redis Pub/Sub to WebSocket Bridge.

Subscribes to worker progress events and forwards to WebSocket clients.

Usage:
    # In WebSocket endpoint:
    subscriber = TaskSubscriber(task_id, websocket)
    await subscriber.listen()
"""

import asyncio
import json
import os
from typing import Optional

import redis.asyncio as redis
from fastapi import WebSocket
from loguru import logger


class TaskSubscriber:
    """
    Subscribe to task progress events and forward to WebSocket.
    
    The worker publishes events to Redis pub/sub.
    This class subscribes and forwards to the connected client.
    """
    
    PROGRESS_CHANNEL = "task:{task_id}:progress"
    
    def __init__(
        self,
        task_id: str,
        websocket: WebSocket,
        redis_url: str = None,
    ):
        self.task_id = task_id
        self.websocket = websocket
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._running = False
    
    async def connect(self) -> None:
        """Connect to Redis."""
        self._redis = redis.from_url(self.redis_url, decode_responses=True)
        self._pubsub = self._redis.pubsub()
        channel = self.PROGRESS_CHANNEL.format(task_id=self.task_id)
        await self._pubsub.subscribe(channel)
        logger.debug(f"Subscribed to {channel}")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        self._running = False
        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
    
    async def listen(self) -> None:
        """
        Listen for events and forward to WebSocket.
        
        Runs until task completes or WebSocket disconnects.
        """
        await self.connect()
        self._running = True
        
        try:
            while self._running:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                
                if message and message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        await self.websocket.send_json(data)
                        
                        # Stop on terminal events
                        if data.get("type") in ("done", "error", "cancelled"):
                            self._running = False
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from pub/sub: {message['data']}")
                
                # Small delay to prevent busy loop
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Subscriber error: {e}")
        finally:
            await self.disconnect()
    
    def stop(self) -> None:
        """Stop listening."""
        self._running = False


class MultiTaskSubscriber:
    """
    Subscribe to multiple tasks at once.
    
    Useful for workflow-level subscriptions where multiple
    tasks may be running.
    """
    
    WORKFLOW_PATTERN = "task:*:progress"
    
    def __init__(
        self,
        workflow_id: str,
        websocket: WebSocket,
        redis_url: str = None,
    ):
        self.workflow_id = workflow_id
        self.websocket = websocket
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._running = False
        self._task_ids: set[str] = set()
    
    async def add_task(self, task_id: str) -> None:
        """Subscribe to a specific task."""
        if task_id in self._task_ids:
            return
        
        self._task_ids.add(task_id)
        if self._pubsub:
            channel = f"task:{task_id}:progress"
            await self._pubsub.subscribe(channel)
            logger.debug(f"Added subscription: {channel}")
    
    async def remove_task(self, task_id: str) -> None:
        """Unsubscribe from a task."""
        self._task_ids.discard(task_id)
        if self._pubsub:
            channel = f"task:{task_id}:progress"
            await self._pubsub.unsubscribe(channel)
    
    async def connect(self) -> None:
        """Connect to Redis and subscribe to known tasks."""
        self._redis = redis.from_url(self.redis_url, decode_responses=True)
        self._pubsub = self._redis.pubsub()
        
        for task_id in self._task_ids:
            channel = f"task:{task_id}:progress"
            await self._pubsub.subscribe(channel)
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        self._running = False
        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
    
    async def listen(self) -> None:
        """Listen and forward events."""
        await self.connect()
        self._running = True
        
        try:
            while self._running:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                
                if message and message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        await self.websocket.send_json(data)
                    except json.JSONDecodeError:
                        pass
                
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Multi-subscriber error: {e}")
        finally:
            await self.disconnect()
    
    def stop(self) -> None:
        """Stop listening."""
        self._running = False