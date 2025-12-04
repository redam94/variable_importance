"""
Chat Router - Streaming chat with RAG context.

Provides:
- POST /chat/stream - Stream chat with RAG retrieval
- POST /chat/query-rag - Query RAG directly
"""

import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Annotated
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from loguru import logger

from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool

from schemas import (
    ChatRequest,
    ChatChunk,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGChunk,
    ErrorResponse,
)
from dependencies import RAGManager, settings
from auth import (
    get_current_user_optional,
    get_current_active_user,
    User,
)


router = APIRouter(prefix="/chat", tags=["Chat"])


# =============================================================================
# STREAMING CHAT
# =============================================================================


@router.post(
    "/stream",
    summary="Stream chat with RAG",
    description="Stream a chat response with automatic RAG context retrieval.",
)
async def stream_chat(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """
    Stream chat response with RAG context.

    Uses Server-Sent Events (SSE) for real-time token streaming.
    The agent automatically retrieves relevant context from RAG.

    Requires authentication.
    """
    rag = await RAGManager.get_rag(request.workflow_id)

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Initialize LLM with streaming
            llm = ChatOllama(
                model=request.model or settings.DEFAULT_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                streaming=True,
            )

            # Build message history
            messages = []
            if request.history:
                for msg in request.history:
                    if msg.role == "user":
                        messages.append(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        messages.append(AIMessage(content=msg.content))

            # Retrieve RAG context
            rag_context = ""
            if rag and rag.enabled:
                try:
                    # Emit context retrieval event
                    yield f"data: {json.dumps({'type': 'status', 'content': 'Retrieving context...'})}\n\n"

                    rag_context = rag.get_context_summary(
                        query=request.message,
                        workflow_id=request.workflow_id,
                        max_tokens=2000,
                    )

                    if rag_context:
                        yield f"data: {json.dumps({'type': 'context', 'content': f'Found {len(rag_context)} chars of context'})}\n\n"

                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}")
                    yield f"data: {json.dumps({'type': 'warning', 'content': 'Context retrieval failed'})}\n\n"

            # Build system message with context
            system_content = (
                "You are a helpful data science assistant. "
                "Answer questions based on the provided context and your knowledge. "
                "Be accurate, cite sources when available, and provide detailed explanations."
            )

            if rag_context:
                system_content += f"\n\nRelevant Context:\n{rag_context}"

            messages.insert(0, SystemMessage(content=system_content))
            messages.append(HumanMessage(content=request.message))

            # Stream response
            yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

            full_response = ""
            async for chunk in llm.astream(messages):
                token = chunk.content
                if token:
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Store the interaction in RAG for future reference
            if rag and rag.enabled and full_response:
                try:
                    rag.add_summary(
                        summary=f"Q: {request.message}\nA: {full_response[:500]}...",
                        stage_name="chat",
                        workflow_id=request.workflow_id,
                        metadata={"timestamp": datetime.now().isoformat()},
                    )
                except Exception as e:
                    logger.warning(f"Failed to store chat in RAG: {e}")

            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# AGENT CHAT (with tool use)
# =============================================================================


@router.post(
    "/agent/stream",
    summary="Stream agent chat with tools",
    description="Stream chat with an agent that can use RAG retrieval and storage tools.",
)
async def stream_agent_chat(request: ChatRequest):
    """
    Stream agent chat with tool capabilities.

    The agent can:
    - retrieve_context: Query RAG for relevant information
    - store_knowledge: Save useful information to RAG
    """
    rag = await RAGManager.get_rag(request.workflow_id)

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            llm = ChatOllama(
                model=request.model or settings.DEFAULT_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                streaming=True,
            )

            # Define tools
            @tool
            def retrieve_context(query: str) -> str:
                """Retrieve relevant context from the knowledge base for the given query."""
                if not rag or not rag.enabled:
                    return "RAG not available"
                logger.info(f"Tool retrieve_context called with query: {query}")
                return rag.get_context_summary(
                    query=query,
                    workflow_id=request.workflow_id,
                    max_tokens=2000,
                )

            @tool
            def store_knowledge(document: str) -> str:
                """Store useful information in the knowledge base for future reference."""
                if not rag or not rag.enabled:
                    return "RAG not available"
                logger.info(f"Tool store_knowledge called")
                rag.add_summary(
                    summary=document,
                    stage_name="chat",
                    workflow_id=request.workflow_id,
                    metadata={"timestamp": datetime.now().isoformat()},
                )
                return "Knowledge stored successfully"

            # Create agent
            try:
                from langchain.agents import create_agent
                from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware
                
                agent = create_agent(
                    model=llm,
                    tools=[retrieve_context, store_knowledge],
                    middleware=[TodoListMiddleware(), SummarizationMiddleware(model=llm, messages_to_keep=20)],
                )
                logger.info(f"Using LangGraph agent for chat streaming have {len(request.history or [])} history messages")
                # Build messages
                messages = [
                    SystemMessage(
                        content=(
                            "You are a helpful data science assistant that has access to tools for retrieving and storing knowledge. "
                            "Create a plan for querying the knowledge base to answer the user's questions effectively. "
                            "Use the tools to assist with the user's queries effectively. "
                            "Be accurate, cite sources when available, and provide detailed explanations. "
                            "Make sure to use the 'retrieve_context' tool to get relevant information from the knowledge base before answering. "
                            "Even if the user does not explicitly ask for it. " 
                            "Try calling the 'retrieve_context' tool multiple times before answering with different queries to gather relevant information. "
                            "Assess the information you find from the 'retrieve_context' tool if you need to gather more context call the tool again. "
                            "The information you find may also help you decide what to query next. "
                            "Use the 'store_knowledge' tool to save information the user provides; this will help you in future interactions."
                        )
                    )
                    ]
                if request.history:
                    for msg in request.history:
                        if msg.role == "user":
                            messages.append(HumanMessage(content=msg.content))
                        elif msg.role == "assistant":
                            messages.append(AIMessage(content=msg.content))

                messages.append(HumanMessage(content=request.message))

                yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

                # Stream agent response
                async for event in agent.astream(
                    {"messages": messages},
                    stream_mode="messages",
                ):
                    message, metadata = event

                    # Handle tool calls
                    if metadata.get("langgraph_node") == "tools":
                        tool_name = getattr(message, "name", "tool")
                        yield f"data: {json.dumps({'type': 'tool', 'content': f'Using {tool_name}...'})}\n\n"

                    # Handle model output
                    elif metadata.get("langgraph_node") == "model":
                        if hasattr(message, "content_blocks") and message.content_blocks and message.content_blocks[-1]['type'] == "text":
                            yield f"data: {json.dumps({'type': 'token', 'content': message.content_blocks[-1]['text']})}\n\n"

                yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

            except ImportError:
                # Fallback to simple streaming if langgraph agent not available
                logger.warning("LangGraph agent not available, using simple streaming")

                # Get context first
                context = ""
                if rag and rag.enabled:
                    context = rag.get_context_summary(
                        query=request.message,
                        workflow_id=request.workflow_id,
                        max_tokens=2000,
                    )

                system = SystemMessage(
                    content=f"""You are a helpful assistant.
                
Context from knowledge base:
{context if context else 'No relevant context found.'}

Answer the user's question based on the context and your knowledge."""
                )

                messages = [system, HumanMessage(content=request.message)]

                yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

                async for chunk in llm.astream(messages):
                    if chunk.content:
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

                yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

        except Exception as e:
            logger.error(f"Agent stream error: {e}")
            import traceback

            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# RAG QUERY
# =============================================================================


@router.post(
    "/query-rag",
    response_model=RAGQueryResponse,
    summary="Query RAG directly",
    description="Search the RAG knowledge base directly without chat.",
)
async def query_rag(request: RAGQueryRequest) -> RAGQueryResponse:
    """Direct RAG query without chat context."""
    workflow_id = request.workflow_id or "default"
    rag = await RAGManager.get_rag(workflow_id)

    if not rag or not rag.enabled:
        raise HTTPException(status_code=503, detail="RAG not available")

    try:
        results = rag.query_relevant_context(
            query=request.query,
            workflow_id=workflow_id,
            doc_types=getattr(request, 'doc_types', None),
            n_results=request.n_results,
        )

        chunks = []
        for r in results:
            relevance = (
                1.0 - r.get("distance", 0.5) if r.get("distance") is not None else 0.5
            )
            chunks.append(
                RAGChunk(
                    content=r["document"],
                    metadata=r["metadata"],
                    relevance_score=relevance,
                )
            )

        return RAGQueryResponse(
            query=request.query,
            results=chunks,
            total_found=len(chunks),
        )

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/rag-stats/{workflow_id}",
    summary="Get RAG statistics",
)
async def get_rag_stats(workflow_id: str) -> dict:
    """Get statistics about the RAG knowledge base."""
    rag = await RAGManager.get_rag(workflow_id)

    if not rag or not rag.enabled:
        return {"enabled": False, "error": "RAG not available"}

    try:
        return rag.get_stats()
    except Exception as e:
        return {"enabled": True, "error": str(e)}
