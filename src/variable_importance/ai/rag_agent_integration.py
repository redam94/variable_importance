"""
RAG Agent Integration for gather_context node.

This module provides the updated gather_context implementation
that uses the RAG search agent instead of simple queries.

Replace the RAG section in nodes_v2.py gather_context with this implementation.
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from .rag_agent import RAGSearchAgent, RAGAgentConfig, search_rag_with_agent
from .state import State, Deps, Context, DEFAULTS


async def gather_rag_context(
    query: str,
    rag,
    workflow_id: str,
    deps: Deps,
    emit_fn: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Gather RAG context using the intelligent search agent.
    
    Args:
        query: User query
        rag: ContextRAG instance
        workflow_id: Current workflow ID
        deps: Runtime dependencies
        emit_fn: Progress emission function
        
    Returns:
        Dict with 'rag_context', 'doc_context', 'search_result'
    """
    if not rag or not rag.enabled:
        return {"rag_context": "", "doc_context": "", "search_result": None}
    
    # Configure agent
    config = RAGAgentConfig(
        max_iterations=10,
        relevance_threshold=0.6,
        relevance_weight=0.6,
        recency_weight=0.4,
        max_chunks_per_query=10,
        max_total_chunks=20,
        max_context_chars=3000,
        recency_decay_days=30,
    )
    
    llm_model = deps.get("llm", DEFAULTS["llm"])
    base_url = deps.get("base_url", DEFAULTS["base_url"])
    
    def on_progress(msg: str):
        if emit_fn:
            emit_fn(f"📚 {msg}")
        logger.debug(f"RAG Agent: {msg}")
    
    try:
        if emit_fn:
            emit_fn("📚 Starting intelligent RAG search...")
        
        result = await search_rag_with_agent(
            query=query,
            rag=rag,
            task_context="Data science analysis task",
            workflow_id=workflow_id,
            llm_model=llm_model,
            base_url=base_url,
            config=config,
            on_progress=on_progress,
        )
        
        rag_context = result.context if result.accepted else ""
        
        if emit_fn:
            status = "✅" if result.accepted else "⚠️"
            emit_fn(
                f"{status} RAG: {result.total_chunks_found} chunks, "
                f"relevance {result.final_relevance:.2f}, "
                f"{result.iterations} iterations"
            )
        
        logger.info(
            f"📚 RAG Agent: {result.total_chunks_found} chunks found, "
            f"relevance={result.final_relevance:.2f}, "
            f"accepted={result.accepted}"
        )
        
        # Also query documents separately for focused retrieval
        doc_context = await _query_documents(query, rag, workflow_id, emit_fn)
        
        return {
            "rag_context": rag_context,
            "doc_context": doc_context,
            "search_result": result,
        }
        
    except Exception as e:
        logger.warning(f"RAG agent search failed: {e}")
        # Fallback to simple search
        return await _fallback_rag_search(query, rag, workflow_id, emit_fn)


async def _query_documents(
    query: str,
    rag,
    workflow_id: str,
    emit_fn: Optional[callable],
) -> str:
    """Query uploaded documents separately."""
    try:
        doc_results = rag.query_documents(
            query=query,
            workflow_id=workflow_id,
            n_results=5,
        )
        
        if not doc_results:
            return ""
        
        doc_texts = []
        for doc in doc_results:
            title = doc["metadata"].get("title", "Document")
            content = doc["document"][:500]
            doc_texts.append(f"[{title}]: {content}")
        
        if emit_fn:
            emit_fn(f"📄 Found {len(doc_results)} document chunks")
        
        return "\n\n".join(doc_texts)
        
    except Exception as e:
        logger.warning(f"Document query failed: {e}")
        return ""


async def _fallback_rag_search(
    query: str,
    rag,
    workflow_id: str,
    emit_fn: Optional[callable],
) -> Dict[str, Any]:
    """Fallback to simple RAG search if agent fails."""
    try:
        if emit_fn:
            emit_fn("📚 Using fallback RAG search...")
        
        rag_summary = rag.get_context_summary(
            query=query,
            workflow_id=workflow_id,
            max_tokens=1500,
        )
        
        if emit_fn and rag_summary:
            emit_fn(f"📚 Fallback found {len(rag_summary)} chars")
        
        return {
            "rag_context": rag_summary or "",
            "doc_context": "",
            "search_result": None,
        }
        
    except Exception as e:
        logger.error(f"Fallback RAG search failed: {e}")
        return {"rag_context": "", "doc_context": "", "search_result": None}


# =============================================================================
# UPDATED GATHER_CONTEXT NODE
# =============================================================================

async def gather_context_with_agent(state: State, runtime) -> dict:
    """
    Gather all context using intelligent RAG agent.
    
    This is a drop-in replacement for the gather_context node.
    """
    deps = runtime.context
    emitter = deps.get("progress_emitter")
    
    if emitter:
        emitter.stage_start("gather_context", "Gathering context with RAG agent")
    
    # Extract query
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""
    workflow_id = state.get("workflow_id", "default")
    stage_name = state.get("stage_name", "analysis")
    
    context = Context(rag="", web="", outputs="", plots=[], combined="")
    parts = []
    
    def emit(msg: str):
        if emitter:
            from utils.progress_events import EventType
            emitter.emit(EventType.PROGRESS, "gather_context", msg)
    
    # 1. RAG Context with Agent
    rag = deps.get("rag")
    rag_result = await gather_rag_context(
        query=query,
        rag=rag,
        workflow_id=workflow_id,
        deps=deps,
        emit_fn=emit,
    )
    
    if rag_result["rag_context"]:
        context["rag"] = rag_result["rag_context"]
        parts.append(f"[Previous Analysis]\n{rag_result['rag_context']}")
        
        if emitter:
            emitter.rag_query("gather_context", len(rag_result["rag_context"].split("\n")))
    
    if rag_result["doc_context"]:
        parts.append(f"[Reference Documents]\n{rag_result['doc_context']}")
    
    # 2. Web Search (unchanged from original)
    if state.get("web_search_enabled"):
        try:
            from .web_search import search_and_synthesize
            
            emit("🌐 Searching web...")
            
            llm_model = deps.get("llm", DEFAULTS["llm"])
            base_url = deps.get("base_url", DEFAULTS["base_url"])
            
            search_result = await search_and_synthesize(
                query=query,
                context=context.get("rag", ""),
                llm_model=llm_model,
                base_url=base_url,
                on_progress=lambda msg: emit(f"🌐 {msg}"),
            )
            
            if search_result["results"]:
                context["web"] = search_result["formatted_text"]
                parts.append(f"[Web Research]\n{search_result['formatted_text']}")
                
                # Store in RAG
                if rag and rag.enabled:
                    web_results = [
                        {
                            "url": r.url,
                            "title": r.title,
                            "content": r.content,
                            "score": r.score,
                            "source": r.source,
                            "enriched": "crawl4ai" in r.source.lower(),
                            "query_used": r.query_used,
                        }
                        for r in search_result["results"]
                    ]
                    rag.add_web_search_batch(
                        query=query,
                        results=web_results,
                        stage_name=stage_name,
                        workflow_id=workflow_id,
                    )
                
                logger.info(f"🌐 Web: {len(search_result['results'])} results")
                
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
    
    # 3. Existing Outputs (unchanged)
    output_manager = deps.get("output_manager")
    if output_manager:
        try:
            emit("📁 Checking existing outputs...")
            
            existing = output_manager.get_all_outputs()
            if existing:
                output_parts = []
                for stage, data in list(existing.items())[-3:]:
                    if "stdout" in data and data["stdout"]:
                        output_parts.append(f"[{stage}]\n{data['stdout'][:500]}")
                
                if output_parts:
                    outputs_text = "\n\n".join(output_parts)
                    context["outputs"] = outputs_text
                    parts.append(f"[Existing Outputs]\n{outputs_text}")
                    
                    emit(f"📁 Found {len(output_parts)} existing outputs")
                    
        except Exception as e:
            logger.warning(f"Output retrieval failed: {e}")
    
    # 4. Plot paths (unchanged)
    plot_cache = deps.get("plot_cache")
    if plot_cache:
        try:
            plots = plot_cache.get_recent_plots(workflow_id, limit=5)
            context["plots"] = [str(p) for p in plots]
            
            if plots:
                emit(f"📊 Found {len(plots)} plots")
                
        except Exception as e:
            logger.warning(f"Plot cache failed: {e}")
    
    # Combine context
    context["combined"] = "\n\n---\n\n".join(parts) if parts else ""
    
    if emitter:
        emitter.stage_end("gather_context", success=True)
    
    logger.info(f"📦 Context assembled: {len(context['combined'])} chars")
    
    return {"context": context}