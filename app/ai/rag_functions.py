from langgraph.runtime import Runtime
from loguru import logger
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .config import DefaultConfig
from .state import State, ExecutionDeps

DEFAULT_CONFIG = DefaultConfig()

def get_context_from_rag(state: State, runtime: Runtime[ExecutionDeps]) -> dict:
    """Check RAG for relevant context to answer user's query."""
    rag = runtime.context.get("rag")
    rag_llm = runtime.context.get("rag_llm", DEFAULT_CONFIG.rag_llm)
    workflow_id = state.get("workflow_id", "default")
    
    user_messages = state.get("messages", [])
    latest_query = user_messages[-1].content if user_messages else ""
    plan = state.get("plan", "")

    logger.info(f"🔍 Checking RAG for existing context...")
    
    rag_context = {
        "has_relevant_context": False,
        "contexts": [],
        "summary": "",
        "confidence": 0.0
    }
    can_answer_from_rag = False
    
    if not rag or not rag.enabled:
        logger.info("❌ RAG not available")
        return {
            "rag_context": rag_context,
            "can_answer_from_rag": can_answer_from_rag
        }
    
    try:
        # Query RAG for relevant context
        contexts = rag.query_relevant_context(
            query=latest_query,
            workflow_id=workflow_id,
            n_results=10
        )
        
        if contexts:
            rag_context["has_relevant_context"] = True
            rag_context["contexts"] = contexts
            
            rag_summary = rag.get_context_summary(
                query=latest_query,
                workflow_id=workflow_id,
                max_tokens=500
            )
            rag_context["summary"] = rag_summary
            logger.info("📝 Generated RAG context summary")
            llm_answer = ChatOllama(
                model=rag_llm,
                temperature=0,
                base_url=DEFAULT_CONFIG.base_url
            )
            logger.debug(f"Using RAG LLM: {rag_llm}")
            class RAGDecision(BaseModel):
                can_answer: bool = Field(
                    description="Whether the RAG context is sufficient to answer the query"
                )
                confidence: float = Field(
                    description="Confidence level (0-1) in this decision",
                    ge=0.0, le=1.0
                )
                reasoning: str = Field(
                    description="Brief explanation of the decision"
                )
            structured_llm = llm_answer.with_structured_output(RAGDecision)
            decision_prompt = f"""
Using the provided rag context, determine if it is sufficient to answer the user's query and complete the provided plan.

Plan:
<plan>
{plan}
</plan>

RAG Context Summary:
<rag_summary>
{rag_summary}
</rag_summary>

Guidelines:
1. If the user asks for specific calculations or analyses not covered by the context you must set, can_answer = False
2. If the context directly addresses the query, can_answer = True
3. If the context is only tangentially related or insufficient, can_answer = False
4. If the user asks for interpretations that can be drawn from the context, can_answer = True
5. If in doubt, err on the side of can_answer = False
6. If steps from the plan can not be completed with the context, can_answer = False

Be decisive in your assessment.

IF USER ASKS FOR ANALYSIS TO BE REPEATED OR NEW CALCULATIONS, SET can_answer = False.
"""
            logger.debug(f"RAG decision prompt: {decision_prompt[:200]}...")
            decision = structured_llm.invoke(
                [SystemMessage(content=decision_prompt),
                 HumanMessage(content=latest_query)]
            )
            rag_context["confidence"] = decision.confidence
            can_answer_from_rag = decision.can_answer

            logger.info(f"✅ Found {len(contexts)} relevant chunks (confidence: {rag_context['confidence']:.2f})")
            logger.info(f"🎯 Can answer from RAG: {can_answer_from_rag} (reasoning: {decision.reasoning})")
    
    except Exception as e:
        logger.error(f"❌ Error querying RAG: {e}")
    
    return {
        "rag_context": rag_context,
        "can_answer_from_rag": can_answer_from_rag
    }


def route_after_rag_check(state: State, runtime: Runtime[ExecutionDeps]) -> str:
    """Route based on RAG assessment."""
    if state.get("can_answer_from_rag", False):
        logger.info("➡️ Routing to summarize (RAG has sufficient context)")
        return "summarize_from_rag"
    else:
        logger.info("➡️ Routing to check existing outputs")
        return "check_existing_outputs"


def summarize_from_rag(state: State, runtime: Runtime[ExecutionDeps]) -> dict:
    """Generate summary directly from RAG context."""
    logger.info("📝 Generating summary from RAG context...")
    rag_summary_llm = runtime.context.get("rag_summary_llm", DEFAULT_CONFIG.rag_summary_llm)
    rag_context = state.get("rag_context", {})
    user_messages = state.get("messages", [])
    latest_query = user_messages[-1].content if user_messages else ""
    plan = state.get("plan", "")
    logger.info(f"Using RAG summary LLM: {rag_summary_llm}")
    llm = ChatOllama(
        model=rag_summary_llm,
        temperature=0,
        base_url=DEFAULT_CONFIG.base_url
    )
    
    context = rag_context.get("summary", "")

    contexts = rag_context.get("contexts", [])

    context = "Context Summary from RAG:\n" + context + "\n\n"
    if contexts:
        for i, ctx in enumerate(contexts, 1):
            context = (
                context 
                + f"[Stage {ctx.get('metadata', {}).get('stage_name', 'unknown')}, Type: {ctx.get('metadata', {}).get('type', '')}, Rating: {1-ctx.get('distance', ''):0.5f}" 
                + (f", plot name: {ctx.get('metadata', {}).get('plot_name', '')}]" if ctx.get('metadata', {}).get('plot_name', '') else "]")
                +f":\n\n{ctx.get('document', '')}\n\n" 
            )
    logger.debug(f"RAG contexts used for summary: {context}")

    system_prompt = f"""
    You are an expert data scientist, leveraging previous analysis context to answer user queries.
    Provide a clear, accurate, and concise response based solely on the provided context.
    You are communicating with a layperson, so avoid jargon and explain concepts simply.
    Suggest next steps or analyses if relevant. You were provided with the following plan to accomplish the user's goals:
    
    Plan:
    <plan>
    {plan}
    </plan>

    Context:
    <context>
    {context}
    </context>
    """

    logger.debug(f"System prompt for RAG summary: {system_prompt[:200]}...")
    response = llm.invoke([
        SystemMessage(
            content=system_prompt
        ),
        HumanMessage(content=f"Query: {latest_query}")
    ])
    
    logger.info("✅ Generated summary from RAG context")
    
    return {"summary": response.content}