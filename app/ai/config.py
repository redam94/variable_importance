
from dataclasses import dataclass

@dataclass
class DefaultConfig:
    base_url: str = "http://100.91.155.118:11434"
    ## LLM Models for RAG and Code Generation
    rag_llm: str = "qwen3:30b"
    rag_summary_llm: str = "qwen3:30b"

    ## LLM for code generation and fixing
    code_llm: str = "qwen3-coder:30b"