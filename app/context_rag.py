"""
Enhanced RAG System with Text Chunking

Features:
- Intelligent text chunking for focused context
- Sentence-aware segmentation
- Overlap between chunks to prevent information loss
- Better context retrieval for code generation
"""

import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("⚠️ ChromaDB not available. Install with: pip install chromadb")


class TextChunker:
    """
    Intelligent text chunking for RAG system.
    
    Splits text into smaller, focused segments while preserving context.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(f"📝 TextChunker initialized (chunk_size={chunk_size}, overlap={overlap})")
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller, focused segments.
        
        Args:
            text: Text to chunk
            metadata: Base metadata to attach to all chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        # Small text - no chunking needed
        if len(text) <= self.chunk_size:
            return [{
                "text": text,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "chunk_size": len(text)
                }
            }]
        
        # Split into sentences (preserve sentence boundaries)
        sentences = re.split(r'(?<=[.!?\n])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Start new chunk if adding sentence exceeds size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Keep overlap from previous chunk
                if self.overlap > 0 and len(chunk_text) > self.overlap:
                    overlap_text = chunk_text[-self.overlap:]
                    # Find sentence boundary in overlap
                    last_period = overlap_text.rfind('. ')
                    if last_period != -1:
                        overlap_text = overlap_text[last_period + 2:]
                    
                    current_chunk = [overlap_text]
                    current_size = len(overlap_text)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Create chunk dictionaries with metadata
        total_chunks = len(chunks)
        result = []
        
        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_size": len(chunk_text)
            }
            result.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        logger.debug(f"📝 Chunked text into {total_chunks} segments")
        return result


class ContextRAG:
    """
    Enhanced RAG system with chunking for better context retrieval.
    
    Features:
    - Stores chunks instead of full documents
    - Better semantic search granularity
    - Focused context for LLM queries
    """
    
    def __init__(
        self,
        collection_name: str = "workflow_context",
        persist_directory: str = "cache/rag_db",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the RAG system.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if not CHROMADB_AVAILABLE:
            logger.error("❌ ChromaDB not available. RAG system disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize chunker
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"📚 ContextRAG initialized: {collection_name}")
            logger.info(f"   Stored documents: {self.collection.count()}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize collection: {e}")
            self.enabled = False
    
    def _generate_id(self, content: str, doc_type: str, chunk_idx: int = 0) -> str:
        """Generate a unique ID for a chunk."""
        hash_input = f"{doc_type}:{content[:100]}:{chunk_idx}:{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata for ChromaDB compatibility.
        ChromaDB only accepts: str, int, float, bool, or None values.
        """
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = None
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list):
                if len(value) == 0:
                    cleaned[key] = ""
                elif all(isinstance(v, str) for v in value):
                    cleaned[key] = ", ".join(value)
                else:
                    cleaned[key] = str(value)
            elif isinstance(value, dict):
                import json
                cleaned[key] = json.dumps(value)
            else:
                cleaned[key] = str(value)
        
        return cleaned
    
    def add_plot_analysis(
        self,
        plot_name: str,
        plot_path: str,
        analysis: str,
        stage_name: str,
        workflow_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a plot analysis to the RAG system with chunking.
        
        Args:
            plot_name: Name of the plot file
            plot_path: Path to the plot file
            analysis: Analysis text from vision LLM
            stage_name: Workflow stage name
            workflow_id: Workflow ID
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        try:
            # Prepare base metadata
            base_metadata = {
                "type": "plot_analysis",
                "plot_name": plot_name,
                "plot_path": plot_path,
                "stage_name": stage_name,
                "workflow_id": workflow_id,
                "timestamp": datetime.now().isoformat()
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            # Chunk the analysis
            doc_text = f"Plot: {plot_name}\n\nAnalysis:\n{analysis}"
            chunks = self.chunker.chunk_text(doc_text, base_metadata)
            
            # Add all chunks to collection
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                chunk_text = chunk["text"]
                chunk_metadata = self._clean_metadata(chunk["metadata"])
                chunk_id = self._generate_id(chunk_text, "plot_analysis", chunk["metadata"]["chunk_index"])
                
                documents.append(chunk_text)
                metadatas.append(chunk_metadata)
                ids.append(chunk_id)
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"📊 Added plot analysis to RAG: {plot_name} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"❌ Failed to add plot analysis: {e}")
    
    def add_code_execution(
        self,
        code: str,
        stdout: str,
        stderr: str,
        stage_name: str,
        workflow_id: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add code execution results to RAG with chunking.
        
        Args:
            code: Executed code
            stdout: Standard output
            stderr: Standard error
            stage_name: Workflow stage name
            workflow_id: Workflow ID
            success: Whether execution was successful
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        try:
            # Prepare base metadata
            base_metadata = {
                "type": "code_execution",
                "stage_name": stage_name,
                "workflow_id": workflow_id,
                "success": success,
                "has_error": bool(stderr),
                "timestamp": datetime.now().isoformat(),
                "code_length": len(code),
                "output_length": len(stdout)
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            # Create document text
            doc_text = f"""
Stage: {stage_name}
Status: {'Success' if success else 'Failed'}

Code:
{code}

Output:
{stdout}
"""
            
            if stderr:
                doc_text += f"\nErrors:\n{stderr}"
            
            # Chunk the document
            chunks = self.chunker.chunk_text(doc_text, base_metadata)
            
            # Add all chunks
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                chunk_text = chunk["text"]
                chunk_metadata = self._clean_metadata(chunk["metadata"])
                chunk_id = self._generate_id(chunk_text, "code_execution", chunk["metadata"]["chunk_index"])
                
                documents.append(chunk_text)
                metadatas.append(chunk_metadata)
                ids.append(chunk_id)
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"💻 Added code execution to RAG: {stage_name} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"❌ Failed to add code execution: {e}")
    
    def add_summary(
        self,
        summary: str,
        stage_name: str,
        workflow_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a summary to RAG with chunking."""
        if not self.enabled:
            return
        
        try:
            base_metadata = {
                "type": "summary",
                "stage_name": stage_name,
                "workflow_id": workflow_id,
                "timestamp": datetime.now().isoformat()
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            doc_text = f"Summary for {stage_name}:\n{summary}"
            chunks = self.chunker.chunk_text(doc_text, base_metadata)
            
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                chunk_text = chunk["text"]
                chunk_metadata = self._clean_metadata(chunk["metadata"])
                chunk_id = self._generate_id(chunk_text, "summary", chunk["metadata"]["chunk_index"])
                
                documents.append(chunk_text)
                metadatas.append(chunk_metadata)
                ids.append(chunk_id)
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"📝 Added summary to RAG: {stage_name} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"❌ Failed to add summary: {e}")
    
    def query_relevant_context(
        self,
        query: str,
        workflow_id: Optional[str] = None,
        stage_name: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query for relevant context chunks.
        
        Args:
            query: User query text
            workflow_id: Filter by workflow ID
            stage_name: Filter by stage name
            doc_types: Filter by document types
            n_results: Number of results to retrieve
            
        Returns:
            List of relevant context chunks with metadata
        """
        if not self.enabled:
            return []
        
        try:
            # Build where filter
            where_filter = None
            
            if workflow_id and doc_types:
                where_filter = {
                    "$and": [
                        {"workflow_id": workflow_id},
                        {"type": {"$in": doc_types}}
                    ]
                }
            elif workflow_id and stage_name:
                where_filter = {
                    "$and": [
                        {"workflow_id": workflow_id},
                        {"stage_name": stage_name}
                    ]
                }
            elif workflow_id:
                where_filter = {"workflow_id": workflow_id}
            elif stage_name:
                where_filter = {"stage_name": stage_name}
            elif doc_types:
                where_filter = {"type": {"$in": doc_types}}
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            if not results['documents'] or not results['documents'][0]:
                logger.info(f"🔍 No relevant context found for query")
                return []
            
            # Format results
            contexts = []
            for i, doc in enumerate(results['documents'][0]):
                context = {
                    "document": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                }
                contexts.append(context)
            
            logger.info(f"🔍 Retrieved {len(contexts)} relevant context chunks")
            return contexts
            
        except Exception as e:
            logger.error(f"❌ Failed to query context: {e}")
            return []
    
    def get_context_summary(
        self,
        query: str,
        workflow_id: Optional[str] = None,
        stage_name: Optional[str] = None,
        max_tokens: int = 2000
    ) -> str:
        """
        Get a concise context summary relevant to the query.
        
        Combines relevant chunks into a focused context string.
        """
        if not self.enabled:
            return ""
        
        contexts = self.query_relevant_context(
            query=query,
            workflow_id=workflow_id,
            stage_name=stage_name,
            n_results=15  # Get more chunks since they're smaller
        )
        
        if not contexts:
            return ""
        
        # Build context string
        context_parts = []
        total_length = 0
        max_chars = max_tokens * 4  # Rough approximation
        
        # Group chunks by document
        doc_groups = {}
        for ctx in contexts:
            doc_type = ctx['metadata'].get('type', 'unknown')
            stage = ctx['metadata'].get('stage_name', 'unknown')
            chunk_idx = ctx['metadata'].get('chunk_index', 0)
            
            key = f"{doc_type}:{stage}"
            if key not in doc_groups:
                doc_groups[key] = []
            doc_groups[key].append((chunk_idx, ctx))
        
        # Add grouped chunks to context
        for key, chunks in doc_groups.items():
            # Sort chunks by index
            chunks.sort(key=lambda x: x[0])
            
            doc_type, stage = key.split(':', 1)
            context_parts.append(f"\n[{doc_type.upper()} - {stage}]")
            
            for _, ctx in chunks:
                doc_text = ctx['document']
                
                if total_length + len(doc_text) > max_chars:
                    remaining = max_chars - total_length
                    if remaining > 100:
                        doc_text = doc_text[:remaining] + "..."
                    else:
                        break
                
                context_parts.append(doc_text)
                total_length += len(doc_text)
                
                if total_length >= max_chars:
                    break
            
            if total_length >= max_chars:
                break
        
        return "\n".join(context_parts)
    
    def delete_by_workflow(self, workflow_id: str):
        """Delete all documents for a specific workflow."""
        if not self.enabled:
            return
        
        try:
            results = self.collection.get(
                where={"workflow_id": workflow_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"🗑️ Deleted {len(results['ids'])} chunks for workflow: {workflow_id}")
        
        except Exception as e:
            logger.error(f"❌ Failed to delete workflow documents: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            total_count = self.collection.count()
            
            # Get type breakdown
            results = self.collection.get()
            type_counts = {}
            if results['metadatas']:
                for metadata in results['metadatas']:
                    doc_type = metadata.get('type', 'unknown')
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            return {
                "enabled": True,
                "total_chunks": total_count,
                "type_breakdown": type_counts,
                "persist_directory": str(self.persist_directory),
                "collection_name": self.collection.name,
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.overlap
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    def clear_all(self):
        """Clear all documents from the RAG system."""
        if not self.enabled:
            return
        
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("🗑️ Cleared all RAG chunks")
        
        except Exception as e:
            logger.error(f"❌ Failed to clear RAG: {e}")


# ============================================================================
# TESTING
# ============================================================================

def test_chunking():
    """Test text chunking functionality."""
    print("\n" + "="*70)
    print("TESTING TEXT CHUNKING")
    print("="*70)
    
    chunker = TextChunker(chunk_size=200, overlap=50)
    
    # Test text
    text = """
    This is the first sentence of our test document. This is the second sentence.
    This is the third sentence that contains important information about data analysis.
    The fourth sentence explains more about machine learning models.
    Here's a fifth sentence about feature engineering. The sixth sentence discusses hyperparameters.
    Finally, the seventh sentence wraps up our discussion about model evaluation metrics.
    """
    
    chunks = chunker.chunk_text(text.strip(), metadata={"source": "test"})
    
    print(f"\nOriginal length: {len(text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Size: {chunk['metadata']['chunk_size']} characters")
        print(f"Text: {chunk['text'][:100]}...")
    
    print("\n✅ Chunking test complete!")


if __name__ == "__main__":
    test_chunking()