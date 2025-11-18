"""
RAG System for Context Management

Uses vector database (ChromaDB) to store and retrieve relevant context.
Reduces token usage by retrieving only relevant information instead of 
sending all outputs to the LLM.
"""

import hashlib
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


class ContextRAG:
    """
    RAG system for managing workflow context.
    
    Features:
    - Stores plot analyses, code outputs, and execution results
    - Retrieves only relevant context based on user query
    - Reduces token usage by filtering irrelevant information
    - Maintains semantic search capabilities
    """
    
    def __init__(
        self,
        collection_name: str = "workflow_context",
        persist_directory: str = "cache/rag_db"
    ):
        """
        Initialize the RAG system.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        if not CHROMADB_AVAILABLE:
            logger.error("❌ ChromaDB not available. RAG system disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
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
    
    def _generate_id(self, content: str, doc_type: str) -> str:
        """Generate a unique ID for a document."""
        hash_input = f"{doc_type}:{content}:{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
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
        Add a plot analysis to the RAG system.
        
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
            # Create document text
            doc_text = f"Plot: {plot_name}\nAnalysis: {analysis}"
            
            # Prepare metadata
            doc_metadata = {
                "type": "plot_analysis",
                "plot_name": plot_name,
                "plot_path": plot_path,
                "stage_name": stage_name,
                "workflow_id": workflow_id,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Generate unique ID
            doc_id = self._generate_id(doc_text, "plot_analysis")
            
            # Add to collection
            self.collection.add(
                documents=[doc_text],
                metadatas=[doc_metadata],
                ids=[doc_id]
            )
            
            logger.info(f"📊 Added plot analysis to RAG: {plot_name}")
            
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
        Add code execution results to the RAG system.
        
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
            # Create document text (truncate if too long)
            code_snippet = code[:500] + "..." if len(code) > 500 else code
            stdout_snippet = stdout[:500] + "..." if len(stdout) > 500 else stdout
            
            doc_text = f"""
Stage: {stage_name}
Code: {code_snippet}
Output: {stdout_snippet}
Status: {'Success' if success else 'Failed'}
"""
            
            # Prepare metadata
            doc_metadata = {
                "type": "code_execution",
                "stage_name": stage_name,
                "workflow_id": workflow_id,
                "success": success,
                "has_error": bool(stderr),
                "timestamp": datetime.now().isoformat(),
                "code_length": len(code),
                "output_length": len(stdout),
                **(metadata or {})
            }
            
            # Generate unique ID
            doc_id = self._generate_id(doc_text, "code_execution")
            
            # Add to collection
            self.collection.add(
                documents=[doc_text],
                metadatas=[doc_metadata],
                ids=[doc_id]
            )
            
            logger.info(f"💻 Added code execution to RAG: {stage_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to add code execution: {e}")
    
    def add_summary(
        self,
        summary: str,
        stage_name: str,
        workflow_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a summary to the RAG system.
        
        Args:
            summary: Summary text
            stage_name: Workflow stage name
            workflow_id: Workflow ID
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        try:
            doc_text = f"Summary for {stage_name}:\n{summary}"
            
            doc_metadata = {
                "type": "summary",
                "stage_name": stage_name,
                "workflow_id": workflow_id,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            doc_id = self._generate_id(doc_text, "summary")
            
            self.collection.add(
                documents=[doc_text],
                metadatas=[doc_metadata],
                ids=[doc_id]
            )
            
            logger.info(f"📝 Added summary to RAG: {stage_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to add summary: {e}")
    
    def query_relevant_context(
        self,
        query: str,
        workflow_id: Optional[str] = None,
        stage_name: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query for relevant context based on user query.
        
        Args:
            query: User query text
            workflow_id: Filter by workflow ID
            stage_name: Filter by stage name
            doc_types: Filter by document types (e.g., ["plot_analysis", "summary"])
            n_results: Number of results to retrieve
            
        Returns:
            List of relevant context documents with metadata
        """
        if not self.enabled:
            return []
        
        try:
            # Build where filter
            where_filter = {}
            if workflow_id:
                where_filter["workflow_id"] = workflow_id
            if stage_name:
                where_filter["stage_name"] = stage_name
            if doc_types:
                where_filter["type"] = {"$in": doc_types}
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter if where_filter else None
            )
            
            # Format results
            if not results['documents'] or not results['documents'][0]:
                logger.info(f"🔍 No relevant context found for query")
                return []
            
            contexts = []
            for i, doc in enumerate(results['documents'][0]):
                context = {
                    "document": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                }
                contexts.append(context)
            
            logger.info(f"🔍 Retrieved {len(contexts)} relevant contexts")
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
        
        Args:
            query: User query
            workflow_id: Filter by workflow ID
            stage_name: Filter by stage name
            max_tokens: Maximum approximate tokens in summary
            
        Returns:
            Formatted context string
        """
        if not self.enabled:
            return ""
        
        contexts = self.query_relevant_context(
            query=query,
            workflow_id=workflow_id,
            stage_name=stage_name,
            n_results=10
        )
        
        if not contexts:
            return ""
        
        # Build context string
        context_parts = []
        total_length = 0
        max_chars = max_tokens * 4  # Rough approximation
        
        for ctx in contexts:
            doc_type = ctx['metadata'].get('type', 'unknown')
            doc_text = ctx['document']
            
            # Truncate if needed
            if total_length + len(doc_text) > max_chars:
                remaining = max_chars - total_length
                if remaining > 100:  # Only add if meaningful
                    doc_text = doc_text[:remaining] + "..."
                else:
                    break
            
            context_parts.append(f"[{doc_type.upper()}]")
            context_parts.append(doc_text)
            context_parts.append("")  # Blank line
            
            total_length += len(doc_text)
        
        return "\n".join(context_parts)
    
    def delete_by_workflow(self, workflow_id: str):
        """Delete all documents for a specific workflow."""
        if not self.enabled:
            return
        
        try:
            # Get all IDs for this workflow
            results = self.collection.get(
                where={"workflow_id": workflow_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"🗑️ Deleted {len(results['ids'])} documents for workflow: {workflow_id}")
        
        except Exception as e:
            logger.error(f"❌ Failed to delete workflow documents: {e}")
    
    def delete_by_stage(self, stage_name: str, workflow_id: Optional[str] = None):
        """Delete all documents for a specific stage."""
        if not self.enabled:
            return
        
        try:
            where_filter = {"stage_name": stage_name}
            if workflow_id:
                where_filter["workflow_id"] = workflow_id
            
            results = self.collection.get(where=where_filter)
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"🗑️ Deleted {len(results['ids'])} documents for stage: {stage_name}")
        
        except Exception as e:
            logger.error(f"❌ Failed to delete stage documents: {e}")
    
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
                "total_documents": total_count,
                "type_breakdown": type_counts,
                "persist_directory": str(self.persist_directory),
                "collection_name": self.collection.name
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    def clear_all(self):
        """Clear all documents from the RAG system."""
        if not self.enabled:
            return
        
        try:
            # Delete collection and recreate
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("🗑️ Cleared all RAG documents")
        
        except Exception as e:
            logger.error(f"❌ Failed to clear RAG: {e}")


# ============================================================================
# STANDALONE TESTING
# ============================================================================

def test_rag():
    """Test the RAG system."""
    import tempfile
    import shutil
    
    print("\n" + "="*70)
    print("TESTING RAG SYSTEM")
    print("="*70)
    
    if not CHROMADB_AVAILABLE:
        print("❌ ChromaDB not available. Skipping tests.")
        return
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Initialize RAG
        rag = ContextRAG(
            collection_name="test_workflow",
            persist_directory=str(temp_dir / "rag_db")
        )
        
        workflow_id = "test_workflow_001"
        
        # Test 1: Add plot analysis
        print("\n1. Testing add plot analysis...")
        rag.add_plot_analysis(
            plot_name="scatter_plot.png",
            plot_path="/path/to/scatter_plot.png",
            analysis="This scatter plot shows a strong positive correlation between variables X and Y.",
            stage_name="eda",
            workflow_id=workflow_id
        )
        print("✅ Plot analysis added")
        
        # Test 2: Add code execution
        print("\n2. Testing add code execution...")
        rag.add_code_execution(
            code="import pandas as pd\ndf = pd.read_csv('data.csv')",
            stdout="Loaded 1000 rows, 5 columns",
            stderr="",
            stage_name="data_loading",
            workflow_id=workflow_id,
            success=True
        )
        print("✅ Code execution added")
        
        # Test 3: Add summary
        print("\n3. Testing add summary...")
        rag.add_summary(
            summary="The data shows strong seasonality with peaks in summer months.",
            stage_name="eda",
            workflow_id=workflow_id
        )
        print("✅ Summary added")
        
        # Test 4: Query relevant context
        print("\n4. Testing query...")
        contexts = rag.query_relevant_context(
            query="What does the scatter plot show?",
            workflow_id=workflow_id,
            n_results=3
        )
        assert len(contexts) > 0, "Should find relevant contexts"
        print(f"   Found {len(contexts)} relevant contexts")
        print(f"   Top result: {contexts[0]['document'][:100]}...")
        print("✅ Query works")
        
        # Test 5: Get context summary
        print("\n5. Testing context summary...")
        summary = rag.get_context_summary(
            query="correlation between variables",
            workflow_id=workflow_id
        )
        assert len(summary) > 0, "Should generate summary"
        print(f"   Summary length: {len(summary)} chars")
        print("✅ Context summary works")
        
        # Test 6: Get statistics
        print("\n6. Testing statistics...")
        stats = rag.get_stats()
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Type breakdown: {stats['type_breakdown']}")
        assert stats['total_documents'] == 3
        print("✅ Statistics work")
        
        # Test 7: Persistence
        print("\n7. Testing persistence...")
        rag2 = ContextRAG(
            collection_name="test_workflow",
            persist_directory=str(temp_dir / "rag_db")
        )
        stats2 = rag2.get_stats()
        assert stats2['total_documents'] == 3, "Should persist across instances"
        print("✅ Persistence works")
        
        # Test 8: Delete by workflow
        print("\n8. Testing deletion...")
        rag.delete_by_workflow(workflow_id)
        stats3 = rag.get_stats()
        assert stats3['total_documents'] == 0
        print("✅ Deletion works")
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✅")
        print("="*70)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_rag()