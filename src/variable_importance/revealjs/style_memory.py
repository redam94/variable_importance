"""
Style Guide Memory

Per-user ChromaDB storage for presentation style preferences.
Supports semantic search to find relevant style rules.
Allows both user additions and agent-learned updates.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from .models import StyleRule, StyleGuide

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class StyleMemory:
    """
    Per-user style guide storage with semantic search.
    
    Features:
    - Store style rules as searchable documents
    - Semantic search for relevant rules
    - User can add/update rules
    - Agent can learn and suggest new rules
    - Export/import style guides
    """
    
    def __init__(
        self,
        user_id: str,
        persist_directory: str = "cache/style_guides",
    ):
        """
        Initialize style memory for a user.
        
        Args:
            user_id: Unique user identifier
            persist_directory: Base directory for all style guide storage
        """
        self.user_id = user_id
        
        if not CHROMADB_AVAILABLE:
            logger.error("❌ ChromaDB not available. Style memory disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Each user gets their own collection
        collection_name = f"style_guide_{self._sanitize_id(user_id)}"
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load or create style guide metadata
        self._metadata_file = self.persist_directory / f"{collection_name}_meta.json"
        self.guide = self._load_metadata()
        
        logger.info(f"📝 StyleMemory initialized for user: {user_id} ({self.collection.count()} rules)")
    
    def _sanitize_id(self, user_id: str) -> str:
        """Sanitize user ID for use as collection name."""
        return hashlib.md5(user_id.encode()).hexdigest()[:16]
    
    def _generate_rule_id(self, rule_text: str, category: str) -> str:
        """Generate unique ID for a rule."""
        hash_input = f"{self.user_id}:{category}:{rule_text[:50]}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _load_metadata(self) -> StyleGuide:
        """Load style guide metadata from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    data = json.load(f)
                return StyleGuide(**data)
            except Exception as e:
                logger.warning(f"Failed to load style guide metadata: {e}")
        
        return StyleGuide(user_id=self.user_id)
    
    def _save_metadata(self):
        """Save style guide metadata to disk."""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump(self.guide.model_dump(mode="json"), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save style guide metadata: {e}")
    
    def add_rule(
        self,
        rule_text: str,
        category: str,
        examples: Optional[List[str]] = None,
        priority: str = "medium",
        source: str = "user",
    ) -> Optional[str]:
        """
        Add a style rule to the memory.
        
        Args:
            rule_text: Natural language description of the rule
            category: Style category (colors, typography, etc.)
            examples: Example applications
            priority: high/medium/low
            source: user/agent/default
            
        Returns:
            Rule ID if successful, None otherwise
        """
        if not self.enabled:
            return None
        
        rule_id = self._generate_rule_id(rule_text, category)
        
        # Create searchable document
        doc_text = f"""Style Rule: {rule_text}
Category: {category}
Priority: {priority}
Examples: {'; '.join(examples or [])}"""
        
        metadata = {
            "rule_id": rule_id,
            "category": category,
            "priority": priority,
            "source": source,
            "examples": "; ".join(examples or []),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        try:
            # Upsert to handle updates
            self.collection.upsert(
                documents=[doc_text],
                metadatas=[metadata],
                ids=[rule_id]
            )
            
            # Update guide metadata
            self.guide.updated_at = datetime.now()
            self._save_metadata()
            
            logger.info(f"📝 Added style rule: {rule_text[:50]}... ({category})")
            return rule_id
            
        except Exception as e:
            logger.error(f"Failed to add rule: {e}")
            return None
    
    def update_rule(
        self,
        rule_id: str,
        rule_text: Optional[str] = None,
        examples: Optional[List[str]] = None,
        priority: Optional[str] = None,
    ) -> bool:
        """Update an existing rule."""
        if not self.enabled:
            return False
        
        try:
            # Get existing rule
            existing = self.collection.get(ids=[rule_id])
            if not existing["documents"]:
                logger.warning(f"Rule not found: {rule_id}")
                return False
            
            old_metadata = existing["metadatas"][0]
            
            # Merge updates
            new_metadata = {
                **old_metadata,
                "updated_at": datetime.now().isoformat(),
            }
            
            if priority:
                new_metadata["priority"] = priority
            if examples:
                new_metadata["examples"] = "; ".join(examples)
            
            # Rebuild document text
            final_rule_text = rule_text or old_metadata.get("rule_text", "")
            doc_text = f"""Style Rule: {final_rule_text}
Category: {new_metadata['category']}
Priority: {new_metadata['priority']}
Examples: {new_metadata.get('examples', '')}"""
            
            self.collection.update(
                documents=[doc_text],
                metadatas=[new_metadata],
                ids=[rule_id]
            )
            
            self.guide.updated_at = datetime.now()
            self._save_metadata()
            
            logger.info(f"📝 Updated style rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update rule: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a style rule."""
        if not self.enabled:
            return False
        
        try:
            self.collection.delete(ids=[rule_id])
            self.guide.updated_at = datetime.now()
            self._save_metadata()
            logger.info(f"🗑️ Removed style rule: {rule_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove rule: {e}")
            return False
    
    def search_rules(
        self,
        query: str,
        category: Optional[str] = None,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant style rules.
        
        Args:
            query: Natural language query
            category: Filter by category
            n_results: Max results
            
        Returns:
            List of matching rules with metadata
        """
        if not self.enabled:
            return []
        
        try:
            where_filter = None
            if category:
                where_filter = {"category": category}
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            rules = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    rules.append({
                        "document": doc,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None
                    })
            
            return rules
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_all_rules(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all style rules, optionally filtered by category."""
        if not self.enabled:
            return []
        
        try:
            if category:
                results = self.collection.get(where={"category": category})
            else:
                results = self.collection.get()
            
            rules = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    rules.append({
                        "document": doc,
                        "metadata": results["metadatas"][i]
                    })
            
            return rules
            
        except Exception as e:
            logger.error(f"Get all rules failed: {e}")
            return []
    
    def get_formatted_context(self, query: Optional[str] = None) -> str:
        """
        Get formatted style rules context for LLM prompts.
        
        Args:
            query: Optional query to find most relevant rules
            
        Returns:
            Formatted string of style rules
        """
        if not self.enabled:
            return "No style guide available."
        
        # Get relevant rules
        if query:
            rules = self.search_rules(query, n_results=10)
        else:
            rules = self.get_all_rules()
        
        if not rules:
            return "No style rules defined yet."
        
        # Format for LLM
        lines = ["## User Style Guide\n"]
        
        # Group by category
        by_category: Dict[str, List] = {}
        for rule in rules:
            cat = rule["metadata"].get("category", "general")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(rule)
        
        for category, cat_rules in by_category.items():
            lines.append(f"\n### {category.title()}")
            for rule in cat_rules:
                priority = rule["metadata"].get("priority", "medium")
                priority_marker = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
                
                # Extract rule text from document
                doc = rule["document"]
                rule_line = doc.split("\n")[0].replace("Style Rule: ", "")
                
                lines.append(f"- {priority_marker} {rule_line}")
                
                examples = rule["metadata"].get("examples", "")
                if examples:
                    lines.append(f"  Examples: {examples}")
        
        return "\n".join(lines)
    
    def set_defaults(
        self,
        theme: Optional[str] = None,
        transition: Optional[str] = None,
        colors: Optional[Dict[str, str]] = None,
        font_heading: Optional[str] = None,
        font_body: Optional[str] = None,
    ):
        """Update default style guide settings."""
        if theme:
            self.guide.default_theme = theme
        if transition:
            self.guide.default_transition = transition
        if colors:
            self.guide.default_colors.update(colors)
        if font_heading:
            self.guide.default_font_heading = font_heading
        if font_body:
            self.guide.default_font_body = font_body
        
        self.guide.updated_at = datetime.now()
        self._save_metadata()
        logger.info("📝 Updated style guide defaults")
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default style settings."""
        return {
            "theme": self.guide.default_theme,
            "transition": self.guide.default_transition,
            "colors": self.guide.default_colors,
            "font_heading": self.guide.default_font_heading,
            "font_body": self.guide.default_font_body,
        }
    
    def export_guide(self) -> Dict[str, Any]:
        """Export complete style guide for backup/sharing."""
        rules = self.get_all_rules()
        
        return {
            "user_id": self.user_id,
            "name": self.guide.name,
            "description": self.guide.description,
            "defaults": self.get_defaults(),
            "rules": rules,
            "exported_at": datetime.now().isoformat(),
        }
    
    def import_guide(self, guide_data: Dict[str, Any], merge: bool = True):
        """
        Import a style guide.
        
        Args:
            guide_data: Exported guide data
            merge: If True, merge with existing; if False, replace
        """
        if not merge:
            # Clear existing rules
            existing = self.collection.get()
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
        
        # Import defaults
        if "defaults" in guide_data:
            self.set_defaults(**guide_data["defaults"])
        
        # Import rules
        for rule in guide_data.get("rules", []):
            metadata = rule.get("metadata", {})
            self.add_rule(
                rule_text=metadata.get("rule_text", rule["document"].split("\n")[0]),
                category=metadata.get("category", "general"),
                examples=metadata.get("examples", "").split("; ") if metadata.get("examples") else None,
                priority=metadata.get("priority", "medium"),
                source=metadata.get("source", "imported"),
            )
        
        logger.info(f"📥 Imported style guide with {len(guide_data.get('rules', []))} rules")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get style memory statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        rules = self.get_all_rules()
        
        # Count by category
        by_category = {}
        by_priority = {"high": 0, "medium": 0, "low": 0}
        by_source = {"user": 0, "agent": 0, "default": 0}
        
        for rule in rules:
            cat = rule["metadata"].get("category", "general")
            by_category[cat] = by_category.get(cat, 0) + 1
            
            priority = rule["metadata"].get("priority", "medium")
            by_priority[priority] = by_priority.get(priority, 0) + 1
            
            source = rule["metadata"].get("source", "user")
            by_source[source] = by_source.get(source, 0) + 1
        
        return {
            "enabled": True,
            "user_id": self.user_id,
            "total_rules": len(rules),
            "by_category": by_category,
            "by_priority": by_priority,
            "by_source": by_source,
            "defaults": self.get_defaults(),
            "created_at": self.guide.created_at.isoformat() if self.guide.created_at else None,
            "updated_at": self.guide.updated_at.isoformat() if self.guide.updated_at else None,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_style_memories: Dict[str, StyleMemory] = {}


def get_style_memory(user_id: str, **kwargs) -> StyleMemory:
    """
    Get or create StyleMemory instance for a user.
    
    Caches instances to avoid re-initialization.
    """
    if user_id not in _style_memories:
        _style_memories[user_id] = StyleMemory(user_id=user_id, **kwargs)
    return _style_memories[user_id]