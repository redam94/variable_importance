"""
Session Persistence System

Saves and restores Streamlit session state across page refreshes.
Uses file-based storage to maintain chat history, settings, and state.

IMPORTANT: Excludes widget keys that cannot be set via st.session_state
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
from loguru import logger
import hashlib


class SessionPersistence:
    """
    Manages persistence of Streamlit session state.
    
    Features:
    - Save/load session state to/from disk
    - Automatic session identification
    - Selective persistence (only save what's needed)
    - Handles complex objects (pickles them separately)
    - Excludes widget keys that can't be restored
    """
    
    def __init__(self, cache_dir: str = "cache/sessions"):
        """
        Initialize session persistence.
        
        Args:
            cache_dir: Directory to store session files
        """
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"💾 SessionPersistence initialized: {self.cache_dir}")
        
        # Widget keys that cannot be set via st.session_state
        self.widget_keys = {
            "file_uploader",
            "_file_uploader", 
            "model_selector",
            "workflow_id_input",
            "stage_name_input",
            
            # Add any other widget keys here
        }
    
    def _generate_session_id(self, custom_id: Optional[str] = None) -> str:
        """
        Generate a session ID.
        
        For Streamlit, we'll use a consistent ID per browser session.
        If custom_id is provided, use that. Otherwise, use a default.
        """
        if custom_id:
            return hashlib.md5(custom_id.encode()).hexdigest()[:16]
        # Use a default session ID - in production, you might want to use
        # Streamlit's session_id or generate based on user info
        return "default_session"
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get the path to the session file."""
        return self.cache_dir / f"session_{session_id}.json"
    
    def _get_pickle_file(self, session_id: str) -> Path:
        """Get the path to the pickle file for complex objects."""
        return self.cache_dir / f"session_{session_id}_objects.pkl"
    
    def save_session(
        self,
        session_state: Dict[str, Any],
        session_id: Optional[str] = None,
        exclude_keys: Optional[list] = None
    ) -> bool:
        """
        Save session state to disk.
        
        Args:
            session_state: Dictionary of session state to save
            session_id: Optional session ID
            exclude_keys: Keys to exclude from saving
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_id = self._generate_session_id(session_id)
            session_file = self._get_session_file(session_id)
            pickle_file = self._get_pickle_file(session_id)
            
            # Default exclusions (including widget keys)
            if exclude_keys is None:
                exclude_keys = []
            
            # Add widget keys to exclusions
            exclude_keys.extend(self.widget_keys)
            
            # Also exclude any key that starts with underscore (internal Streamlit keys)
            exclude_keys.extend([k for k in session_state.keys() if k.startswith('_')])
            
            # Separate serializable and non-serializable data
            json_data = {}
            pickle_data = {}
            
            for key, value in session_state.items():
                if key in exclude_keys:
                    continue
                if 'del_' in key.lower():
                    continue
                if 'load_' in key.lower():
                    continue
                
                # Try to JSON serialize
                try:
                    json.dumps(value)
                    json_data[key] = value
                except (TypeError, ValueError):
                    # If not JSON serializable, pickle it
                    pickle_data[key] = value
            
            # Add metadata
            json_data["_metadata"] = {
                "saved_at": datetime.now().isoformat(),
                "session_id": session_id
            }
            
            # Save JSON data
            with open(session_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            # Save pickled data if any
            if pickle_data:
                with open(pickle_file, 'wb') as f:
                    pickle.dump(pickle_data, f)
            
            logger.info(f"💾 Saved session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save session: {e}")
            return False
    
    def load_session(
        self,
        session_id: Optional[str] = None,
        max_age_hours: Optional[int] = 24
    ) -> Dict[str, Any]:
        """
        Load session state from disk.
        
        Args:
            session_id: Optional session ID
            max_age_hours: Maximum age of session in hours (None = no limit)
            
        Returns:
            Dictionary of session state, or empty dict if not found
        """
        try:
            session_id = self._generate_session_id(session_id)
            session_file = self._get_session_file(session_id)
            pickle_file = self._get_pickle_file(session_id)
            
            if not session_file.exists():
                logger.info(f"📭 No saved session found: {session_id}")
                return {}
            
            # Load JSON data
            with open(session_file, 'r') as f:
                json_data = json.load(f)
            
            # Check age
            if max_age_hours and "_metadata" in json_data:
                saved_at = datetime.fromisoformat(json_data["_metadata"]["saved_at"])
                age_hours = (datetime.now() - saved_at).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    logger.warning(f"⏰ Session too old ({age_hours:.1f}h), ignoring")
                    return {}
            
            # Remove metadata
            json_data.pop("_metadata", None)
            
            # Load pickled data if exists
            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    pickle_data = pickle.load(f)
                json_data.update(pickle_data)
            
            logger.info(f"📥 Loaded session: {session_id} ({len(json_data)} keys)")
            return json_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load session: {e}")
            return {}
    
    def delete_session(self, session_id: Optional[str] = None) -> bool:
        """
        Delete a saved session.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_id = self._generate_session_id(session_id)
            session_file = self._get_session_file(session_id)
            pickle_file = self._get_pickle_file(session_id)
            
            deleted = False
            if session_file.exists():
                session_file.unlink()
                deleted = True
            
            if pickle_file.exists():
                pickle_file.unlink()
                deleted = True
            
            if deleted:
                logger.info(f"🗑️ Deleted session: {session_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"❌ Failed to delete session: {e}")
            return False
    
    def list_sessions(self) -> list[Dict[str, Any]]:
        """
        List all saved sessions.
        
        Returns:
            List of session info dictionaries
        """
        sessions = []
        
        for session_file in self.cache_dir.glob("session_*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                metadata = data.get("_metadata", {})
                sessions.append({
                    "session_id": metadata.get("session_id", "unknown"),
                    "saved_at": metadata.get("saved_at", "unknown"),
                    "file_path": str(session_file)
                })
            except Exception as e:
                logger.warning(f"⚠️ Failed to read session file {session_file}: {e}")
        
        return sorted(sessions, key=lambda x: x["saved_at"], reverse=True)
    
    def cleanup_old_sessions(self, max_age_hours: int = 168):
        """
        Clean up old session files.
        
        Args:
            max_age_hours: Delete sessions older than this (default: 1 week)
        """
        deleted_count = 0
        
        for session_file in self.cache_dir.glob("session_*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                metadata = data.get("_metadata", {})
                if "saved_at" in metadata:
                    saved_at = datetime.fromisoformat(metadata["saved_at"])
                    age_hours = (datetime.now() - saved_at).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        session_id = metadata.get("session_id")
                        if self.delete_session(session_id):
                            deleted_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Failed to process session file {session_file}: {e}")
        
        if deleted_count > 0:
            logger.info(f"🗑️ Cleaned up {deleted_count} old sessions")


# ============================================================================
# HELPER FUNCTIONS FOR STREAMLIT
# ============================================================================

def save_streamlit_session(st_session_state, session_id: Optional[str] = None):
    """
    Save Streamlit session state.
    
    Args:
        st_session_state: Streamlit session_state object
        session_id: Optional custom session ID
    """
    persistence = SessionPersistence()
    
    # Convert session_state to dict
    state_dict = dict(st_session_state)
    
    # Keys to exclude from persistence (cannot be pickled or should be recreated)
    exclude_keys = [
        "executor",  # Will be recreated (contains thread objects)
        "plot_cache",  # Will be recreated
        "rag",  # Will be recreated (contains ChromaDB client)
        "output_mgr",  # Will be recreated based on workflow_id
        "task_manager",  # Will be recreated (contains threads!) ⭐
    ]
    
    persistence.save_session(state_dict, session_id, exclude_keys)


def load_streamlit_session(st_session_state, session_id: Optional[str] = None):
    """
    Load and restore Streamlit session state.
    
    Args:
        st_session_state: Streamlit session_state object
        session_id: Optional custom session ID
    """
    persistence = SessionPersistence()
    saved_state = persistence.load_session(session_id, max_age_hours=24)
    
    if saved_state:
        # Restore saved state (excluding widget keys)
        for key, value in saved_state.items():
            if key not in st_session_state:
                if key.startswith("_"):
                    continue
                st_session_state[key] = value
        
        logger.info(f"✅ Restored session with {len(saved_state)} items")
        return True
    
    return False


def clear_streamlit_session(session_id: Optional[str] = None):
    """
    Clear saved Streamlit session.
    
    Args:
        session_id: Optional custom session ID
    """
    persistence = SessionPersistence()
    persistence.delete_session(session_id)


# ============================================================================
# STANDALONE TESTING
# ============================================================================

def test_persistence():
    """Test the session persistence system."""
    import tempfile
    import shutil
    
    print("\n" + "="*70)
    print("TESTING SESSION PERSISTENCE")
    print("="*70)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        persistence = SessionPersistence(cache_dir=str(temp_dir))
        
        # Test 1: Save session
        print("\n1. Testing save session...")
        test_state = {
            "messages": ["Hello", "World"],
            "workflow_id": "test_workflow",
            "counter": 42,
            "settings": {"model": "test-model", "temperature": 0.7},
            "file_uploader": "SHOULD BE EXCLUDED",  # Widget key
            "_internal": "SHOULD BE EXCLUDED"  # Internal key
        }
        
        success = persistence.save_session(test_state, session_id="test_001")
        assert success, "Should save successfully"
        print("✅ Save works")
        
        # Test 2: Load session
        print("\n2. Testing load session...")
        loaded_state = persistence.load_session(session_id="test_001")
        assert loaded_state["workflow_id"] == "test_workflow"
        assert loaded_state["counter"] == 42
        assert "file_uploader" not in loaded_state, "Widget keys should be excluded"
        assert "_internal" not in loaded_state, "Internal keys should be excluded"
        print("✅ Load works (widget keys excluded)")
        
        # Test 3: List sessions
        print("\n3. Testing list sessions...")
        sessions = persistence.list_sessions()
        assert len(sessions) >= 1
        print(f"   Found {len(sessions)} session(s)")
        print("✅ List works")
        
        # Test 4: Delete session
        print("\n4. Testing delete session...")
        deleted = persistence.delete_session(session_id="test_001")
        assert deleted, "Should delete successfully"
        
        loaded_state = persistence.load_session(session_id="test_001")
        assert len(loaded_state) == 0, "Should be empty after deletion"
        print("✅ Delete works")
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✅")
        print("="*70)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_persistence()