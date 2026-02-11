import json
import logging
import os
import time
import tempfile
import threading
from typing import Dict, Any, List, Optional


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable types safely."""
    
    def default(self, obj):
        """Convert non-serializable objects to serializable formats."""
        if isinstance(obj, (set, frozenset)):
            # Convert sets and frozensets to sorted lists for consistency
            return sorted(list(obj))
        elif hasattr(obj, '__dict__'):
            # Convert objects with __dict__ to dictionary representation
            return obj.__dict__
        elif hasattr(obj, '_asdict'):
            # Handle namedtuples
            return obj._asdict()
        else:
            # Let the default encoder handle it (will raise TypeError for truly non-serializable types)
            return super().default(obj)


class ContextMemory:
    """
    Implementation of the Persistent Context Memory system described in the AI Co-scientist paper.
    
    This system handles sophisticated state tracking, including:
    - Saving the full state of the system
    - Keeping track of state history
    - Retrieving previous states
    - Maintaining agent-specific memory sections
    """
    
    def __init__(self, 
                 base_dir: str = "memory",
                 primary_file: str = "state.json",
                 history_dir: str = "history",
                 max_history: int = 10):
        """
        Initialize the ContextMemory system.
        
        Args:
            base_dir: Base directory for all persistent storage
            primary_file: Main state file within the base_dir
            history_dir: Directory for historical states
            max_history: Maximum number of historical states to keep
        """
        self.base_dir = base_dir
        self.primary_file = os.path.join(base_dir, primary_file)
        self.history_dir = os.path.join(base_dir, history_dir)
        self.max_history = max_history
        
        # File locking for thread safety
        self._file_lock = threading.Lock()
        
        # Ensure directories exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        
        # Initialize empty state if it doesn't exist
        if not os.path.exists(self.primary_file):
            self._write_json(self.primary_file, {
                "meta": {
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "version": "1.0"
                },
                "state": {},
                "agent_memory": {}
            })
    
    def load_state(self) -> Dict[str, Any]:
        """Load the current state."""
        try:
            data = self._read_json(self.primary_file)
            state = data.get("state", {})
            logging.info("Loaded state from %s", self.primary_file)
            return state
        except Exception as e:
            logging.error("Error loading state: %s", e)
            return {}
    
    def save_state(self, state: Dict[str, Any]) -> bool:
        """
        Save the current state and archive the previous one.
        
        Args:
            state: The current state to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First archive the current state
            self._archive_current_state()
            
            # Then update the current state
            current_data = self._read_json(self.primary_file)
            current_data["state"] = state
            current_data["meta"]["updated_at"] = time.time()
            
            # Write the updated state
            self._write_json(self.primary_file, current_data)
            logging.info("Saved state to %s", self.primary_file)
            return True
        except Exception as e:
            logging.error("Error saving state: %s", e)
            return False
    
    def save_agent_memory(self, agent_type: str, memory_data: Dict[str, Any]) -> bool:
        """
        Save agent-specific memory data.
        
        Args:
            agent_type: Type of agent (e.g., 'generation', 'reflection')
            memory_data: Memory data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            current_data = self._read_json(self.primary_file)
            
            if "agent_memory" not in current_data:
                current_data["agent_memory"] = {}
                
            current_data["agent_memory"][agent_type] = memory_data
            current_data["meta"]["updated_at"] = time.time()
            
            self._write_json(self.primary_file, current_data)
            logging.info("Saved %s agent memory", agent_type)
            return True
        except Exception as e:
            logging.error("Error saving agent memory: %s", e)
            return False
    
    def load_agent_memory(self, agent_type: str) -> Dict[str, Any]:
        """
        Load agent-specific memory data.
        
        Args:
            agent_type: Type of agent (e.g., 'generation', 'reflection')
            
        Returns:
            The agent's memory data
        """
        try:
            current_data = self._read_json(self.primary_file)
            agent_memory = current_data.get("agent_memory", {}).get(agent_type, {})
            return agent_memory
        except Exception as e:
            logging.error("Error loading agent memory: %s", e)
            return {}
    
    def get_state_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get a list of historical states, newest first.
        
        Args:
            limit: Maximum number of historical states to return
            
        Returns:
            List of historical states
        """
        try:
            history_files = self._get_history_files()
            
            if limit:
                history_files = history_files[:limit]
                
            history = []
            for history_file in history_files:
                data = self._read_json(os.path.join(self.history_dir, history_file))
                if "state" in data:
                    entry = {
                        "timestamp": data.get("meta", {}).get("updated_at"),
                        "filename": history_file,
                        "state": data["state"]
                    }
                    history.append(entry)
                    
            return history
        except Exception as e:
            logging.error("Error getting state history: %s", e)
            return []
    
    def get_state_at_timestamp(self, timestamp: float) -> Dict[str, Any]:
        """
        Get a state at a specific timestamp.
        
        Args:
            timestamp: Unix timestamp to search for
            
        Returns:
            The state at that timestamp, or empty dict if not found
        """
        try:
            history = self.get_state_history()
            
            for entry in history:
                if abs(entry.get("timestamp", 0) - timestamp) < 1.0:  # Within 1 second
                    return entry.get("state", {})
                    
            return {}
        except Exception as e:
            logging.error("Error getting state at timestamp: %s", e)
            return {}
    
    def _archive_current_state(self) -> bool:
        """Archive the current state to history."""
        try:
            if os.path.exists(self.primary_file):
                data = self._read_json(self.primary_file)
                timestamp = int(time.time())
                history_file = f"state_{timestamp}.json"
                history_path = os.path.join(self.history_dir, history_file)
                
                self._write_json(history_path, data)
                
                # Clean up old history files if needed
                self._cleanup_history()
                
                return True
            return False
        except Exception as e:
            logging.error("Error archiving state: %s", e)
            return False
    
    def _cleanup_history(self) -> None:
        """Clean up old history files if there are too many."""
        try:
            history_files = self._get_history_files()
            
            if len(history_files) > self.max_history:
                # Delete oldest files
                files_to_delete = history_files[self.max_history:]
                
                for file in files_to_delete:
                    os.remove(os.path.join(self.history_dir, file))
        except Exception as e:
            logging.error("Error cleaning up history: %s", e)
    
    def _get_history_files(self) -> List[str]:
        """Get a list of history files, sorted by newest first."""
        try:
            files = [f for f in os.listdir(self.history_dir) if f.startswith("state_") and f.endswith(".json")]
            return sorted(files, reverse=True)
        except Exception as e:
            logging.error("Error getting history files: %s", e)
            return []
    
    def _read_json(self, file_path: str) -> Dict[str, Any]:
        """Read and parse a JSON file with thread safety."""
        with self._file_lock:
            with open(file_path, 'r') as f:
                return json.load(f)
    
    def _write_json(self, file_path: str, data: Dict[str, Any]) -> None:
        """Write data to a JSON file atomically with thread safety."""
        with self._file_lock:
            # Use atomic write: write to temporary file first, then rename
            temp_file = None
            try:
                # Create temporary file in same directory as target file
                dir_name = os.path.dirname(file_path)
                file_name = os.path.basename(file_path)
                
                with tempfile.NamedTemporaryFile(
                    mode='w', 
                    dir=dir_name, 
                    prefix=f'.tmp_{file_name}_',
                    suffix='.json',
                    delete=False
                ) as f:
                    temp_file = f.name
                    json.dump(data, f, indent=2, cls=SafeJSONEncoder)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                
                # Atomic rename (only works on same filesystem)
                if os.name == 'nt':  # Windows
                    # On Windows, remove target file first if it exists
                    if os.path.exists(file_path):
                        os.remove(file_path)
                os.rename(temp_file, file_path)
                temp_file = None  # Successfully renamed
                
            except Exception as e:
                # Clean up temporary file if operation failed
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass  # Ignore cleanup errors
                raise e 