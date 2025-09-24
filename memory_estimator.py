import sys
from typing import Any, Set


def estimate_memory_size(obj: Any, seen: Set[int] = None) -> int:
    """
    Estimate the memory size of an object in bytes, including nested objects.

    Args:
        obj: The object to measure
        seen: Set of object IDs already counted (to avoid double-counting)

    Returns:
        Estimated memory size in bytes
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)
    size = sys.getsizeof(obj)

    # Handle different container types
    if isinstance(obj, dict):
        size += sum(
            estimate_memory_size(k, seen) + estimate_memory_size(v, seen)
            for k, v in obj.items()
        )
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(estimate_memory_size(item, seen) for item in obj)
    elif hasattr(obj, "__dict__"):
        size += estimate_memory_size(obj.__dict__, seen)
    elif hasattr(obj, "__slots__"):
        size += sum(
            estimate_memory_size(getattr(obj, slot, None), seen)
            for slot in obj.__slots__
            if hasattr(obj, slot)
        )

    return size


def format_bytes(size: int) -> str:
    """Format bytes into human-readable units."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


# Example usage
if __name__ == "__main__":
    # Example agent state
    agent_state = {
        "conversation_history": ["Hello", "How can I help?", "Tell me about Python"],
        "user_preferences": {"language": "en", "theme": "dark"},
        "session_data": {
            "start_time": "2024-01-01T10:00:00",
            "messages_count": 42,
            "context": [
                "Previous conversation about ML",
                "User asked about memory estimation",
            ],
        },
        "model_params": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "embeddings": [0.1] * 1536,  # Simulated embedding vector
        },
    }

    memory_size = estimate_memory_size(agent_state)
    print(
        f"Agent state memory usage: {memory_size} bytes ({format_bytes(memory_size)})"
    )
