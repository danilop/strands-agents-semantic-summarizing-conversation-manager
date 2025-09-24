"""Message container with automatic semantic indexing and memory management."""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from memory_estimator import estimate_memory_size

if TYPE_CHECKING:
    from semantic_search import SemanticSearch


class ArchivedMessageContainer:
    """
    Container that manages message storage with automatic semantic indexing.

    Provides automatic memory management with configurable limits by message count
    or memory usage. Messages are stored with semantic indexing for intelligent
    retrieval, and oldest messages are automatically removed when limits are exceeded.
    """

    def __init__(
        self,
        semantic_search: Optional["SemanticSearch"] = None,
        max_messages: Optional[int] = None,
        max_memory_bytes: Optional[int] = None,
    ):
        self._messages = []  # Single source of truth for message data
        self._semantic_search = semantic_search
        self._max_messages = max_messages
        self._max_memory_bytes = max_memory_bytes

        # Memory tracking (O(1) operations)
        self._total_memory = 0
        self._bytes_per_embedding = None
        self._model_analyzed = False

    def _analyze_model(self):
        """Extract embedding specs from semantic search model (one-time)."""
        if self._model_analyzed or not self._semantic_search:
            return

        try:
            encoder = self._semantic_search._encoder
            sample = encoder.encode("test", convert_to_numpy=True)
            self._bytes_per_embedding = sample.nbytes
            self._model_analyzed = True
        except Exception:
            self._bytes_per_embedding = 1536  # Conservative default
            self._model_analyzed = True

    def add_message(self, msg_data: Dict[str, Any]) -> None:
        """Add a message with automatic semantic indexing and memory management."""
        # Add to message list
        self._messages.append(msg_data)

        # Update memory counter
        msg_memory = estimate_memory_size(msg_data)
        if not self._model_analyzed:
            self._analyze_model()
        embedding_memory = self._bytes_per_embedding or 1536
        self._total_memory += msg_memory + embedding_memory

        # Add to semantic index
        if self._semantic_search is not None:
            from message_utils import format_message_for_indexing

            content = format_message_for_indexing(msg_data["message"])
            self._semantic_search.add(content)

        # Apply limits if needed
        self._apply_limits()

    def _apply_limits(self) -> None:
        """Apply memory and count limits by removing oldest messages."""
        while self._should_remove_oldest():
            self._remove_oldest()

    def _should_remove_oldest(self) -> bool:
        """Check if we need to remove the oldest message."""
        count_exceeded = (
            self._max_messages is not None and len(self._messages) > self._max_messages
        )
        memory_exceeded = (
            self._max_memory_bytes is not None
            and self._total_memory > self._max_memory_bytes
        )
        return (count_exceeded or memory_exceeded) and len(self._messages) > 0

    def _remove_oldest(self) -> None:
        """Remove the oldest message from both storage and index."""
        if not self._messages:
            return

        # Remove from message list (index 0 = oldest)
        oldest_msg = self._messages.pop(0)

        # Update memory counter
        msg_memory = estimate_memory_size(oldest_msg)
        embedding_memory = self._bytes_per_embedding or 1536
        self._total_memory -= msg_memory + embedding_memory
        self._total_memory = max(0, self._total_memory)

        # Remove from semantic index (always index 0 since FIFO)
        if self._semantic_search is not None:
            try:
                self._semantic_search.remove([0])
            except Exception as e:
                print(f"Warning: Could not remove from semantic index: {e}")

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get a copy of all stored messages."""
        return self._messages.copy()

    def search(self, query: str, top_k: int = 3) -> List:
        """Search for relevant messages."""
        if self._semantic_search is None:
            return []
        return self._semantic_search.search(query, top_k=top_k)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "message_count": len(self._messages),
            "total_memory": self._total_memory,
            "max_messages": self._max_messages,
            "max_memory_bytes": self._max_memory_bytes,
            "memory_usage_percent": (
                (self._total_memory / self._max_memory_bytes * 100)
                if self._max_memory_bytes
                else 0
            ),
        }

    def __len__(self) -> int:
        return len(self._messages)

    def __bool__(self) -> bool:
        return len(self._messages) > 0


def test_container():
    """Test the message container."""
    print("Testing Message Container")
    print("=" * 50)

    from semantic_search import SemanticSearch

    # Create semantic search
    semantic_search = SemanticSearch()

    # Create container with limits
    container = ArchivedMessageContainer(
        semantic_search=semantic_search, max_messages=3, max_memory_bytes=10000
    )

    print("✓ Created message container")

    # Add messages
    for i in range(5):
        msg_data = {
            "message": {"role": "user", "content": [{"text": f"Message {i} content"}]},
            "index": i,
        }
        container.add_message(msg_data)
        print(f"  Added message {i}, container size: {len(container)}")

    print("\nFinal state:")
    print(f"  Messages in container: {len(container)}")
    print(f"  Messages in semantic index: {len(semantic_search)}")
    print(f"  Synchronized: {'✅' if len(container) == len(semantic_search) else '❌'}")

    stats = container.get_memory_stats()
    print(f"  Memory usage: {stats['total_memory']:,} bytes")
    print(f"  Limit applied: {'✅' if len(container) <= 3 else '❌'}")


if __name__ == "__main__":
    test_container()
