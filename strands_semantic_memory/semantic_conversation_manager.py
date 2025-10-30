"""Conversation Manager with summarization and semantic memory.

This manager provides intelligent conversation management by summarizing older messages
while maintaining a semantic index for retrieving relevant historical context.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, cast

from typing_extensions import override

from strands.types.content import Message
from strands.types.exceptions import ContextWindowOverflowException
from strands.agent.conversation_manager.conversation_manager import ConversationManager

# Import the local SemanticSearch
from .semantic_search import SemanticSearch, SearchConfig
from .message_container import ArchivedMessageContainer

if TYPE_CHECKING:
    from strands.agent.agent import Agent


logger = logging.getLogger(__name__)


DEFAULT_SUMMARIZATION_PROMPT = """You are a conversation summarizer. Provide a concise summary of the conversation \
history.

Format Requirements:
- You MUST create a structured and concise summary in bullet-point format.
- You MUST NOT respond conversationally.
- You MUST NOT address the user directly.

Task:
Your task is to create a structured summary document:
- It MUST contain bullet points with key topics and questions covered
- It MUST contain bullet points for all significant tools executed and their results
- It MUST contain bullet points for any code or technical information shared
- It MUST contain a section of key insights gained
- It MUST format the summary in the third person

Example format:

## Conversation Summary
* Topic 1: Key information
* Topic 2: Key information
*
## Tools Executed
* Tool X: Result Y"""


class SemanticSummarizingConversationManager(ConversationManager):
    """Conversation manager with summarization and semantic memory.

    This manager provides:
    1. Automatic summarization when context overflows
    2. Exact message storage in agent K/V state and semantic index
    3. Semantic search for retrieving relevant historical context

    The manager maintains a semantic index of all archived messages,
    enabling intelligent context retrieval based on relevance to current topics.
    """

    def __init__(
        self,
        summary_ratio: float = 0.3,
        preserve_recent_messages: int = 10,
        summarization_agent: Optional["Agent"] = None,
        summarization_system_prompt: Optional[str] = None,
        message_context_radius: int = 2,
        semantic_search_top_k: int = 3,
        semantic_search_min_score: float = -2.0,
        max_num_archived_messages: Optional[int] = None,
        max_memory_archived_messages: Optional[int] = None,
        embedding_model: str = "all-MiniLM-L12-v2",
        aws_region: Optional[str] = None,
        embedding_dimensions: Optional[int] = None,
        backend: str = "numpy",
        ann_engine: Optional[str] = "hnswlib",
    ):
        """Initialize the conversation manager with semantic memory.

        Args:
            summary_ratio: Ratio of messages to summarize vs keep when context overflow occurs.
                Value between 0.1 and 0.8. Defaults to 0.3 (summarize 30% of oldest messages).
            preserve_recent_messages: Minimum number of recent messages to always keep.
                Defaults to 10 messages.
            summarization_agent: Optional agent to use for summarization instead of the parent agent.
                If provided, this agent can use tools as part of the summarization process.
            summarization_system_prompt: Optional system prompt override for summarization.
                If None, uses the default summarization prompt.
            message_context_radius: Number of messages before and after a semantically relevant message
                to include for context. Defaults to 2 (include 2 messages before and 2 after).
            semantic_search_top_k: Maximum number of semantically similar messages to retrieve.
                Defaults to 3.
            semantic_search_min_score: Minimum relevance score threshold for semantic search results.
                Uses cross-encoder logits where > -2.0 indicates relevant content. Defaults to -2.0.
            max_num_archived_messages: Optional maximum number of archived messages to keep.
                If None, no limit is applied. When limit is reached, oldest messages are removed.
            max_memory_archived_messages: Optional maximum memory usage in bytes for archived
                messages and their semantic embeddings. If None, no limit is applied.
            embedding_model: Embedding model to use. Can be:
                - "model_name" or "local:model_name" for sentence-transformers models
                - "bedrock:model_id" for AWS Bedrock models (e.g., "bedrock:amazon.titan-embed-text-v2:0")
                Defaults to "all-MiniLM-L12-v2".
            aws_region: AWS region for Bedrock models. Optional, uses boto3 default if not specified.
            embedding_dimensions: Dimensions for models that support variable dimensions.
                For example, Titan v2 supports 256, 512, or 1024 dimensions.
            backend: Search backend. "numpy" for exact search (default), "ann" for approximate nearest neighbors.
            ann_engine: ANN engine when backend="ann". Options: "hnswlib" (default), "faiss".
        """
        super().__init__()
        if summarization_agent is not None and summarization_system_prompt is not None:
            raise ValueError(
                "Cannot provide both summarization_agent and summarization_system_prompt. "
                "Agents come with their own system prompt."
            )

        self.summarization_agent = summarization_agent
        self.summarization_system_prompt = summarization_system_prompt
        self.message_context_radius = max(0, message_context_radius)
        self.semantic_search_top_k = semantic_search_top_k
        self.semantic_search_min_score = semantic_search_min_score

        # Configure summarization parameters
        self.summary_ratio = max(0.1, min(0.8, summary_ratio))
        self.preserve_recent_count = preserve_recent_messages
        self.min_messages_for_summary = 5

        self._summary_message: Optional[Message] = None
        self._semantic_index: Optional[SemanticSearch] = None
        self._message_id_counter = 0  # To track message indices in the full history
        self._container: Optional[ArchivedMessageContainer] = None

        # Store limits for lazy container initialization
        self._max_num_archived_messages = max_num_archived_messages
        self._max_memory_archived_messages = max_memory_archived_messages

        # Store embedding configuration
        self._embedding_model = embedding_model
        self._aws_region = aws_region
        self._embedding_dimensions = embedding_dimensions
        self._backend = backend
        self._ann_engine = ann_engine

    def _initialize_semantic_index(self) -> SemanticSearch:
        """Initialize or get the semantic search index."""
        if self._semantic_index is None:
            # Create search config with embedding model settings
            config = SearchConfig(
                embedding_model=self._embedding_model,
                aws_region=self._aws_region,
                embedding_dimensions=self._embedding_dimensions,
                backend=self._backend,
                ann_engine=self._ann_engine,
                auto_index=True,
            )
            self._semantic_index = SemanticSearch(config=config)
        return self._semantic_index

    def _ensure_container(self) -> ArchivedMessageContainer:
        """Initialize the message container with semantic indexing."""
        if self._container is None:
            if self._semantic_index is None:
                self._semantic_index = self._initialize_semantic_index()

            self._container = ArchivedMessageContainer(
                semantic_search=self._semantic_index,
                max_messages=self._max_num_archived_messages,
                max_memory_bytes=self._max_memory_archived_messages,
            )
        return self._container

    @override
    def restore_from_session(self, state: dict[str, Any]) -> Optional[list[Message]]:
        """Restore the conversation manager from its previous state in a session.

        Args:
            state: The previous state of the conversation manager.

        Returns:
            Optionally returns the previous conversation summary if it exists.
        """
        super().restore_from_session(state)
        self._summary_message = state.get("summary_message")
        self._message_id_counter = state.get("message_id_counter", 0)

        # Restore embedding configuration if available
        self._embedding_model = state.get("embedding_model", self._embedding_model)
        self._aws_region = state.get("aws_region", self._aws_region)
        self._embedding_dimensions = state.get(
            "embedding_dimensions", self._embedding_dimensions
        )

        # Restore semantic index and message container from stored messages
        archived_messages = state.get("archived_messages", [])
        if archived_messages:
            self._semantic_index = self._initialize_semantic_index()
            # Initialize container and restore messages
            container = self._ensure_container()
            for msg_data in archived_messages:
                container.add_message(msg_data)

        return [self._summary_message] if self._summary_message else None

    def get_state(self) -> dict[str, Any]:
        """Returns a dictionary representation of the state for the conversation manager."""
        state = {
            "summary_message": self._summary_message,
            "message_id_counter": self._message_id_counter,
            "embedding_model": self._embedding_model,
            "aws_region": self._aws_region,
            "embedding_dimensions": self._embedding_dimensions,
            **super().get_state(),
        }

        # Note: archived_messages will be stored in agent.state separately
        # as it needs to be accessed by both the manager and the hook

        return state

    def apply_management(self, agent: "Agent", **_kwargs: Any) -> None:
        """Apply management strategy to conversation history.

        This implementation performs no proactive management.
        Summarization only occurs when context overflow triggers reduce_context.

        Args:
            agent: The agent whose conversation history will be managed.
                The agent's messages list is modified in-place.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        # No proactive management - summarization only happens on context overflow
        pass

    def reduce_context(
        self, agent: "Agent", e: Optional[Exception] = None, **_kwargs: Any
    ) -> None:
        """Reduce context using summarization and store exact messages for semantic recall.

        Args:
            agent: The agent whose conversation history will be reduced.
                The agent's messages list is modified in-place.
            e: The exception that triggered the context reduction, if any.
            **kwargs: Additional keyword arguments for future extensibility.

        Raises:
            ContextWindowOverflowException: If the context cannot be summarized.
        """
        try:
            # Determine what to summarize
            messages_to_summarize_count = self._calculate_messages_to_summarize(
                len(agent.messages)
            )

            if messages_to_summarize_count <= 0:
                raise ContextWindowOverflowException(
                    "Cannot summarize: insufficient messages for summarization"
                )

            # Adjust split point to avoid breaking ToolUse/ToolResult pairs
            messages_to_summarize_count = self._adjust_split_point_for_tool_pairs(
                agent.messages, messages_to_summarize_count
            )

            if messages_to_summarize_count <= 0:
                raise ContextWindowOverflowException(
                    "Cannot summarize: insufficient messages for summarization"
                )

            # Extract messages to summarize
            messages_to_summarize = agent.messages[:messages_to_summarize_count]
            remaining_messages = agent.messages[messages_to_summarize_count:]

            # Store exact messages in agent K/V state and semantic index
            self._store_messages_in_state(agent, messages_to_summarize)

            # Keep track of the number of messages that have been summarized thus far
            self.removed_message_count += len(messages_to_summarize)
            # If there is a summary message, don't count it in the removed_message_count
            if self._summary_message:
                self.removed_message_count -= 1

            # Generate summary
            self._summary_message = self._generate_summary(messages_to_summarize, agent)

            # Replace the summarized messages with the summary
            agent.messages[:] = [self._summary_message] + remaining_messages

        except Exception as summarization_error:
            logger.error("Summarization failed: %s", summarization_error)
            raise summarization_error from e

    def _store_messages_in_state(self, agent: "Agent", messages: List[Message]) -> None:
        """Store messages in agent K/V state and semantic index with memory management.

        Args:
            agent: The agent instance
            messages: Messages to store
        """
        # Get message container
        container = self._ensure_container()

        # Add each message with automatic semantic indexing and memory management
        for msg in messages:
            msg_data = {"message": msg, "index": self._message_id_counter}
            container.add_message(msg_data)
            self._message_id_counter += 1

        # Update agent state with current container messages
        agent.state.set("archived_messages", container.get_messages())

    def search_relevant_messages(
        self, agent: "Agent", query: str
    ) -> List[Tuple[int, Message]]:
        """Search for relevant messages in the semantic index with context radius.

        Args:
            agent: The agent instance
            query: The search query (typically the current user message)

        Returns:
            List of (index, message) tuples, sorted by index, with context radius applied
        """
        if self._container is None:
            return []

        # Use container's search method
        results = self._container.search(query, top_k=self.semantic_search_top_k)

        if not results:
            return []

        # Get all messages from container
        all_messages = self._container.get_messages()
        if not all_messages:
            return []

        # Collect indices with context radius
        indices_to_include: Set[int] = set()

        for result in results:
            # The result.index corresponds to the position in the semantic index
            # We need to map this back to the original message index
            if result.index < len(all_messages):
                center_idx = all_messages[result.index]["index"]

                # Add context radius around this message
                for offset in range(
                    -self.message_context_radius, self.message_context_radius + 1
                ):
                    target_idx = center_idx + offset
                    # Check if this index exists in our stored messages
                    for msg_data in all_messages:
                        if msg_data["index"] == target_idx:
                            indices_to_include.add(target_idx)
                            break

        # Collect messages for included indices
        messages_to_return = []
        for msg_data in all_messages:
            if msg_data["index"] in indices_to_include:
                messages_to_return.append((msg_data["index"], msg_data["message"]))

        # Sort by index to maintain chronological order
        messages_to_return.sort(key=lambda x: x[0])

        return messages_to_return

    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage statistics for archived messages.

        Returns:
            Dictionary containing detailed memory usage information including:
            - message_count: Number of archived messages stored
            - archived_messages_memory: Memory used by message text data
            - semantic_index_memory: Memory used by semantic embeddings
            - total_memory: Total memory usage
            - limits: Current memory limits
            - memory_usage_percent: Percentage of memory limit used (if limit is set)
        """
        if self._container is None:
            return {"message_count": 0, "total_memory": 0}
        return self._container.get_memory_stats()

    def get_memory_usage_summary(self) -> str:
        """
        Get a human-readable summary of memory usage.

        Returns:
            Formatted string with memory usage details
        """
        if self._container is None:
            return "Memory Usage: No messages stored"

        stats = self._container.get_memory_stats()

        def format_bytes(size: int) -> str:
            for unit in ["B", "KB", "MB", "GB"]:
                if size < 1024:
                    return f"{size:.1f}{unit}"
                size /= 1024
            return f"{size:.1f}TB"

        lines = [
            "Archived Messages Memory Usage:",
            f"  Messages: {stats['message_count']}",
            f"  Total: {format_bytes(stats['total_memory'])}",
        ]

        # Add limits info
        if stats.get("max_messages") is not None:
            lines.append(f"  Message Limit: {stats['max_messages']}")
        if stats.get("max_memory_bytes") is not None:
            lines.append(f"  Memory Limit: {format_bytes(stats['max_memory_bytes'])}")
            if "memory_usage_percent" in stats:
                lines.append(f"  Usage: {stats['memory_usage_percent']:.1f}%")

        return "\n".join(lines)

    def _calculate_messages_to_summarize(self, total_messages: int) -> int:
        """Calculate how many messages should be summarized."""
        if total_messages < self.min_messages_for_summary:
            return 0

        messages_to_summarize = max(1, int(total_messages * self.summary_ratio))
        return min(messages_to_summarize, total_messages - self.preserve_recent_count)

    def _generate_summary(self, messages: List[Message], agent: "Agent") -> Message:
        """Generate a summary of the provided messages.

        Args:
            messages: The messages to summarize.
            agent: The agent instance to use for summarization.

        Returns:
            A message containing the conversation summary.

        Raises:
            Exception: If summary generation fails.
        """
        # Choose which agent to use for summarization
        summarization_agent = (
            self.summarization_agent if self.summarization_agent is not None else agent
        )

        # Save original system prompt and messages to restore later
        original_system_prompt = summarization_agent.system_prompt
        original_messages = summarization_agent.messages.copy()

        try:
            # Only override system prompt if no agent was provided during initialization
            if self.summarization_agent is None:
                # Use custom system prompt if provided, otherwise use default
                system_prompt = (
                    self.summarization_system_prompt
                    if self.summarization_system_prompt is not None
                    else DEFAULT_SUMMARIZATION_PROMPT
                )
                # Temporarily set the system prompt for summarization
                summarization_agent.system_prompt = system_prompt
            summarization_agent.messages = messages

            # Use the agent to generate summary with rich content (can use tools if needed)
            result = summarization_agent("Please summarize this conversation.")
            return cast(Message, {**result.message, "role": "user"})

        finally:
            # Restore original agent state
            summarization_agent.system_prompt = original_system_prompt
            summarization_agent.messages = original_messages

    def _adjust_split_point_for_tool_pairs(
        self, messages: List[Message], split_point: int
    ) -> int:
        """Adjust the split point to avoid breaking ToolUse/ToolResult pairs.

        Args:
            messages: The full list of messages.
            split_point: The initially calculated split point.

        Returns:
            The adjusted split point that doesn't break ToolUse/ToolResult pairs.

        Raises:
            ContextWindowOverflowException: If no valid split point can be found.
        """
        if split_point > len(messages):
            raise ContextWindowOverflowException(
                "Split point exceeds message array length"
            )

        if split_point == len(messages):
            return split_point

        # Find the next valid split_point
        while split_point < len(messages):
            if (
                # Oldest message cannot be a toolResult because it needs a toolUse preceding it
                any(
                    "toolResult" in content
                    for content in messages[split_point]["content"]
                )
                or (
                    # Oldest message can be a toolUse only if a toolResult immediately follows it.
                    any(
                        "toolUse" in content
                        for content in messages[split_point]["content"]
                    )
                    and split_point + 1 < len(messages)
                    and not any(
                        "toolResult" in content
                        for content in messages[split_point + 1]["content"]
                    )
                )
            ):
                split_point += 1
            else:
                break
        else:
            # If we didn't find a valid split_point, then we throw
            raise ContextWindowOverflowException("Unable to trim conversation context!")

        return split_point
