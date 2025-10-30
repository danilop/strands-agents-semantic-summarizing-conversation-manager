"""Memory Hook for enriching messages with historical context.

This hook intercepts user messages and enriches them with relevant historical
context retrieved via semantic search from stored conversations.
"""

import logging
from typing import TYPE_CHECKING, List

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import MessageAddedEvent
from strands.types.content import Message
from .message_utils import extract_text_content, generate_message_id

if TYPE_CHECKING:
    from strands.agent.agent import Agent


logger = logging.getLogger(__name__)


class SemanticMemoryHook(HookProvider):
    """Hook that enriches user messages with relevant historical context.

    This hook:
    1. Detects when a new user message is added to the conversation
    2. Searches for relevant historical messages using semantic search
    3. Prepends relevant context to the user message for better responses

    The hook maintains awareness of which messages have already been enriched
    to avoid duplicate context injection.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_context_length: int = 2000,
        include_metadata: bool = True,
    ):
        """Initialize the memory hook.

        Args:
            enabled: Whether the hook is enabled. Defaults to True.
            max_context_length: Maximum character length for injected context.
                Defaults to 2000 characters.
            include_metadata: Whether to include metadata (e.g., message indices)
                in the context. Defaults to True.
        """
        self.enabled = enabled
        self.max_context_length = max_context_length
        self.include_metadata = include_metadata
        self._enriched_message_ids: set = (
            set()
        )  # Track which messages we've already enriched

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register hook callbacks with the registry.

        Args:
            registry: The hook registry to register callbacks with.
        """
        if self.enabled:
            # Use MessageAddedEvent to enrich the message when it's added
            registry.add_callback(MessageAddedEvent, self._enrich_user_message_on_add)

    def _enrich_user_message_on_add(self, event: MessageAddedEvent) -> None:
        """Enrich user messages when they are added to the conversation.

        Args:
            event: The message added event.
        """
        # Only process user messages
        if event.message.get("role") != "user":
            return

        agent = event.agent

        # Check if the conversation manager supports semantic search
        conv_manager = getattr(agent, "conversation_manager", None)
        if conv_manager is None:
            return

        # Import here to avoid circular dependency
        from .semantic_conversation_manager import SemanticSummarizingConversationManager

        if not isinstance(conv_manager, SemanticSummarizingConversationManager):
            logger.debug(
                "Conversation manager does not support semantic search, skipping enrichment"
            )
            return

        user_message = event.message

        # Check if we've already enriched this message (using a simple hash of content)
        message_id = generate_message_id(user_message)
        if message_id in self._enriched_message_ids:
            logger.debug("Message already enriched, skipping")
            return

        # Extract text content from the user message for search
        query = extract_text_content(user_message)
        if not query:
            logger.debug("No text content found in user message")
            return

        # Search for relevant historical messages
        relevant_messages = conv_manager.search_relevant_messages(agent, query)

        if not relevant_messages:
            logger.debug("No relevant historical messages found")
            return

        # Filter out historical messages that have already been injected in current conversation
        new_relevant_messages = []
        for idx, message in relevant_messages:
            if not self._is_message_already_in_conversation(agent, message):
                new_relevant_messages.append((idx, message))

        if not new_relevant_messages:
            logger.debug(
                "All relevant historical messages have already been injected in this conversation"
            )
            return

        # Format the historical context
        context = self._format_historical_context(new_relevant_messages)
        if not context:
            return

        # Find the message in agent.messages and enrich it
        # The message should be the last one added
        for i in range(len(agent.messages) - 1, -1, -1):
            if agent.messages[i] is event.message:
                self._prepend_context_to_message(agent.messages, i, context)
                break

        # Mark this message as enriched
        self._enriched_message_ids.add(message_id)
        logger.info(
            f"Enriched user message with {len(new_relevant_messages)} new historical messages"
        )

    def _format_historical_context(
        self, relevant_messages: List[tuple[int, Message]]
    ) -> str:
        """Format historical messages into a context string.

        Args:
            relevant_messages: List of (index, message) tuples.

        Returns:
            Formatted context string.
        """
        if not relevant_messages:
            return ""

        context_parts = []
        context_parts.append(
            "Based on our previous conversation, these earlier exchanges may be relevant to your current question:\n"
        )
        context_parts.append("---Previous Context---\n")

        current_length = len(context_parts[0]) + len(context_parts[1])

        for idx, message in relevant_messages:
            role = message.get("role", "unknown")
            content = extract_text_content(message)

            # Format the message
            if self.include_metadata:
                formatted = f"[Message {idx}, {role}]: {content}\n"
            else:
                formatted = f"[{role}]: {content}\n"

            # Check if adding this would exceed our limit
            if current_length + len(formatted) > self.max_context_length:
                # Try to add a truncated version
                remaining = (
                    self.max_context_length - current_length - 20
                )  # Leave room for ellipsis
                if remaining > 50:  # Only add if we have reasonable space
                    truncated = formatted[:remaining] + "...\n"
                    context_parts.append(truncated)
                break

            context_parts.append(formatted)
            current_length += len(formatted)

        context_parts.append("---End Previous Context---\n\n")
        context_parts.append("Current question: ")

        return "".join(context_parts)

    def _prepend_context_to_message(
        self, messages: List[Message], message_index: int, context: str
    ) -> None:
        """Prepend context to a message's content.

        Args:
            messages: The list of messages.
            message_index: Index of the message to modify.
            context: Context string to prepend.
        """
        message = messages[message_index]

        # Get existing content
        existing_content = message.get("content", [])

        # Prepend context to the first text content
        modified = False
        new_content = []

        for content in existing_content:
            if not modified and isinstance(content, dict) and "text" in content:
                # Prepend context to the first text block
                content["text"] = context + content["text"]
                modified = True
            elif not modified and isinstance(content, str):
                # If content is a plain string (shouldn't happen in Strands but handle it)
                content = context + content
                modified = True
            new_content.append(content)

        # If no text content was found, add context as a new text block
        if not modified:
            new_content.insert(0, {"text": context + "(No previous message content)"})

        # Update the message
        message["content"] = new_content

    def _is_message_already_in_conversation(
        self, agent: "Agent", historical_message: Message
    ) -> bool:
        """Check if a historical message's content is already present in the current conversation.

        This prevents duplicate injection of the same historical content across sessions.

        Args:
            agent: The agent instance
            historical_message: The historical message to check

        Returns:
            True if the message content is already in the conversation
        """
        # Extract text content from the historical message
        historical_text = extract_text_content(historical_message)
        if (
            not historical_text or len(historical_text.strip()) < 10
        ):  # Skip very short messages
            return False

        # Check if this text appears in any current conversation message
        for current_message in agent.messages:
            current_text = extract_text_content(current_message)

            # Check if the historical text is already contained in the current message
            # (this would indicate it was previously injected)
            if historical_text in current_text:
                return True

        return False
