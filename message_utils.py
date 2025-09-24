#!/usr/bin/env python3
"""
Message utility functions for consistent message processing across the system.

This module provides common functionality for extracting and formatting message content,
following DRY principles to avoid code duplication.
"""

from strands.types.content import Message


def extract_text_content(message: Message) -> str:
    """Extract text content from a message.

    This is the single source of truth for text extraction across the system.

    Args:
        message: The message to extract text from.

    Returns:
        Extracted text content.
    """
    text_parts = []
    for content in message.get("content", []):
        if isinstance(content, dict) and "text" in content:
            text_parts.append(content["text"])
        elif isinstance(content, str):
            text_parts.append(content)

    return " ".join(text_parts)


def format_message_for_indexing(message: Message) -> str:
    """Format a message for semantic indexing.

    Args:
        message: The message to format

    Returns:
        Formatted string for indexing
    """
    role = message.get("role", "unknown")
    content_parts = []

    for content in message.get("content", []):
        if isinstance(content, dict):
            if "text" in content:
                content_parts.append(content["text"])
            elif "toolUse" in content:
                tool_use = content["toolUse"]
                content_parts.append(
                    f"Tool: {tool_use.get('name', 'unknown')} with input: {tool_use.get('input', {})}"
                )
            elif "toolResult" in content:
                tool_result = content["toolResult"]
                content_parts.append(
                    f"Tool Result: {tool_result.get('output', 'no output')}"
                )
        elif isinstance(content, str):
            content_parts.append(content)

    return f"[{role}] {' '.join(content_parts)}"


def generate_message_id(message: Message) -> str:
    """Generate a unique ID for a message.

    Args:
        message: The message to generate an ID for.

    Returns:
        A string ID for the message.
    """
    content_str = str(message.get("content", []))
    return f"{message.get('role', 'unknown')}_{hash(content_str)}"
