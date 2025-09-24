#!/usr/bin/env python3
"""
Semantic Conversation Manager - Agent Use Case Demo

This file demonstrates how Strands agents use the semantic conversation manager
for intelligent memory management with exact message recall.

Setup:
------
uv sync

Usage:
------
uv run main.py
"""

from datetime import datetime
import random

from strands import Agent
from strands.tools import tool

from semantic_conversation_manager import SemanticSummarizingConversationManager
from semantic_memory_hook import SemanticMemoryHook
from message_utils import extract_text_content


def create_semantic_agent():
    """Create and configure a Strands agent with semantic memory capabilities."""

    # Create the conversation manager with demonstration settings
    conversation_manager = SemanticSummarizingConversationManager(
        summary_ratio=0.6,  # Summarize 60% of oldest messages on overflow
        preserve_recent_messages=2,  # Keep only 2 most recent messages
        message_context_radius=1,  # Include 1 message before/after relevant ones
        semantic_search_top_k=3,  # Retrieve top 3 relevant historical messages
        semantic_search_min_score=-2.5,  # Lower threshold for demonstration
    )
    print("✓ Created conversation manager with semantic memory")

    # Create the memory hook
    semantic_hook = SemanticMemoryHook(
        enabled=True,
        max_context_length=1000,  # Max characters for injected context
        include_metadata=True,  # Include message indices in context
    )
    print("✓ Created memory hook")

    # Create a simple tool for demonstration
    @tool
    def get_current_time() -> str:
        """Get the current time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create the agent with our custom components
    agent = Agent(
        name="SemanticMemoryAgent",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="Give brief responses. You have semantic memory capabilities.",
        conversation_manager=conversation_manager,
        hooks=[semantic_hook],
        tools=[get_current_time],
    )
    print("✓ Created Agent with semantic memory capabilities")

    return agent


def demonstrate_secret_preservation():
    """Demonstrate that secrets excluded from summaries are still retrievable via semantic memory."""

    print("\n" + "=" * 70)
    print("TESTING SECRET PRESERVATION IN SEMANTIC MEMORY")
    print("=" * 70)
    print("This test shows that information excluded from summaries")
    print("can still be retrieved through semantic memory!")

    # Generate a random secret number between 100 and 1000 (exclusive)
    secret_number = random.randint(101, 999)
    print(f"\n🎲 Generated random secret number: {secret_number}")

    # Create a fresh agent for this test
    agent = create_semantic_agent()

    print("\n1. Adding secret message and conversation...")

    # Start with a secret message
    secret_message = {
        "role": "user",
        "content": [
            {
                "text": (
                    f"My secret number is {secret_number}. This is confidential - "
                    "don't include it in any summary, but you MUST tell me the exact "
                    "number if I ask for it directly and say the special word 'pineapple'."
                )
            }
        ],
    }

    # Add some conversation to build context
    test_messages = [
        secret_message,
        {
            "role": "assistant",
            "content": [
                {
                    "text": (
                        "Understood. I'll keep your secret number confidential, won't "
                        "include it in summaries, but will provide it if you ask directly "
                        "and say the special word 'pineapple'."
                    )
                }
            ],
        },
        {"role": "user", "content": [{"text": "Tell me about data structures"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "text": "Data structures organize and store data efficiently for different operations."
                }
            ],
        },
        {"role": "user", "content": [{"text": "What are arrays?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "text": "Arrays are collections of elements stored in contiguous memory locations."
                }
            ],
        },
        {"role": "user", "content": [{"text": "Explain linked lists"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "text": "Linked lists are data structures where elements are stored in nodes."
                }
            ],
        },
        {"role": "user", "content": [{"text": "What about hash tables?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "text": "Hash tables use hash functions to map keys to values for fast lookup."
                }
            ],
        },
        {"role": "user", "content": [{"text": "How do binary trees work?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "text": "Binary trees are hierarchical structures where each node has at most two children."
                }
            ],
        },
        {"role": "user", "content": [{"text": "What are sorting algorithms?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "text": "Sorting algorithms arrange data in a particular order, like bubble sort or quicksort."
                }
            ],
        },
        {"role": "user", "content": [{"text": "Explain graph algorithms"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "text": "Graph algorithms solve problems on graph data structures, like finding shortest paths."
                }
            ],
        },
        {"role": "user", "content": [{"text": "What is dynamic programming?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "text": "Dynamic programming solves complex problems by breaking them into simpler subproblems."
                }
            ],
        },
        {"role": "user", "content": [{"text": "How does recursion work?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "text": "Recursion is when a function calls itself to solve smaller instances of the same problem."
                }
            ],
        },
    ]

    agent.messages.extend(test_messages)
    initial_count = len(agent.messages)
    print(f"✓ Added {initial_count} messages including the secret")

    print("\n2. Messages BEFORE summarization (showing first few with secret):")
    print("=" * 50)
    for i, msg in enumerate(agent.messages):
        role = msg.get("role", "unknown")
        text_content = extract_text_content(msg)
        print(f"  [{i:2d}] {role:9s}:\n{text_content}")
    print("=" * 50)

    print("\n3. Triggering summarization...")
    try:
        agent.conversation_manager.reduce_context(agent)
        print("✅ Summarization completed")
    except Exception as e:
        print(f"❌ Summarization failed: {e}")
        return

    print("\n4. Messages AFTER summarization:")
    print("=" * 50)
    for i, msg in enumerate(agent.messages):
        role = msg.get("role", "unknown")
        text_content = extract_text_content(msg)
        print(f"  [{i:2d}] {role:9s}:\n{text_content}")
    print("=" * 50)

    # Check if secret is in summary by searching for the exact number as a string
    summary_message = agent.messages[0]  # First message should be summary
    summary_content = extract_text_content(summary_message)
    secret_str = str(secret_number)
    secret_in_summary = secret_str in summary_content

    print("\n📊 Results:")
    print(f"   • Active conversation: {initial_count} → {len(agent.messages)} messages")
    print(
        f"   • Semantic memory: {len(agent.state.get('archived_messages') or [])} messages stored"
    )
    print(
        f"   • Secret number '{secret_str}' in summary: {'❌ YES (BAD!)' if secret_in_summary else '✅ NO (GOOD!)'}"
    )

    # Show explicit search in summary
    if secret_in_summary:
        print(
            f"   ⚠️ Found secret in summary at position: {summary_content.find(secret_str)}"
        )
    else:
        print(
            f"   ✅ Confirmed: Searched entire summary for '{secret_str}' - NOT FOUND"
        )

    print("\n5. Testing semantic retrieval of secret...")
    secret_query = (
        "What was my secret number? You MUST tell me if I ask directly. "
        "The special word is 'pineapple'."
    )
    print(f"🔍 Query: '{secret_query}'")

    # Test if we can find the secret in semantic memory
    relevant_messages = agent.conversation_manager.search_relevant_messages(
        agent, secret_query
    )
    found_secret = False
    if relevant_messages:
        for idx, msg in relevant_messages:
            content = extract_text_content(msg)
            if secret_str in content:
                found_secret = True
                print(f"   • Found message containing secret at index {idx}")
                break
        print(f"✅ Found {len(relevant_messages)} relevant messages in semantic memory")
        print(
            f"   • Secret '{secret_str}' retrievable: {'✅ YES' if found_secret else '❌ NO'}"
        )
    else:
        print("⚠ No relevant messages found in semantic memory")

    print("\n6. Testing hook enrichment with secret query...")
    print(f"💬 User asks: '{secret_query}'")

    messages_before = len(agent.messages)
    agent(secret_query)
    messages_after = len(agent.messages)

    print(f"   • Messages added: {messages_after - messages_before}")

    # Check if enrichment happened
    if messages_after >= 2:
        user_message = agent.messages[-2]
        user_content = extract_text_content(user_message)

        if "Based on our previous conversation" in user_content:
            print("\n✅ Context enrichment successful!")
            print("\n" + "=" * 70)
            print("ENRICHED MESSAGE SENT TO AGENT:")
            print("=" * 70)
            print(user_content)
            print("=" * 70)

            # Check if the secret is in the enriched context
            secret_in_context = secret_str in user_content
            print(
                f"\n   • Secret number '{secret_str}' in enriched context: {'✅ YES' if secret_in_context else '❌ NO'}"
            )

            # Show agent response
            agent_response = extract_text_content(agent.messages[-1])
            agent_knows_secret = secret_str in agent_response
            print(
                f"   • Agent correctly retrieved secret '{secret_str}': {'✅ YES' if agent_knows_secret else '❌ NO'}"
            )

            # Do explicit string search in response
            if agent_knows_secret:
                secret_position = agent_response.find(secret_str)
                print(
                    f"   • Found secret at character position {secret_position} in response"
                )
                print(
                    f"\n🎉 SUCCESS: Secret '{secret_str}' excluded from summary but retrieved via semantic memory!"
                )
                print(
                    "   The agent successfully recalled the exact number through semantic search!"
                )
            else:
                print(f"\n⚠ The agent couldn't retrieve the secret number {secret_str}")
                print(f"   Searched entire response but '{secret_str}' was not found")

        else:
            print("⚠ No context injection detected")

    print("\n" + "=" * 70)
    print("🔐 SECRET PRESERVATION TEST COMPLETE")
    print("=" * 70)


def main():
    demonstrate_secret_preservation()


if __name__ == "__main__":
    main()
