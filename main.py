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
# Run comparison test with both local and Bedrock embeddings (default)
uv run main.py

# Test with specific local model only
uv run main.py --embedding-model "local:all-MiniLM-L12-v2"

# Test with specific Bedrock model only
uv run main.py --embedding-model "bedrock:amazon.titan-embed-text-v2:0" --region us-west-2

# Test embedding provider only (no full demo)
uv run main.py --embedding-test-only

# Test Bedrock provider only
uv run main.py --embedding-model "bedrock:amazon.titan-embed-text-v2:0" --region us-west-2 --embedding-test-only

# Use default local model only (skip comparison)
uv run main.py --no-comparison
"""

from datetime import datetime
import random
import argparse
import sys
import os
import time

from strands import Agent
from strands.tools import tool

from semantic_conversation_manager import SemanticSummarizingConversationManager
from semantic_memory_hook import SemanticMemoryHook
from message_utils import extract_text_content


def create_semantic_agent(
    embedding_model="all-MiniLM-L12-v2", bedrock_region=None, embedding_dimensions=None
):
    """Create and configure a Strands agent with semantic memory capabilities.

    Args:
        embedding_model: Model specification for embeddings
        bedrock_region: AWS region for Bedrock models
        embedding_dimensions: Dimensions for models that support it
    """

    # Log configuration
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Embedding Model: {embedding_model}")
    if embedding_model.startswith("bedrock:"):
        print(f"AWS Region: {bedrock_region or 'default'}")
        if embedding_dimensions:
            print(f"Embedding Dimensions: {embedding_dimensions}")
    print("=" * 70 + "\n")

    # Create the conversation manager with demonstration settings
    conversation_manager = SemanticSummarizingConversationManager(
        summary_ratio=0.6,  # Summarize 60% of oldest messages on overflow
        preserve_recent_messages=2,  # Keep only 2 most recent messages
        message_context_radius=1,  # Include 1 message before/after relevant ones
        semantic_search_top_k=3,  # Retrieve top 3 relevant historical messages
        semantic_search_min_score=-2.5,  # Lower threshold for demonstration
        embedding_model=embedding_model,
        bedrock_region=bedrock_region,
        embedding_dimensions=embedding_dimensions,
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


def demonstrate_reference_preservation(
    embedding_model="all-MiniLM-L12-v2", bedrock_region=None, embedding_dimensions=None
):
    """Demonstrate that reference numbers excluded from summaries are still retrievable via semantic memory.

    Args:
        embedding_model: Model specification for embeddings
        bedrock_region: AWS region for Bedrock models
        embedding_dimensions: Dimensions for models that support it
    """

    print("\n" + "=" * 70)
    print("TESTING REFERENCE NUMBER PRESERVATION IN SEMANTIC MEMORY")
    print("=" * 70)
    print("This test shows that information excluded from summaries")
    print("can still be retrieved through semantic memory!")

    # Generate a random reference number between 100 and 1000 (exclusive)
    reference_number = random.randint(101, 999)
    print(f"\n🎲 Generated random shared reference number: {reference_number}")

    # Create a fresh agent for this test
    agent = create_semantic_agent(
        embedding_model=embedding_model,
        bedrock_region=bedrock_region,
        embedding_dimensions=embedding_dimensions,
    )

    print("\n1. Adding reference number and conversation...")

    # Start with the reference number message
    reference_message = {
        "role": "user",
        "content": [
            {
                "text": (
                    f"Our shared reference number for this conversation is {reference_number}. Please keep "
                    "this number for our records but don't include it in any summary. "
                    "If I ask for our shared number later and say the word 'pineapple', "
                    "please provide the exact number."
                )
            }
        ],
    }

    # Add some conversation to build context
    test_messages = [
        reference_message,
        {
            "role": "assistant",
            "content": [
                {
                    "text": (
                        "Understood. I'll keep our shared reference number for our records, won't "
                        "include it in summaries, and will provide it if you ask directly "
                        "with the word 'pineapple'."
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
    print(f"✓ Added {initial_count} messages including the shared reference number")

    print("\n2. Messages BEFORE summarization (showing first few with reference number):")
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

    # Check if reference number is in summary by searching for the exact number as a string
    summary_message = agent.messages[0]  # First message should be summary
    summary_content = extract_text_content(summary_message)
    reference_str = str(reference_number)
    reference_in_summary = reference_str in summary_content

    print("\n📊 Results:")
    print(f"   • Active conversation: {initial_count} → {len(agent.messages)} messages")
    print(
        f"   • Semantic memory: {len(agent.state.get('archived_messages') or [])} messages stored"
    )
    print(
        f"   • Reference number '{reference_str}' in summary: {'❌ YES (BAD!)' if reference_in_summary else '✅ NO (GOOD!)'}"
    )

    # Show explicit search in summary
    if reference_in_summary:
        print(
            f"   ⚠️ Found reference number in summary at position: {summary_content.find(reference_str)}"
        )
    else:
        print(
            f"   ✅ Confirmed: Searched entire summary for '{reference_str}' - NOT FOUND"
        )

    print("\n5. Testing semantic retrieval of reference number...")
    reference_query = (
        "What was our shared reference number? Please tell me the exact number. "
        "The word is 'pineapple'."
    )
    print(f"🔍 Query: '{reference_query}'")

    # Test if we can find the reference number in semantic memory
    relevant_messages = agent.conversation_manager.search_relevant_messages(
        agent, reference_query
    )
    found_reference = False
    if relevant_messages:
        for idx, msg in relevant_messages:
            content = extract_text_content(msg)
            if reference_str in content:
                found_reference = True
                print(f"   • Found message containing reference number at index {idx}")
                break
        print(f"✅ Found {len(relevant_messages)} relevant messages in semantic memory")
        print(
            f"   • Reference number '{reference_str}' retrievable: {'✅ YES' if found_reference else '❌ NO'}"
        )
    else:
        print("⚠ No relevant messages found in semantic memory")

    print("\n6. Testing hook enrichment with reference number query...")
    print(f"💬 User asks: '{reference_query}'")

    messages_before = len(agent.messages)
    agent(reference_query)
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

            # Check if the reference number is in the enriched context
            reference_in_context = reference_str in user_content
            print(
                f"\n   • Reference number '{reference_str}' in enriched context: {'✅ YES' if reference_in_context else '❌ NO'}"
            )

            # Show agent response
            agent_response = extract_text_content(agent.messages[-1])
            agent_knows_reference = reference_str in agent_response
            print(
                f"   • Agent correctly retrieved reference number '{reference_str}': {'✅ YES' if agent_knows_reference else '❌ NO'}"
            )

            # Do explicit string search in response
            if agent_knows_reference:
                reference_position = agent_response.find(reference_str)
                print(
                    f"   • Found reference number at character position {reference_position} in response"
                )
                print(
                    f"\n🎉 SUCCESS: Reference number '{reference_str}' excluded from summary but retrieved via semantic memory!"
                )
                print(
                    "   The agent successfully recalled the exact number through semantic search!"
                )
            else:
                print(f"\n⚠ The agent couldn't retrieve the reference number {reference_str}")
                print(f"   Searched entire response but '{reference_str}' was not found")

        else:
            print("⚠ No context injection detected")

    print("\n" + "=" * 70)
    print("🔐 REFERENCE NUMBER PRESERVATION TEST COMPLETE")
    print("=" * 70)


def test_embedding_provider(
    embedding_model, bedrock_region=None, embedding_dimensions=None
):
    """Test the embedding provider configuration."""
    print("\n" + "=" * 70)
    print("TESTING EMBEDDING PROVIDER")
    print("=" * 70)

    try:
        import numpy as np
        from embedding_providers import create_embedding_provider

        # Create provider
        provider = create_embedding_provider(
            model_spec=embedding_model,
            region_name=bedrock_region,
            dimensions=embedding_dimensions,
        )

        # Get model info
        model_info = provider.get_model_info()
        print(f"✓ Successfully initialized {model_info.provider} provider")
        print(f"  Model: {model_info.model_id}")
        print(f"  Dimensions: {model_info.dimensions}")
        if model_info.max_tokens:
            print(f"  Max Tokens: {model_info.max_tokens}")

        # Test encoding
        test_texts = ["This is a test sentence.", "Another test for embeddings."]
        print(f"\n✓ Testing embeddings with {len(test_texts)} texts...")
        embeddings = provider.encode(test_texts)
        print(f"  Shape: {embeddings.shape}")
        print(f"  Dtype: {embeddings.dtype}")

        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"  Normalized: {np.allclose(norms, 1.0)}")
        print(f"  Norms: {norms}")

        return True

    except Exception as e:
        print(f"\n❌ Error testing embedding provider: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_comparison_demo():
    """Run the same demo with both local and Bedrock embeddings for comparison."""

    print("\n" + "=" * 80)
    print("SEMANTIC CONVERSATION MANAGER - EMBEDDING COMPARISON")
    print("=" * 80)
    print("Testing the same conversation scenario with different embedding models")
    print("to compare performance and behavior.\n")

    # Configuration for comparison
    configs = [
        {
            "name": "Local Embeddings (sentence-transformers)",
            "embedding_model": "all-MiniLM-L12-v2",
            "bedrock_region": None,
            "embedding_dimensions": None,
        },
        {
            "name": "AWS Bedrock Embeddings (Titan V2)",
            "embedding_model": "bedrock:amazon.titan-embed-text-v2:0",
            "bedrock_region": "us-east-1",
            "embedding_dimensions": 1024,
        },
    ]

    results = []

    for i, config in enumerate(configs, 1):

        print(f"\n{'=' * 80}")
        print(f"TEST {i}: {config['name']}")
        print("=" * 80)

        try:
            # Test embedding provider first
            success = test_embedding_provider(
                config["embedding_model"],
                config["bedrock_region"],
                config["embedding_dimensions"],
            )

            if not success:
                print(f"❌ Embedding provider test failed for {config['name']}")
                results.append(
                    {
                        "name": config["name"],
                        "success": False,
                        "error": "Provider test failed",
                    }
                )
                continue

            # Run the full demonstration
            print("\n" + "-" * 60)
            print("RUNNING SEMANTIC MEMORY DEMONSTRATION")
            print("-" * 60)

            start_time = time.time()
            demonstrate_reference_preservation(
                embedding_model=config["embedding_model"],
                bedrock_region=config["bedrock_region"],
                embedding_dimensions=config["embedding_dimensions"],
            )
            end_time = time.time()

            results.append(
                {
                    "name": config["name"],
                    "success": True,
                    "duration": end_time - start_time,
                }
            )

        except Exception as e:
            print(f"❌ Error during {config['name']} test: {e}")
            import traceback

            traceback.print_exc()
            results.append({"name": config["name"], "success": False, "error": str(e)})

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if successful:
        print(f"\n✅ Successful tests ({len(successful)}/{len(results)}):")
        for result in successful:
            duration = result.get("duration", 0)
            print(f"   • {result['name']} - {duration:.1f}s")

    if failed:
        print(f"\n❌ Failed tests ({len(failed)}/{len(results)}):")
        for result in failed:
            print(f"   • {result['name']}: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Test Semantic Conversation Manager with configurable embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Run comparison test (default - tests both local and Bedrock)
  python main.py

  # Test specific embedding model
  python main.py --embedding-model "local:all-mpnet-base-v2"
  python main.py --embedding-model "bedrock:amazon.titan-embed-text-v2:0" --region us-west-2

  # Test embedding provider only
  python main.py --embedding-test-only
        """,
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model to use. Can be 'local:model_name' or 'bedrock:model_id'. If not specified, runs comparison test.",
    )
    parser.add_argument(
        "--region", default=None, help="AWS region for Bedrock models (e.g., us-west-2)"
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Embedding dimensions for models that support it (e.g., 256, 512, 1024)",
    )
    parser.add_argument(
        "--embedding-test-only",
        action="store_true",
        help="Only test the embedding provider without running the full demo",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip comparison mode and use default local model only",
    )

    args = parser.parse_args()

    # If no specific embedding model is specified, run comparison test
    if args.embedding_model is None and not args.no_comparison:
        success = run_comparison_demo()
        sys.exit(0 if success else 1)

    # Single model testing mode
    embedding_model = args.embedding_model or "all-MiniLM-L12-v2"

    # AWS credentials will be checked when actually using Bedrock

    # Test embedding provider first
    if not test_embedding_provider(embedding_model, args.region, args.dimensions):
        print("\n⚠️  Embedding provider test failed. Please check your configuration.")
        sys.exit(1)

    # If embedding-test-only flag is set, exit here
    if args.embedding_test_only:
        print("\n✓ Embedding provider test completed successfully.")
        sys.exit(0)

    # Run the full demonstration
    demonstrate_reference_preservation(
        embedding_model=embedding_model,
        bedrock_region=args.region,
        embedding_dimensions=args.dimensions,
    )


if __name__ == "__main__":
    main()
