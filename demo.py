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
import time

from strands import Agent
from strands.tools import tool

from strands_semantic_memory.semantic_conversation_manager import SemanticSummarizingConversationManager
from strands_semantic_memory.semantic_memory_hook import SemanticMemoryHook
from strands_semantic_memory.message_utils import extract_text_content


def create_semantic_agent(
    embedding_model="all-MiniLM-L12-v2", aws_region=None, embedding_dimensions=None, ann_engine=None
):
    """Create and configure a Strands agent with semantic memory capabilities.

    Args:
        embedding_model: Model specification for embeddings
        aws_region: AWS region for Bedrock models
        embedding_dimensions: Dimensions for models that support it
        ann_engine: Optional ANN engine (hnswlib or faiss) for approximate search
    """

    # Log configuration
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Embedding Model: {embedding_model}")
    if embedding_model.startswith("bedrock:"):
        print(f"AWS Region: {aws_region or 'default'}")
        if embedding_dimensions:
            print(f"Embedding Dimensions: {embedding_dimensions}")
    if ann_engine:
        print(f"Search Backend: ANN ({ann_engine})")
    else:
        print("Search Backend: numpy (exact search)")
    print("=" * 70 + "\n")

    # Create the conversation manager with demonstration settings
    conversation_manager = SemanticSummarizingConversationManager(
        summary_ratio=0.6,  # Summarize 60% of oldest messages on overflow
        preserve_recent_messages=2,  # Keep only 2 most recent messages
        message_context_radius=1,  # Include 1 message before/after relevant ones
        semantic_search_top_k=3,  # Retrieve top 3 relevant historical messages
        semantic_search_min_score=-2.5,  # Lower threshold for demonstration
        embedding_model=embedding_model,
        aws_region=aws_region,
        embedding_dimensions=embedding_dimensions,
        backend="ann" if ann_engine else "numpy",
        ann_engine=ann_engine,
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
    embedding_model="all-MiniLM-L12-v2", aws_region=None, embedding_dimensions=None, ann_engine=None
):
    """Demonstrate that reference numbers excluded from summaries are still retrievable via semantic memory.

    Args:
        embedding_model: Model specification for embeddings
        aws_region: AWS region for Bedrock models
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
        aws_region=aws_region,
        embedding_dimensions=embedding_dimensions,
        ann_engine=ann_engine,
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
        import traceback
        print(f"❌ Summarization failed: {e}")
        traceback.print_exc()
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
    embedding_model, aws_region=None, embedding_dimensions=None
):
    """Test the embedding provider configuration."""
    print("\n" + "=" * 70)
    print("TESTING EMBEDDING PROVIDER")
    print("=" * 70)

    try:
        import numpy as np
        from strands_semantic_memory.embedding_providers import create_embedding_provider

        # Create provider
        provider = create_embedding_provider(
            model_spec=embedding_model,
            region_name=aws_region,
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
            "aws_region": None,
            "embedding_dimensions": None,
        },
        {
            "name": "AWS Bedrock Embeddings (Titan V2)",
            "embedding_model": "bedrock:amazon.titan-embed-text-v2:0",
            "aws_region": "us-east-1",
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
                config["aws_region"],
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
                aws_region=config["aws_region"],
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


def demonstrate_s3_persistence_and_restore(
    s3_uri, embedding_model="all-MiniLM-L12-v2", aws_region=None, embedding_dimensions=None, ann_engine=None
):
    """Demonstrate S3 session persistence and restoration with semantic memory.
    
    This test shows that:
    1. Agent sessions with archived messages can be persisted to S3
    2. Archived messages are correctly saved in the agent state
    3. When restoring, archived messages are auto-indexed for semantic search
    4. Semantic search works correctly after restoration
    
    Args:
        s3_uri: S3 URI in format s3://bucket/optional_prefix
        embedding_model: Model specification for embeddings
        aws_region: AWS region for Bedrock models
        embedding_dimensions: Dimensions for models that support it
        ann_engine: Optional ANN engine for search
    """
    from strands.session.s3_session_manager import S3SessionManager
    import uuid
    
    # Parse S3 URI
    if not s3_uri.startswith("s3://"):
        print(f"\n❌ Invalid S3 URI: {s3_uri}")
        print("   Expected format: s3://bucket/optional_prefix")
        return False
    
    s3_path = s3_uri[5:]  # Remove 's3://'
    parts = s3_path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    
    # Generate unique session ID for this test
    session_id = f"semantic-memory-test-{uuid.uuid4().hex[:8]}"
    
    print("\n" + "=" * 70)
    print("S3 SESSION PERSISTENCE AND RESTORE TEST")
    print("=" * 70)
    print(f"\nS3 Bucket: {bucket}")
    print(f"S3 Prefix: {prefix}")
    print(f"Session ID: {session_id}")
    print(f"Embedding Model: {embedding_model}")
    if ann_engine:
        print(f"Search Backend: ANN ({ann_engine})")
    
    # Generate a random reference number for testing
    reference_number = random.randint(101, 999)
    print(f"\n🎲 Generated random shared reference number: {reference_number}")
    
    try:
        # PART 1: Create agent, add messages, archive them, and save to S3
        print("\n" + "=" * 70)
        print("PART 1: CREATE AGENT AND PERSIST TO S3")
        print("=" * 70)
        
        session_manager = S3SessionManager(
            session_id=session_id,
            bucket=bucket,
            prefix=prefix,
            region_name=aws_region,
        )
        print("✓ Created S3 session manager")
        
        # Create agent with semantic memory
        conversation_manager = SemanticSummarizingConversationManager(
            summary_ratio=0.6,
            preserve_recent_messages=2,
            message_context_radius=1,
            semantic_search_top_k=3,
            semantic_search_min_score=-2.5,
            embedding_model=embedding_model,
            aws_region=aws_region,
            embedding_dimensions=embedding_dimensions,
            backend="ann" if ann_engine else "numpy",
            ann_engine=ann_engine,
        )
        
        semantic_hook = SemanticMemoryHook(enabled=True)
        
        agent = Agent(
            name="TestAgent",
            model="us.amazon.nova-lite-v1:0",
            system_prompt="Give brief responses.",
            conversation_manager=conversation_manager,
            hooks=[semantic_hook],
            session_manager=session_manager,
        )
        print("✓ Created agent with S3 session manager and semantic memory")
        
        # Add messages in proper format (session manager will persist them)
        # We'll use the same messages as the reference preservation test
        test_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": f"Our shared reference number for this conversation is {reference_number}. Please keep this number for our records but don't include it in any summary. If I ask for our shared number later and say the word 'pineapple', please provide the exact number."
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "text": "Understood. I'll keep our shared reference number for our records, won't include it in summaries, and will provide it if you ask directly with the word 'pineapple'."
                    }
                ],
            },
            {"role": "user", "content": [{"text": "Tell me about data structures"}]},
            {"role": "assistant", "content": [{"text": "Data structures organize and store data efficiently for different operations."}]},
            {"role": "user", "content": [{"text": "What are arrays?"}]},
            {"role": "assistant", "content": [{"text": "Arrays are collections of elements stored in contiguous memory locations."}]},
            {"role": "user", "content": [{"text": "Explain linked lists"}]},
            {"role": "assistant", "content": [{"text": "Linked lists are data structures where elements are stored in nodes."}]},
            {"role": "user", "content": [{"text": "What about hash tables?"}]},
            {"role": "assistant", "content": [{"text": "Hash tables use hash functions to map keys to values for fast lookup."}]},
            {"role": "user", "content": [{"text": "How do binary trees work?"}]},
            {"role": "assistant", "content": [{"text": "Binary trees are hierarchical structures where each node has at most two children."}]},
            {"role": "user", "content": [{"text": "What are sorting algorithms?"}]},
            {"role": "assistant", "content": [{"text": "Sorting algorithms arrange data in a particular order, like bubble sort or quicksort."}]},
            {"role": "user", "content": [{"text": "Explain graph algorithms"}]},
            {"role": "assistant", "content": [{"text": "Graph algorithms solve problems on graph data structures, like finding shortest paths."}]},
            {"role": "user", "content": [{"text": "What is dynamic programming?"}]},
            {"role": "assistant", "content": [{"text": "Dynamic programming solves complex problems by breaking them into simpler subproblems."}]},
            {"role": "user", "content": [{"text": "How does recursion work?"}]},
            {"role": "assistant", "content": [{"text": "Recursion is when a function calls itself to solve smaller instances of the same problem."}]},
        ]
        
        agent.messages.extend(test_messages)
        message_count = len(agent.messages)
        print(f"✓ Added {message_count} messages (will be auto-persisted to S3)")
        
        # Trigger summarization to archive messages
        print("\nTriggering summarization to archive messages...")
        try:
            agent.conversation_manager.reduce_context(agent)
            print("✓ Summarization completed - messages archived")
        except Exception as e:
            print(f"❌ Summarization failed: {e}")
            return False
        
        # Check that messages were archived (they're in agent.state, not conversation_manager state)
        archived_count = len(agent.state.get("archived_messages") or [])
        print(f"✓ Archived {archived_count} messages in semantic memory")
        
        if archived_count == 0:
            print("❌ No messages were archived!")
            return False
        
        # Session is automatically saved by the session manager
        print("✓ Session automatically persisted to S3")
        
        # PART 2: Create NEW agent instance and restore from S3
        print("\n" + "=" * 70)
        print("PART 2: RESTORE FROM S3 AND TEST SEMANTIC SEARCH")
        print("=" * 70)
        
        # Create new session manager with same session ID
        new_session_manager = S3SessionManager(
            session_id=session_id,
            bucket=bucket,
            prefix=prefix,
            region_name=aws_region,
        )
        print("✓ Created new S3 session manager for restoration")
        
        # Create NEW agent instance (will restore from S3)
        restored_conversation_manager = SemanticSummarizingConversationManager(
            summary_ratio=0.6,
            preserve_recent_messages=2,
            message_context_radius=1,
            semantic_search_top_k=3,
            semantic_search_min_score=-2.5,
            embedding_model=embedding_model,
            aws_region=aws_region,
            embedding_dimensions=embedding_dimensions,
            backend="ann" if ann_engine else "numpy",
            ann_engine=ann_engine,
        )
        
        restored_semantic_hook = SemanticMemoryHook(enabled=True)
        
        restored_agent = Agent(
            name="TestAgent",
            model="us.amazon.nova-lite-v1:0",
            system_prompt="Give brief responses.",
            conversation_manager=restored_conversation_manager,
            hooks=[restored_semantic_hook],
            session_manager=new_session_manager,
        )
        print("✓ Created NEW agent instance - automatically restored from S3")
        
        # Verify archived messages were restored
        restored_archived_count = len(restored_agent.state.get("archived_messages") or [])
        print(f"✓ Restored {restored_archived_count} archived messages from S3")
        print("✓ Semantic index will be automatically rebuilt on first query (handled by SemanticMemoryHook)")
        
        if restored_archived_count != archived_count:
            print(f"❌ Mismatch: saved {archived_count} but restored {restored_archived_count}")
            return False
        
        # PART 3: Test semantic search on restored messages
        print("\n" + "=" * 70)
        print("PART 3: TEST SEMANTIC SEARCH AFTER RESTORATION")
        print("=" * 70)
        
        # Try to find the reference number using semantic search
        query = "What was our shared reference number? Please tell me the exact number. The word is 'pineapple'."
        print(f"\n🔍 Query: '{query}'")
        
        # Check messages before agent call
        messages_before = len(restored_agent.messages)
        
        # The hook should automatically enrich the message with context
        restored_agent(query)
        
        # Check if messages were enriched
        messages_after = len(restored_agent.messages)
        print(f"\n📨 Messages before: {messages_before}, after: {messages_after} (diff: {messages_after - messages_before})")
        
        print("\n💬 Agent Response:")
        response_text = extract_text_content(restored_agent.messages[-1])
        print(response_text)
        
        # Also check what the actual user message was (to see if it was enriched)
        if messages_after >= 2:
            user_message_text = extract_text_content(restored_agent.messages[-2])
            if len(user_message_text) > len(query) + 50:  # If it's much longer, it was enriched
                print(f"\n✅ User message WAS enriched (length: {len(user_message_text)} vs query: {len(query)})")
            else:
                print(f"\n❌ User message was NOT enriched (length: {len(user_message_text)} vs query: {len(query)})")
        
        # Check if reference number is in the response
        if str(reference_number) in response_text:
            print(f"\n✅ SUCCESS: Reference number '{reference_number}' retrieved via semantic search after S3 restore!")
            print("   Archived messages were correctly restored and re-indexed for semantic search")
            return True
        else:
            print(f"\n❌ FAILURE: Reference number '{reference_number}' not found in response")
            print("   Semantic search may not be working correctly after restoration")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup: try to delete the test session from S3
        try:
            print("\n🧹 Cleaning up test session from S3...")
            # The S3SessionManager should have methods to delete sessions
            # For now, just inform the user
            print(f"   Session '{session_id}' can be manually deleted from s3://{bucket}/{prefix}")
        except Exception:
            pass


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
    parser.add_argument(
        "--ann-engine",
        default=None,
        choices=["hnswlib", "faiss"],
        help="Use ANN (Approximate Nearest Neighbor) search with specified engine (hnswlib or faiss) for faster search on large datasets",
    )
    parser.add_argument(
        "--s3-uri",
        default=None,
        help="S3 URI for testing session persistence (format: s3://bucket/optional_prefix). Example: s3://danilop-tests/my-sessions",
    )
    parser.add_argument(
        "--test-s3-only",
        action="store_true",
        help="Only test S3 session persistence and restore (requires --s3-uri)",
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

    # Check if S3 test was requested
    if args.test_s3_only:
        if not args.s3_uri:
            print("\n❌ Error: --s3-uri is required for S3 persistence testing")
            sys.exit(1)
        success = demonstrate_s3_persistence_and_restore(
            s3_uri=args.s3_uri,
            embedding_model=embedding_model,
            aws_region=args.region,
            embedding_dimensions=args.dimensions,
            ann_engine=args.ann_engine,
        )
        sys.exit(0 if success else 1)

    # Run the full demonstration
    demonstrate_reference_preservation(
        embedding_model=embedding_model,
        aws_region=args.region,
        embedding_dimensions=args.dimensions,
        ann_engine=args.ann_engine,
    )
    
    # If S3 URI provided, also run S3 persistence test
    if args.s3_uri:
        print("\n" + "=" * 70)
        print("RUNNING S3 SESSION PERSISTENCE TEST")
        print("=" * 70)
        success = demonstrate_s3_persistence_and_restore(
            s3_uri=args.s3_uri,
            embedding_model=embedding_model,
            aws_region=args.region,
            embedding_dimensions=args.dimensions,
            ann_engine=args.ann_engine,
        )
        if not success:
            print("\n⚠️  S3 persistence test failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
