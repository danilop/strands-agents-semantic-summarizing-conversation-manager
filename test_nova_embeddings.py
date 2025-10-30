#!/usr/bin/env python3
"""
Quick test to verify Nova multimodal embeddings work correctly.
"""

import sys
from embedding_providers import create_embedding_provider
import numpy as np


def test_nova_embeddings():
    """Test Nova multimodal embeddings."""
    print("Testing Nova Multimodal Embeddings")
    print("=" * 60)
    
    # Test 1: Using the alias
    print("\n1. Testing with alias 'nova-multimodal-embeddings'...")
    try:
        provider = create_embedding_provider(
            model_spec="bedrock:nova-multimodal-embeddings",
            region_name="us-east-1",
            dimensions=1024
        )
        
        info = provider.get_model_info()
        print("   ✓ Provider created successfully")
        print(f"   ✓ Model ID: {info.model_id}")
        print(f"   ✓ Dimensions: {info.dimensions}")
        print(f"   ✓ Max tokens: {info.max_tokens}")
        
        # Test encoding
        test_text = "Amazon Nova is a multimodal foundation model"
        print(f"\n   Testing encoding with text: '{test_text[:50]}...'")
        embeddings = provider.encode(test_text)
        print(f"   ✓ Embedding shape: {embeddings.shape}")
        print(f"   ✓ Embedding dtype: {embeddings.dtype}")
        
        # Check if embeddings are normalized
        norm = np.linalg.norm(embeddings)
        print(f"   ✓ Embedding norm: {norm:.6f}")
        
        print("\n   ✅ Test 1 PASSED\n")
        
    except Exception as e:
        print(f"\n   ❌ Test 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Using the full model ID
    print("2. Testing with full model ID...")
    try:
        provider = create_embedding_provider(
            model_spec="bedrock:amazon.nova-2-multimodal-embeddings-v1:0",
            region_name="us-east-1",
            dimensions=384
        )
        
        info = provider.get_model_info()
        print("   ✓ Provider created successfully")
        print(f"   ✓ Model ID: {info.model_id}")
        print(f"   ✓ Dimensions: {info.dimensions}")
        
        # Test encoding
        test_texts = [
            "First test sentence.",
            "Second test sentence with different content."
        ]
        print(f"\n   Testing batch encoding with {len(test_texts)} texts...")
        embeddings = provider.encode(test_texts)
        print(f"   ✓ Embedding shape: {embeddings.shape}")
        print(f"   ✓ Expected shape: ({len(test_texts)}, 384)")
        
        if embeddings.shape == (len(test_texts), 384):
            print("   ✓ Shape matches expected dimensions")
        
        print("\n   ✅ Test 2 PASSED\n")
        
    except Exception as e:
        print(f"\n   ❌ Test 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 60)
    print("✅ All tests PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_nova_embeddings()
    sys.exit(0 if success else 1)
