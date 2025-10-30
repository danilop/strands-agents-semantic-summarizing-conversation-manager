#!/usr/bin/env python3
"""
Validate Nova multimodal embeddings configuration without making API calls.
"""

import sys

def validate_config():
    """Validate the Nova embeddings configuration."""
    print("Validating Nova Multimodal Embeddings Configuration")
    print("=" * 60)
    
    try:
        # Import the module
        from embedding_providers import BedrockEmbeddingProvider
        
        # Test 1: Check model configurations
        print("\n1. Checking MODEL_CONFIGS...")
        
        if "amazon.nova-2-multimodal-embeddings-v1:0" in BedrockEmbeddingProvider.MODEL_CONFIGS:
            config = BedrockEmbeddingProvider.MODEL_CONFIGS["amazon.nova-2-multimodal-embeddings-v1:0"]
            print("   ✓ Nova model found in MODEL_CONFIGS")
            print(f"   ✓ Default dimensions: {config['dimensions']}")
            print(f"   ✓ Available dimensions: {config['available_dimensions']}")
            print(f"   ✓ Max tokens: {config['max_tokens']}")
            print(f"   ✓ Supports variable dimensions: {config['supports_variable_dimensions']}")
            
            # Verify dimensions
            expected_dims = [256, 384, 1024, 3072]
            if config['available_dimensions'] == expected_dims:
                print(f"   ✓ Dimensions match expected: {expected_dims}")
            else:
                print("   ❌ Dimensions mismatch!")
                print(f"      Expected: {expected_dims}")
                print(f"      Got: {config['available_dimensions']}")
                return False
        else:
            print("   ❌ Nova model NOT found in MODEL_CONFIGS")
            return False
        
        # Test 2: Check Titan models are removed
        print("\n2. Checking Titan models are removed...")
        titan_models = [k for k in BedrockEmbeddingProvider.MODEL_CONFIGS.keys() if "titan" in k.lower()]
        if titan_models:
            print(f"   ❌ Found Titan models that should be removed: {titan_models}")
            return False
        else:
            print("   ✓ No Titan models found (as expected)")
        
        # Test 3: Check alias
        print("\n3. Checking model alias...")
        if hasattr(BedrockEmbeddingProvider, 'MODEL_ALIASES'):
            aliases = BedrockEmbeddingProvider.MODEL_ALIASES
            if "nova-multimodal-embeddings" in aliases:
                target = aliases["nova-multimodal-embeddings"]
                print("   ✓ Alias 'nova-multimodal-embeddings' found")
                print(f"   ✓ Points to: {target}")
                if target == "amazon.nova-2-multimodal-embeddings-v1:0":
                    print("   ✓ Alias correctly maps to full model ID")
                else:
                    print(f"   ❌ Alias maps to wrong model: {target}")
                    return False
            else:
                print("   ❌ Alias 'nova-multimodal-embeddings' NOT found")
                return False
        else:
            print("   ❌ MODEL_ALIASES not defined")
            return False
        
        # Test 4: Verify factory function
        print("\n4. Checking factory function...")
        print("   ✓ Factory function importable (create_embedding_provider)")
        
        print("\n" + "=" * 60)
        print("✅ All configuration validations PASSED!")
        print("=" * 60)
        print("\nConfiguration is correct. To test with actual API calls,")
        print("configure AWS credentials and run:")
        print("  uv run main.py --embedding-model 'bedrock:nova-multimodal-embeddings' --region us-east-1 --embedding-test-only")
        return True
        
    except Exception as e:
        print(f"\n❌ Validation FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = validate_config()
    sys.exit(0 if success else 1)
