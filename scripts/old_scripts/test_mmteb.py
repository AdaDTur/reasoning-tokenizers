#!/usr/bin/env python3
"""
Test script to debug MTEB evaluation issues
"""

import os
import sys
import traceback
import mteb

def test_mteb_installation():
    """Test if MTEB is properly installed and can load tasks"""
    print("Testing MTEB installation...")
    try:
        # Try to get a simple task
        tasks = mteb.get_tasks(tasks=["STS12"])
        print(f"Successfully loaded {len(tasks)} tasks")
        for task in tasks:
            print(f"  - {task.__class__.__name__}")
        return True
    except Exception as e:
        print(f"Error loading MTEB tasks: {e}")
        traceback.print_exc()
        return False

def test_embedder_creation():
    """Test if we can create the embedder"""
    print("\nTesting embedder creation...")
    try:
        from mmteb import Tok, _load_checkpoint, _init_model_from_checkpoint, CustomGPTEmbedder
        
        tokenizer_json = os.environ.get("TOKENIZER_JSON", os.path.join(".gitignore", "tokenizers", "bpe", "en", "tokenizer.json"))
        ckpt_path = os.environ.get("CKPT_PATH", os.path.join(".gitignore", "out", "bpe_en_train.pt"))
        
        if not os.path.exists(tokenizer_json):
            print(f"Tokenizer file not found: {tokenizer_json}")
            return False
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint file not found: {ckpt_path}")
            return False
            
        tok = Tok(tokenizer_json)
        ckpt = _load_checkpoint(ckpt_path, map_location="cpu")
        model, wte_key, block_size = _init_model_from_checkpoint(ckpt, device="cpu")
        embedder = CustomGPTEmbedder(model=model, tok=tok, wte_key=wte_key, device="cpu", block_size=block_size)
        
        print("Successfully created embedder")
        return True
    except Exception as e:
        print(f"Error creating embedder: {e}")
        traceback.print_exc()
        return False

def test_simple_encoding():
    """Test if the embedder can encode simple text"""
    print("\nTesting simple encoding...")
    try:
        from mmteb import Tok, _load_checkpoint, _init_model_from_checkpoint, CustomGPTEmbedder
        
        tokenizer_json = os.environ.get("TOKENIZER_JSON", os.path.join(".gitignore", "tokenizers", "bpe", "en", "tokenizer.json"))
        ckpt_path = os.environ.get("CKPT_PATH", os.path.join(".gitignore", "out", "bpe_en_train.pt"))
        
        tok = Tok(tokenizer_json)
        ckpt = _load_checkpoint(ckpt_path, map_location="cpu")
        model, wte_key, block_size = _init_model_from_checkpoint(ckpt, device="cpu")
        embedder = CustomGPTEmbedder(model=model, tok=tok, wte_key=wte_key, device="cpu", block_size=block_size)
        
        # Test encoding
        test_texts = ["Hello world", "This is a test", ""]
        embeddings = embedder.encode(test_texts)
        print(f"Successfully encoded {len(test_texts)} texts, got embeddings of shape {embeddings.shape}")
        return True
    except Exception as e:
        print(f"Error in simple encoding: {e}")
        traceback.print_exc()
        return False

def main():
    print("MTEB Debug Test Script")
    print("=" * 50)
    
    # Test 1: MTEB installation
    if not test_mteb_installation():
        print("❌ MTEB installation test failed")
        return 1
    
    # Test 2: Embedder creation
    if not test_embedder_creation():
        print("❌ Embedder creation test failed")
        return 1
    
    # Test 3: Simple encoding
    if not test_simple_encoding():
        print("❌ Simple encoding test failed")
        return 1
    
    print("\n✅ All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())






















