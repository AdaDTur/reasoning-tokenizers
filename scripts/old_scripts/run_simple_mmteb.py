#!/usr/bin/env python3
"""
Simple MTEB runner with minimal tasks to test error handling
"""

import os
import sys
from mmteb import main as mmteb_main

def run_simple_test():
    """Run MTEB with minimal configuration"""
    print("Running simple MTEB test...")
    
    # Set minimal environment variables
    os.environ.setdefault("MMTEB_TASKS", "STS12")  # Use a simple, well-known task
    os.environ.setdefault("MMTEB_LANGS", "en")     # English only
    os.environ.setdefault("EMBED_BATCH", "16")     # Small batch size
    os.environ.setdefault("MMTEB_OUT", "results/simple_test")
    
    print("Environment variables set:")
    print(f"  MMTEB_TASKS: {os.environ.get('MMTEB_TASKS')}")
    print(f"  MMTEB_LANGS: {os.environ.get('MMTEB_LANGS')}")
    print(f"  EMBED_BATCH: {os.environ.get('EMBED_BATCH')}")
    print(f"  MMTEB_OUT: {os.environ.get('MMTEB_OUT')}")
    
    try:
        mmteb_main()
        print("✅ Simple MTEB test completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Simple MTEB test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_simple_test())






















