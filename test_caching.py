#!/usr/bin/env python3
"""
Test script to demonstrate caching behavior within a single session.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bilateral_truth import Assertion, zeta_c, get_cache_size, ModelRouter

def main():
    # Load .env file
    from zeta_c.cli import load_env_file
    load_env_file()
    
    print("Testing caching behavior with OpenRouter Llama model")
    print("=" * 50)
    
    evaluator = ModelRouter.create_evaluator('llama3')
    assertion = Assertion("The sky is blue")
    
    print(f"Initial cache size: {get_cache_size()}")
    
    print(f"\nFirst evaluation of: '{assertion}'")
    result1 = zeta_c(assertion, evaluator.evaluate_bilateral)
    print(f"Result: {result1}")
    print(f"Cache size after first evaluation: {get_cache_size()}")
    
    print(f"\nSecond evaluation of same assertion:")
    result2 = zeta_c(assertion, evaluator.evaluate_bilateral)
    print(f"Result: {result2}")
    print(f"Cache size after second evaluation: {get_cache_size()}")
    print(f"Results identical: {result1 == result2}")

if __name__ == "__main__":
    main()