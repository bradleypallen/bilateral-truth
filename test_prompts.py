#!/usr/bin/env python3
"""
Test script to demonstrate the actual prompts being sent to LLMs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bilateral_truth import Assertion, ModelRouter

def show_prompt_examples():
    """Show what prompts look like for different types of assertions."""
    
    # Load environment
    from zeta_c.cli import load_env_file
    load_env_file()
    
    print("Bilateral Evaluation Prompts Being Sent to LLMs")
    print("=" * 60)
    
    # Create a mock evaluator to access the prompt generation
    evaluator = ModelRouter.create_evaluator('mock')
    
    # Test different types of assertions
    test_assertions = [
        "The sky is blue",
        "The capital of France is Paris", 
        "Climate change is caused by human activities",
        "There is life on other planets",
        "2 + 2 = 5",
        "Democracy is the best form of government",
        "Artificial intelligence will be beneficial for humanity"
    ]
    
    for i, assertion_text in enumerate(test_assertions, 1):
        print(f"\n{i}. ASSERTION: '{assertion_text}'")
        print("-" * 50)
        
        assertion = Assertion(assertion_text)
        prompt = evaluator._create_prompt(assertion)
        print("PROMPT SENT TO LLM:")
        print(prompt)
        
        if i < len(test_assertions):
            print("\n" + "="*60)

def show_prompt_structure():
    """Analyze the structure of the prompt."""
    print("\nPrompt Structure Analysis")
    print("=" * 30)
    
    evaluator = ModelRouter.create_evaluator('mock')
    sample_assertion = Assertion("The Earth is round")
    prompt = evaluator._create_prompt(sample_assertion)
    
    lines = prompt.strip().split('\n')
    
    print("Prompt Components:")
    print("1. Task Definition:", "✓" if "bilateral factuality evaluation" in prompt else "✗")
    print("2. Component Definitions:", "✓" if "VERIFIABILITY" in prompt and "REFUTABILITY" in prompt else "✗")
    print("3. Response Scale:", "✓" if "t|e|f" in prompt else "✗")
    print("4. JSON Format:", "✓" if "JSON format" in prompt else "✗")
    print("5. Examples:", "✓" if "example" in prompt.lower() else "✗")
    print("6. Reasoning Field:", "✓" if "reasoning" in prompt else "✗")
    
    print(f"\nPrompt Length: {len(prompt)} characters")
    print(f"Number of Lines: {len(lines)}")

def main():
    show_prompt_examples()
    show_prompt_structure()

if __name__ == "__main__":
    main()