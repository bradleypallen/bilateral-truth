#!/usr/bin/env python3
"""
Example usage of the bilateral-truth function with LLM-based bilateral evaluation.

This demonstrates how to use language models to perform proper bilateral
factuality evaluation as described in the research paper.
"""

from bilateral_truth import (
    Assertion, 
    GeneralizedTruthValue, 
    TruthValueComponent,
    zeta, 
    zeta_c, 
    create_llm_evaluator,
    MockLLMEvaluator,
    clear_cache, 
    get_cache_size
)


def demo_mock_llm_evaluation():
    """Demonstrate bilateral evaluation using mock LLM evaluator."""
    print("=== Mock LLM Bilateral Evaluation Demo ===")
    
    # Create mock evaluator with some predefined responses
    predefined_responses = {
        "The sky is blue": GeneralizedTruthValue(TruthValueComponent.TRUE, TruthValueComponent.FALSE),
        "It is raining": GeneralizedTruthValue(TruthValueComponent.TRUE, TruthValueComponent.TRUE),
        "Alice loves Bob": GeneralizedTruthValue(TruthValueComponent.UNDEFINED, TruthValueComponent.UNDEFINED),
    }
    
    mock_evaluator = MockLLMEvaluator(predefined_responses)
    
    # Create test assertions
    assertions = [
        Assertion("The sky is blue"),
        Assertion("It is raining"), 
        Assertion("Alice loves Bob"),
        Assertion("The temperature is 72°F"),  # Will use mock logic
        Assertion("sunny"),  # Will use mock logic
        Assertion("cloudy")  # Will use mock logic
    ]
    
    clear_cache()
    print(f"Starting with empty cache (size: {get_cache_size()})")
    print()
    
    for assertion in assertions:
        result = zeta_c(assertion, mock_evaluator.evaluate_bilateral)
        print(f"Formula: {assertion}")
        print(f"zeta_c result: {result}")
        print(f"  Verifiability (u): {result.u.value}")
        print(f"  Refutability (v): {result.v.value}")
        print()
    
    print(f"Final cache size: {get_cache_size()}")
    print()


def demo_real_llm_setup():
    """Show how to set up real LLM evaluators (requires API keys)."""
    print("=== Real LLM Evaluator Setup ===")
    print("To use real LLM evaluation, you need to:")
    print("1. Set environment variables:")
    print("   export OPENAI_API_KEY='your-openai-key'")
    print("   export ANTHROPIC_API_KEY='your-anthropic-key'")
    print()
    print("2. Create evaluators:")
    print("   openai_evaluator = create_llm_evaluator('openai', model='gpt-4')")
    print("   claude_evaluator = create_llm_evaluator('anthropic', model='claude-3-sonnet-20240229')")
    print()
    print("3. Use with zeta_c:")
    print("   assertion = Assertion('The capital of France is Paris')")
    print("   result = zeta_c(assertion, openai_evaluator.evaluate_bilateral)")
    print()
    
    # Try to create a real evaluator (will fail without API key, but show the pattern)
    try:
        openai_evaluator = create_llm_evaluator('openai')
        print("✓ OpenAI evaluator created successfully")
        
        # Example usage (commented out to avoid API calls without proper setup)
        # assertion = Assertion("The capital of France is Paris")
        # result = zeta_c(assertion, openai_evaluator.evaluate_bilateral)
        # print(f"Result: {result}")
        
    except Exception as e:
        print(f"✗ OpenAI evaluator setup failed: {e}")
        print("  (This is expected without proper API key setup)")
    
    print()


def demo_bilateral_evaluation_concepts():
    """Explain bilateral evaluation with examples."""
    print("=== Bilateral Evaluation Concepts ===")
    print("Bilateral evaluation assesses two independent dimensions:")
    print()
    print("VERIFIABILITY (u): Can we find evidence that supports this statement?")
    print("REFUTABILITY (v): Can we find evidence that contradicts this statement?")
    print()
    print("Examples of different combinations:")
    print()
    
    examples = [
        ("Classical Truth <t,f>", "Strong evidence supports it, no evidence against it"),
        ("Classical False <f,t>", "No evidence supports it, strong evidence against it"),
        ("Unknown <e,e>", "Insufficient evidence either way"),
        ("Controversial <t,t>", "Evidence both for and against"),
        ("Unverifiable True <e,f>", "Can't verify, but no contradicting evidence"),
        ("Unrefutable False <f,e>", "No supporting evidence, but can't refute")
    ]
    
    for pattern, description in examples:
        print(f"{pattern}: {description}")
    
    print()


def demo_caching_with_llm():
    """Demonstrate caching behavior with LLM evaluation."""
    print("=== Caching with LLM Evaluation ===")
    
    # Create a mock evaluator that tracks calls
    call_count = 0
    def counting_evaluator(assertion):
        nonlocal call_count
        call_count += 1
        print(f"  LLM call #{call_count} for: {assertion}")
        # Return a simple result
        return GeneralizedTruthValue.unknown()
    
    clear_cache()
    assertion = Assertion("The Earth is round")
    
    print("First evaluation (should trigger LLM call):")
    result1 = zeta_c(assertion, counting_evaluator)
    print(f"Result: {result1}")
    print(f"Cache size: {get_cache_size()}")
    print()
    
    print("Second evaluation (should use cache):")
    result2 = zeta_c(assertion, counting_evaluator)
    print(f"Result: {result2}")
    print(f"Cache size: {get_cache_size()}")
    print(f"Total LLM calls: {call_count}")
    print(f"Results identical: {result1 == result2}")
    print()


def demo_assertion_to_natural_language():
    """Show how atomic assertions can represent natural language statements."""
    print("=== Assertions as Natural Language ===")
    print("Assertions can represent natural language statements:")
    print()
    
    assertions = [
        Assertion("The capital of France is Paris"),
        Assertion("climate_change", "exists"),
        Assertion("temperature", location="New York", value=25, unit="Celsius"),
        Assertion("loves", "Alice", "Bob"),
        Assertion("It will rain tomorrow"),
        Assertion("The stock market will go up next week"),
        Assertion("COVID-19 vaccines are effective")
    ]
    
    mock_evaluator = MockLLMEvaluator()
    
    for assertion in assertions:
        result = zeta(assertion, mock_evaluator.evaluate_bilateral)
        print(f"'{assertion}' → {result}")
    
    print()
    print("In practice, these would be evaluated by prompting an LLM with:")
    print("1. The natural language statement")
    print("2. Instructions for bilateral assessment")
    print("3. Request for verifiability and refutability judgments")
    print()


def main():
    """Run all demonstrations."""
    print("bilateral-truth LLM-Based Bilateral Evaluation Examples")
    print("=" * 60)
    print()
    
    demo_bilateral_evaluation_concepts()
    demo_mock_llm_evaluation()
    demo_caching_with_llm()
    demo_assertion_to_natural_language()
    demo_real_llm_setup()
    
    print("LLM examples completed!")
    print()
    print("Next steps:")
    print("1. Set up API keys for OpenAI or Anthropic")
    print("2. Test with real LLM evaluation")
    print("3. Customize prompts for your specific domain")


if __name__ == "__main__":
    main()