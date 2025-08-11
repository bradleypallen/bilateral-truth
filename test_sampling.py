#!/usr/bin/env python3
"""
Test script to demonstrate sampling and majority voting functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bilateral_truth import Assertion, GeneralizedTruthValue, TruthValueComponent, MockLLMEvaluator
import random

def create_variable_mock_evaluator():
    """Create a mock evaluator that gives variable responses to simulate real LLM variance."""
    
    def variable_evaluation(assertion: Assertion) -> GeneralizedTruthValue:
        # Simulate variability in LLM responses
        outcomes = [
            GeneralizedTruthValue(TruthValueComponent.TRUE, TruthValueComponent.FALSE),   # <t,f>
            GeneralizedTruthValue(TruthValueComponent.TRUE, TruthValueComponent.UNDEFINED),   # <t,e> 
            GeneralizedTruthValue(TruthValueComponent.UNDEFINED, TruthValueComponent.FALSE),  # <e,f>
        ]
        
        # Weight toward <t,f> for "obvious" truths
        if "sky is blue" in assertion.predicate.lower():
            weights = [0.6, 0.3, 0.1]
        else:
            weights = [0.4, 0.4, 0.2]
        
        return random.choices(outcomes, weights=weights)[0]
    
    # Create a mock evaluator with the variable function
    mock = MockLLMEvaluator()
    mock._single_evaluation = variable_evaluation
    return mock

def test_majority_voting():
    """Test majority voting with different sample sizes."""
    print("Testing Majority Voting with Variable Mock Evaluator")
    print("=" * 60)
    
    evaluator = create_variable_mock_evaluator()
    assertion = Assertion("The sky is blue")
    
    sample_sizes = [1, 3, 5, 7]
    tiebreak_strategies = ["random", "pessimistic", "optimistic"]
    
    for samples in sample_sizes:
        print(f"\nTesting with {samples} sample(s):")
        print("-" * 30)
        
        if samples == 1:
            # Single sample
            result = evaluator.evaluate_bilateral(assertion, samples)
            print(f"Single result: {result}")
        else:
            # Multiple samples with different tiebreaking
            for strategy in tiebreak_strategies:
                result = evaluator.evaluate_with_majority_voting(assertion, samples, strategy)
                print(f"{strategy.capitalize()} tiebreak: {result}")

def demonstrate_tiebreaking():
    """Demonstrate tiebreaking with controlled results."""
    print("\n\nDemonstrating Tiebreaking Strategies")
    print("=" * 60)
    
    # Create controlled results that will tie
    tied_results = [
        GeneralizedTruthValue(TruthValueComponent.TRUE, TruthValueComponent.FALSE),   # <t,f>
        GeneralizedTruthValue(TruthValueComponent.UNDEFINED, TruthValueComponent.UNDEFINED), # <e,e>
        GeneralizedTruthValue(TruthValueComponent.TRUE, TruthValueComponent.FALSE),   # <t,f> 
        GeneralizedTruthValue(TruthValueComponent.UNDEFINED, TruthValueComponent.UNDEFINED), # <e,e>
    ]
    
    evaluator = MockLLMEvaluator()
    
    # Manually test majority voting logic
    for strategy in ["random", "pessimistic", "optimistic"]:
        result = evaluator._majority_vote(tied_results, strategy)
        print(f"{strategy.capitalize()} tiebreak result: {result}")
        
        # Show the vote breakdown
        u_votes = [r.u for r in tied_results]
        v_votes = [r.v for r in tied_results]
        print(f"  Verifiability votes: {[v.value for v in u_votes]} -> {result.u.value}")
        print(f"  Refutability votes: {[v.value for v in v_votes]} -> {result.v.value}")
        print()

def main():
    # Set random seed for reproducible testing
    random.seed(42)
    
    # Load .env file
    from bilateral_truth.cli import load_env_file
    load_env_file()
    
    test_majority_voting()
    demonstrate_tiebreaking()
    
    print("\nSampling and voting functionality test completed!")

if __name__ == "__main__":
    main()