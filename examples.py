#!/usr/bin/env python3
"""
Example usage of the bilateral-truth caching bilateral factuality evaluation function.

This script demonstrates how to use the bilateral_truth package to evaluate atomic assertions
with generalized truth values and caching.
"""

from bilateral_truth import Assertion, GeneralizedTruthValue, TruthValueComponent, zeta, zeta_c, clear_cache, get_cache_size


def basic_usage_example():
    """Demonstrate basic usage of the bilateral_truth function."""
    print("=== Basic Usage Example ===")
    
    # Create some atomic assertions
    assertion1 = Assertion("true")
    assertion2 = Assertion("false")
    assertion3 = Assertion("undefined")
    assertion4 = Assertion("loves", "alice", "bob")
    
    print(f"Formula 1: {assertion1}")
    print(f"Formula 2: {assertion2}")
    print(f"Formula 3: {assertion3}")
    print(f"Formula 4: {assertion4}")
    
    # Evaluate using zeta_c (with caching)
    result1 = zeta_c(assertion1)
    result2 = zeta_c(assertion2)
    result3 = zeta_c(assertion3)
    result4 = zeta_c(assertion4)
    
    print(f"\nEvaluation results:")
    print(f"zeta_c({assertion1}) = {result1}")
    print(f"zeta_c({assertion2}) = {result2}")
    print(f"zeta_c({assertion3}) = {result3}")
    print(f"zeta_c({assertion4}) = {result4}")
    
    print(f"\nCache size: {get_cache_size()}")
    print()


def caching_demonstration():
    """Demonstrate the caching behavior of zeta_c."""
    print("=== Caching Demonstration ===")
    
    clear_cache()
    print(f"Initial cache size: {get_cache_size()}")
    
    assertion = Assertion("test_predicate", "arg1", "arg2")
    
    # First evaluation
    print(f"First evaluation of {assertion}")
    result1 = zeta_c(assertion)
    print(f"Result: {result1}")
    print(f"Cache size after first evaluation: {get_cache_size()}")
    
    # Second evaluation (should use cache)
    print(f"Second evaluation of {assertion}")
    result2 = zeta_c(assertion)
    print(f"Result: {result2}")
    print(f"Cache size after second evaluation: {get_cache_size()}")
    
    print(f"Results are identical: {result1 == result2}")
    print()


def custom_evaluator_example():
    """Demonstrate using a custom evaluator function."""
    print("=== Custom Evaluator Example ===")
    
    def domain_specific_evaluator(assertion):
        """Custom evaluator for a specific domain."""
        predicate = assertion.predicate.lower()
        
        if predicate == "sunny":
            return GeneralizedTruthValue.true()
        elif predicate == "raining":
            return GeneralizedTruthValue.false()
        elif predicate.startswith("temperature"):
            # For temperature predicates, check arguments
            if assertion.args and len(assertion.args) > 0:
                temp = assertion.args[0]
                if isinstance(temp, (int, float)) and temp > 20:
                    return GeneralizedTruthValue.true()  # Warm
                else:
                    return GeneralizedTruthValue.false()  # Cold
        
        # Default to undefined
        return GeneralizedTruthValue.undefined()
    
    # Test assertions
    assertions = [
        Assertion("sunny"),
        Assertion("raining"),
        Assertion("temperature", 25),
        Assertion("temperature", 15),
        Assertion("humidity", 60)
    ]
    
    print("Using custom domain-specific evaluator:")
    for assertion in assertions:
        result = zeta(assertion, domain_specific_evaluator)
        cached_result = zeta_c(assertion, domain_specific_evaluator)
        print(f"ζ({assertion}) = {result}")
        print(f"zeta_c({assertion}) = {cached_result}")
        print(f"Results match: {result == cached_result}")
        print()


def generalized_truth_values_example():
    """Demonstrate working with generalized truth values."""
    print("=== Generalized Truth Values Example ===")
    
    # Create different types of truth values
    classical_true = GeneralizedTruthValue.true()
    classical_false = GeneralizedTruthValue.false()
    undefined = GeneralizedTruthValue.undefined()
    custom = GeneralizedTruthValue(TruthValueComponent.TRUE, TruthValueComponent.UNDEFINED)
    
    print("Different generalized truth values:")
    print(f"Classical true (verifiable, not refutable): {classical_true}")
    print(f"Classical false (not verifiable, refutable): {classical_false}")
    print(f"Undefined (undefined verifiability, undefined refutability): {undefined}")
    print(f"Custom (verifiable, undefined refutability): {custom}")
    
    print(f"\nComponents of classical true: u={classical_true.u.value}, v={classical_true.v.value}")
    print(f"Components of classical false: u={classical_false.u.value}, v={classical_false.v.value}")
    print(f"Components of undefined: u={undefined.u.value}, v={undefined.v.value}")
    print()


def complex_assertions_example():
    """Demonstrate working with complex atomic assertions."""
    print("=== Complex Formulas Example ===")
    
    # Create assertions with various argument patterns
    assertions = [
        Assertion("P"),  # Simple predicate
        Assertion("loves", "alice", "bob"),  # Binary relation
        Assertion("between", "point", "A", "B"),  # Ternary relation
        Assertion("distance", start="Alice", end="Bob", value=5.5, unit="km"),  # Named args
        Assertion("temperature", 23.5, location="New York", scale="celsius"),  # Mixed args
    ]
    
    clear_cache()
    print(f"Starting with empty cache (size: {get_cache_size()})")
    
    for i, assertion in enumerate(assertions, 1):
        result = zeta_c(assertion)
        print(f"{i}. {assertion} → {result}")
    
    print(f"\nFinal cache size: {get_cache_size()}")
    
    # Demonstrate that identical assertions are cached
    duplicate = Assertion("loves", "alice", "bob")  # Same as assertions[1]
    print(f"\nEvaluating duplicate assertion: {duplicate}")
    result_duplicate = zeta_c(duplicate)
    print(f"Result: {result_duplicate}")
    print(f"Cache size after duplicate (should be unchanged): {get_cache_size()}")
    print()


def main():
    """Run all example demonstrations."""
    print("bilateral-truth Function Examples")
    print("=" * 50)
    
    basic_usage_example()
    caching_demonstration()
    custom_evaluator_example()
    generalized_truth_values_example()
    complex_assertions_example()
    
    print("Examples completed!")


if __name__ == "__main__":
    main()