"""
Implementation of the ζ (zeta) and ζ_c (zeta_c) functions.

The ζ function provides bilateral factuality evaluation of atomic formulas,
while ζ_c adds persistent caching for improved performance.
"""

from typing import Dict, Callable, Optional, Union, Tuple, Any
from .assertions import Assertion
from .truth_values import GeneralizedTruthValue


class ZetaCache:
    """
    Persistent, immutable-style cache for ζ_c function.

    Implements the cache c used in the ζ_c definition:
    ζ_c(φ) = c(φ) if φ ∈ dom(c), else ζ(φ) and c := c ∪ {(φ, ζ(φ))}
    """

    def __init__(self):
        self._cache: Dict[Union[Assertion, Tuple[Any, ...]], GeneralizedTruthValue] = {}

    def __contains__(self, key: Union[Assertion, Tuple[Any, ...]]) -> bool:
        """Check if key is in the domain of the cache."""
        return key in self._cache

    def get(
        self, key: Union[Assertion, Tuple[Any, ...]]
    ) -> Optional[GeneralizedTruthValue]:
        """Get cached value for key, or None if not cached."""
        return self._cache.get(key)

    def update(
        self, key: Union[Assertion, Tuple[Any, ...]], value: GeneralizedTruthValue
    ) -> "ZetaCache":
        """
        Return a new cache with the key-value pair added.

        Note: In practice, we modify the existing cache for efficiency,
        but the interface maintains the mathematical abstraction of
        immutable cache updates.
        """
        self._cache[key] = value
        return self

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


def zeta(
    assertion: Assertion, 
    evaluator: Callable[[Assertion], GeneralizedTruthValue],
    samples: int = 1,
    tiebreak_strategy: str = "random",
    system_prompt: Optional[str] = None,
    context: Optional[str] = None,
) -> GeneralizedTruthValue:
    """
    The base ζ function for bilateral factuality evaluation.

    This function performs bilateral evaluation by assessing both verifiability (u)
    and refutability (v) of assertions using LLM-based evaluation.

    Args:
        assertion: The assertion φ ∈ ℒ_AT to evaluate
        evaluator: LLM evaluator function that performs bilateral assessment.
                  Must be provided - no default evaluation.
        samples: Number of samples for majority voting (default: 1)
        tiebreak_strategy: Strategy for breaking ties ("random", "pessimistic", "optimistic")
        system_prompt: Optional custom system prompt for verification/refutation instructions
        context: Optional background information to inform the evaluation

    Returns:
        A GeneralizedTruthValue <u,v> representing the bilateral evaluation
    """
    # Handle LLM evaluation with sampling and prompts
    if hasattr(evaluator, "evaluate_bilateral"):
        # New interface with sampling support - check if it supports new parameters
        import inspect
        try:
            sig = inspect.signature(evaluator.evaluate_bilateral)
            if 'system_prompt' in sig.parameters and 'context' in sig.parameters:
                # New interface with system_prompt and context support
                truth_value = evaluator.evaluate_bilateral(assertion, samples, system_prompt=system_prompt, context=context)
            else:
                # Old interface without system_prompt/context support
                if system_prompt or context:
                    print(
                        "Warning: Evaluator doesn't support system_prompt or context parameters."
                    )
                truth_value = evaluator.evaluate_bilateral(assertion, samples)
        except (TypeError, ValueError):
            # Fallback if signature inspection fails
            if system_prompt or context:
                print(
                    "Warning: Could not detect evaluator parameter support. Ignoring system_prompt and context."
                )
            truth_value = evaluator.evaluate_bilateral(assertion, samples)
    elif callable(evaluator):
        # Legacy callable interface or function reference
        import inspect
        try:
            sig = inspect.signature(evaluator)
            if 'system_prompt' in sig.parameters and 'context' in sig.parameters:
                # Function supports new parameters
                if 'samples' in sig.parameters:
                    truth_value = evaluator(assertion, samples, system_prompt=system_prompt, context=context)
                else:
                    # Function doesn't support samples but supports new parameters
                    if samples > 1:
                        print("Warning: Evaluator doesn't support sampling. Using single evaluation.")
                    truth_value = evaluator(assertion, system_prompt=system_prompt, context=context)
            else:
                # Function doesn't support new parameters
                if samples > 1:
                    print("Warning: Evaluator doesn't support sampling. Using single evaluation.")
                if system_prompt or context:
                    print("Warning: Evaluator doesn't support system_prompt or context parameters.")
                truth_value = evaluator(assertion)
        except (TypeError, ValueError):
            # Fallback for inspection failure
            if samples > 1:
                print("Warning: Evaluator doesn't support sampling. Using single evaluation.")
            if system_prompt or context:
                print("Warning: Evaluator doesn't support system_prompt or context parameters.")
            truth_value = evaluator(assertion)
    else:
        raise ValueError(f"Invalid evaluator: {evaluator}")

    return truth_value


# Global cache instance
_global_cache = ZetaCache()


def zeta_c(
    assertion: Assertion,
    evaluator: Callable[[Assertion], GeneralizedTruthValue],
    cache: Optional[ZetaCache] = None,
    samples: int = 1,
    tiebreak_strategy: str = "random",
    system_prompt: Optional[str] = None,
    context: Optional[str] = None,
) -> GeneralizedTruthValue:
    """
    The cached ζ_c function: ℒ_AT → 𝒱³ × 𝒱³

    Implements: ζ_c(φ) = c(φ) if φ ∈ dom(c), else ζ(φ) and c := c ∪ {(φ, ζ(φ))}

    Args:
        assertion: The assertion φ ∈ ℒ_AT to evaluate
        evaluator: LLM evaluator function for bilateral assessment. Required.
        cache: Optional cache instance. If None, uses global cache.
        samples: Number of samples for majority voting (default: 1)
        tiebreak_strategy: Strategy for breaking ties ("random", "pessimistic", "optimistic")
        system_prompt: Optional custom system prompt for verification/refutation instructions
        context: Optional background information to inform the evaluation

    Returns:
        A GeneralizedTruthValue <u,v> from cache or computed via ζ
    """
    if cache is None:
        cache = _global_cache

    # Cache key includes assertion, system_prompt, and context to ensure correct caching
    # Different prompts/contexts should produce different cached results
    cache_key = (assertion, system_prompt, context)

    # Check if result is in cache domain: φ ∈ dom(c)
    if cache_key in cache:
        cached_result = cache.get(cache_key)  # Return c(φ)
        assert cached_result is not None  # Should never be None since key exists
        return cached_result

    # Compute ζ(φ) - this is the ζ function call from Definition 3.5
    truth_value = zeta(assertion, evaluator, samples, tiebreak_strategy, system_prompt, context)

    # Update cache: c := c ∪ {(φ, ζ(φ))}
    cache.update(cache_key, truth_value)

    return truth_value


def clear_cache() -> None:
    """Clear the global cache."""
    _global_cache.clear()


def get_cache_size() -> int:
    """Get the number of entries in the global cache."""
    return len(_global_cache)
