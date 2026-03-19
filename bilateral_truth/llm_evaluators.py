"""
LLM-based evaluators for bilateral factuality assessment.

This module provides implementations that use language models to perform
bilateral evaluation of atomic formulas, assessing both verifiability
and refutability as described in the research paper.
"""

import os
import time

# import json  # unused currently
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
from collections import Counter
import random

from .assertions import Assertion
from .truth_values import GeneralizedTruthValue, TruthValueComponent


class LLMEvaluator(ABC):
    """Abstract base class for LLM-based evaluators."""

    @abstractmethod
    def evaluate_bilateral(
        self, assertion: Assertion, samples: int = 1, system_prompt: Optional[str] = None, context: Optional[str] = None
    ) -> GeneralizedTruthValue:
        """
        Perform bilateral evaluation of an assertion.

        Args:
            assertion: The assertion to evaluate
            samples: Number of samples to take for majority voting (default: 1)
            system_prompt: Optional custom system prompt for verification/refutation instructions
            context: Optional background information to inform the evaluation

        Returns:
            GeneralizedTruthValue with verifiability (u) and refutability (v) components
        """
        pass

    def evaluate_with_majority_voting(
        self, assertion: Assertion, samples: int, tiebreak_strategy: str = "random", system_prompt: Optional[str] = None, context: Optional[str] = None
    ) -> GeneralizedTruthValue:
        """
        Evaluate assertion using multiple samples and majority voting.

        Args:
            assertion: The assertion to evaluate
            samples: Number of samples to take
            tiebreak_strategy: How to break ties ("random", "optimistic", "pessimistic")
                - "random": Randomly select from tied components
                - "optimistic": Prefer t (verified/refuted) - bias toward strong claims
                - "pessimistic": Prefer f (cannot verify/refute) - bias toward epistemic caution
            system_prompt: Optional custom system prompt for verification/refutation instructions
            context: Optional background information to inform the evaluation

        Returns:
            GeneralizedTruthValue determined by majority vote with tiebreaking
        """
        if samples <= 0:
            raise ValueError("Number of samples must be positive")

        if samples == 1:
            # Single sample - no voting needed
            return self._single_evaluation(assertion, system_prompt=system_prompt, context=context)

        # Collect multiple samples
        results = []
        for i in range(samples):
            try:
                result = self._single_evaluation(assertion, system_prompt=system_prompt, context=context)
                results.append(result)
            except Exception as e:
                print(f"Warning: Sample {i+1} failed: {e}")
                # Continue with fewer samples if some fail
                continue

        if not results:
            # All samples failed
            return GeneralizedTruthValue(TruthValueComponent.UNDEFINED, TruthValueComponent.UNDEFINED)

        # Apply majority voting
        return self._majority_vote(results, tiebreak_strategy)

    def _single_evaluation(self, assertion: Assertion, system_prompt: Optional[str] = None, context: Optional[str] = None) -> GeneralizedTruthValue:
        """
        Perform a single bilateral evaluation using separate verification and refutation calls.

        This implements Definition 3.4 from the paper with separate API calls.
        
        Args:
            assertion: The assertion to evaluate
            system_prompt: Optional custom system prompt for verification/refutation instructions
            context: Optional background information to inform the evaluation
        """
        try:
            # Make separate calls for verification and refutation
            u_component = self._evaluate_verification(assertion, system_prompt=system_prompt, context=context)
            v_component = self._evaluate_refutation(assertion, system_prompt=system_prompt, context=context)

            return GeneralizedTruthValue(u_component, v_component)

        except Exception as e:
            print(f"Warning: Bilateral evaluation failed: {e}")
            return GeneralizedTruthValue(TruthValueComponent.UNDEFINED, TruthValueComponent.UNDEFINED)

    def _majority_vote(
        self, results: List[GeneralizedTruthValue], tiebreak_strategy: str
    ) -> GeneralizedTruthValue:
        """
        Apply majority voting to a list of GeneralizedTruthValue results.

        Args:
            results: List of evaluation results
            tiebreak_strategy: Tiebreaking strategy

        Returns:
            The majority vote result with tiebreaking applied
        """
        if not results:
            return GeneralizedTruthValue(TruthValueComponent.UNDEFINED, TruthValueComponent.UNDEFINED)

        if len(results) == 1:
            return results[0]

        # Separate voting for verifiability (u) and refutability (v)
        u_votes = [result.u for result in results]
        v_votes = [result.v for result in results]

        # Get majority for each component
        u_majority = self._component_majority_vote(u_votes, tiebreak_strategy)
        v_majority = self._component_majority_vote(v_votes, tiebreak_strategy)

        return GeneralizedTruthValue(u_majority, v_majority)

    def _component_majority_vote(
        self, votes: List[TruthValueComponent], tiebreak_strategy: str
    ) -> TruthValueComponent:
        """
        Determine majority vote for a single component (u or v).

        Args:
            votes: List of TruthValueComponent votes
            tiebreak_strategy: How to break ties

        Returns:
            The winning TruthValueComponent
        """
        vote_counts = Counter(votes)

        # Find the maximum count
        max_count = max(vote_counts.values())
        winners = [
            component for component, count in vote_counts.items() if count == max_count
        ]

        if len(winners) == 1:
            # Clear winner
            return winners[0]

        # Tie detected - apply tiebreaking strategy
        return self._tiebreak(winners, tiebreak_strategy)

    def _tiebreak(
        self, tied_components: List[TruthValueComponent], strategy: str
    ) -> TruthValueComponent:
        """
        Break ties between components using the specified strategy.

        Args:
            tied_components: List of tied components
            strategy: Tiebreaking strategy ("random", "optimistic", "pessimistic")

        Returns:
            The component chosen by the tiebreaking strategy

        Tiebreaking Strategies:
        - "random": Randomly select from tied components
        - "optimistic": Prefer t (verified/refuted) > f (cannot verify/refute) > e (parse error)
        - "pessimistic": Prefer f (cannot verify/refute) > t (verified/refuted) > e (parse error)
        """
        if strategy == "random":
            return random.choice(tied_components)
        elif strategy == "optimistic":
            # Optimistic: prefer TRUE (verified/refuted) when in doubt
            if TruthValueComponent.TRUE in tied_components:
                return TruthValueComponent.TRUE
            elif TruthValueComponent.FALSE in tied_components:
                return TruthValueComponent.FALSE
            else:
                return TruthValueComponent.UNDEFINED
        elif strategy == "pessimistic":
            # Pessimistic: prefer FALSE (cannot verify/cannot refute) when in doubt
            if TruthValueComponent.FALSE in tied_components:
                return TruthValueComponent.FALSE
            elif TruthValueComponent.TRUE in tied_components:
                return TruthValueComponent.TRUE
            else:
                return TruthValueComponent.UNDEFINED
        else:
            # Default to random for unknown strategies
            return random.choice(tied_components)

    def _create_verification_prompt(self, assertion: Assertion, context: Optional[str] = None) -> str:
        """
        Create a prompt for verification assessment as per Definition 3.4.

        The LLM must respond with exactly "VERIFIED" or "CANNOT VERIFY".
        
        Args:
            assertion: The assertion to evaluate
            context: Optional background information to inform the evaluation
        """
        context_section = f"\n\nContext: {context}\n" if context else ""
        
        return f"""You are tasked with determining whether the following assertion can be verified as true based on available evidence and knowledge.

Assertion: {assertion}{context_section}

Your task is to determine if this assertion can be verified. Consider all available evidence, facts, and reliable sources of information.

You must respond with exactly one of these two token sequences:
- VERIFIED (if the assertion can be confirmed as true based on evidence)
- CANNOT VERIFY (if the assertion cannot be confirmed as true, either due to lack of evidence, uncertainty, or because it is false)

Do not provide any explanation or additional text. Respond with only the required token sequence.

Response:"""

    def _create_refutation_prompt(self, assertion: Assertion, context: Optional[str] = None) -> str:
        """
        Create a prompt for refutation assessment as per Definition 3.4.

        The LLM must respond with exactly "REFUTED" or "CANNOT REFUTE".
        
        Args:
            assertion: The assertion to evaluate
            context: Optional background information to inform the evaluation
        """
        context_section = f"\n\nContext: {context}\n" if context else ""
        
        return f"""You are tasked with determining whether the following assertion can be refuted (shown to be false) based on available evidence and knowledge.

Assertion: {assertion}{context_section}

Your task is to determine if this assertion can be refuted. Consider all available evidence, facts, and reliable sources of information that might contradict the assertion.

You must respond with exactly one of these two token sequences:
- REFUTED (if the assertion can be shown to be false based on evidence)
- CANNOT REFUTE (if the assertion cannot be shown to be false, either due to lack of contradictory evidence, uncertainty, or because it is true)

Do not provide any explanation or additional text. Respond with only the required token sequence.

Response:"""

    def _parse_verification_response(self, response_text: str) -> TruthValueComponent:
        """
        Parse verification response to extract verifiability value.

        Args:
            response_text: Raw response from the LLM

        Returns:
            TruthValueComponent for verifiability (u)
        """
        # Clean up the response
        response = response_text.strip().upper()

        # Look for exact token sequences as per Definition 3.4
        if "VERIFIED" in response and "CANNOT VERIFY" not in response:
            return TruthValueComponent.TRUE
        elif "CANNOT VERIFY" in response:
            return TruthValueComponent.FALSE
        else:
            # Model failed to return required token sequence
            print(f"WARNING [parse_verification]: Unexpected response: '{response[:80]}' [PARSE_FAILURE]")
            return TruthValueComponent.UNDEFINED

    def _parse_refutation_response(self, response_text: str) -> TruthValueComponent:
        """
        Parse refutation response to extract refutability value.

        Args:
            response_text: Raw response from the LLM

        Returns:
            TruthValueComponent for refutability (v)
        """
        # Clean up the response
        response = response_text.strip().upper()

        # Look for exact token sequences as per Definition 3.4
        if "REFUTED" in response and "CANNOT REFUTE" not in response:
            return TruthValueComponent.TRUE
        elif "CANNOT REFUTE" in response:
            return TruthValueComponent.FALSE
        else:
            # Model failed to return required token sequence
            print(f"WARNING [parse_refutation]: Unexpected response: '{response[:80]}' [PARSE_FAILURE]")
            return TruthValueComponent.UNDEFINED

    def _call_with_retry(self, api_callable, call_type: str = "api",
                         max_retries: int = 5, base_delay: float = 2.0):
        """Execute an API call with exponential backoff retry on transient errors.

        Args:
            api_callable: Zero-argument callable that performs the API call and returns text
            call_type: Label for logging (e.g., "openai_verification/gpt-4.1")
            max_retries: Maximum number of retry attempts after the initial attempt
            base_delay: Initial delay in seconds (doubles each retry)

        Returns:
            The return value of api_callable on success

        Raises:
            The last exception if all retries are exhausted or error is non-transient
        """
        _TRANSIENT_MARKERS = (
            "rate limit", "ratelimit", "rate_limit",
            "429", "503", "502", "overloaded",
            "timeout", "timed out", "read timeout",
            "connection error", "connectionerror",
            "service unavailable", "serviceunavailable",
            "internal server error", "internalservererror",
            "too many requests",
        )

        for attempt in range(max_retries + 1):
            try:
                return api_callable()
            except Exception as e:
                error_str = (str(e) + type(e).__name__).lower()
                is_transient = any(m in error_str for m in _TRANSIENT_MARKERS)

                if not is_transient or attempt == max_retries:
                    if attempt > 0:
                        print(f"ERROR [{call_type}]: Failed after {attempt} retries: "
                              f"{type(e).__name__} [API_ERROR_RETRY_EXHAUSTED]")
                    raise

                delay = base_delay * (2 ** attempt)
                print(f"WARNING [{call_type}]: Transient error (attempt {attempt + 1}/{max_retries + 1}), "
                      f"retrying in {delay:.0f}s: {type(e).__name__}: {str(e)[:120]}")
                time.sleep(delay)

    def _raw_complete(self, prompt: str, system_prompt: str, max_tokens: int,
                      call_type: str) -> str:
        """Make a single provider API call and return the response text.

        Must be overridden by concrete evaluator classes. All retry logic is
        handled here via _call_with_retry; callers receive the raw text string
        or an exception if all retries are exhausted.
        """
        raise NotImplementedError("Subclasses must implement _raw_complete")

    def _evaluate_verification(self, assertion: Assertion, system_prompt: Optional[str] = None, context: Optional[str] = None) -> TruthValueComponent:
        """Evaluate verification component. Must be overridden by concrete classes."""
        raise NotImplementedError("Subclasses must implement _evaluate_verification")

    def _evaluate_refutation(self, assertion: Assertion, system_prompt: Optional[str] = None, context: Optional[str] = None) -> TruthValueComponent:
        """Evaluate refutation component. Must be overridden by concrete classes."""
        raise NotImplementedError("Subclasses must implement _evaluate_refutation")

    # ------------------------------------------------------------------
    # Non-bilateral evaluation methods (forced unilateral / ternary / confidence)
    # ------------------------------------------------------------------

    def _create_forced_unilateral_prompt(self, assertion: Assertion,
                                          context: Optional[str] = None) -> str:
        context_section = f"\n\nContext: {context}\n" if context else ""
        return (
            f"Determine whether the following statement is correct or incorrect."
            f"{context_section}\nStatement: {assertion}\n\n"
            "You must respond with exactly one of:\n"
            "- TRUE (if the statement is correct)\n"
            "- FALSE (if the statement is incorrect)\n\n"
            "Do not provide any explanation. Respond with only TRUE or FALSE.\n\nResponse:"
        )

    def _create_ternary_prompt(self, assertion: Assertion,
                                context: Optional[str] = None) -> str:
        context_section = f"\n\nContext: {context}\n" if context else ""
        return (
            f"Based on available evidence and knowledge, determine whether the following "
            f"statement is supported by evidence, contradicted by evidence, or undetermined."
            f"{context_section}\nStatement: {assertion}\n\n"
            "Consider all available facts and reliable sources. You must respond with exactly one of:\n"
            "- TRUE (if evidence supports the statement as correct)\n"
            "- FALSE (if evidence contradicts the statement as incorrect)\n"
            "- UNCERTAIN (if there is insufficient evidence to either support or refute it)\n\n"
            "Do not provide any explanation. Respond with only TRUE, FALSE, or UNCERTAIN.\n\nResponse:"
        )

    def _create_confidence_prompt_unilateral(self, assertion: Assertion,
                                              context: Optional[str] = None) -> str:
        context_section = f"\n\nContext: {context}\n" if context else ""
        return (
            f"Rate your confidence that the following statement is correct."
            f"{context_section}\nStatement: {assertion}\n\n"
            "Provide a single number between 0.0 and 1.0:\n"
            "- 0.0 = Definitely incorrect\n"
            "- 0.5 = Completely uncertain\n"
            "- 1.0 = Definitely correct\n\n"
            "Respond with ONLY: CONFIDENCE: [number]\n\nResponse:"
        )

    def _parse_forced_unilateral_response(self, response_text: str) -> TruthValueComponent:
        r = response_text.strip().upper()
        if "TRUE" in r:
            return TruthValueComponent.TRUE
        elif "FALSE" in r:
            return TruthValueComponent.FALSE
        print(f"WARNING [parse_forced_unilateral]: Unexpected: '{r[:80]}' [PARSE_FAILURE] — defaulting FALSE")
        return TruthValueComponent.FALSE

    def _parse_ternary_response(self, response_text: str) -> TruthValueComponent:
        r = response_text.strip().upper()
        if "UNCERTAIN" in r:
            return TruthValueComponent.UNDEFINED
        elif "TRUE" in r:
            return TruthValueComponent.TRUE
        elif "FALSE" in r:
            return TruthValueComponent.FALSE
        print(f"WARNING [parse_ternary]: Unexpected: '{r[:80]}' [PARSE_FAILURE]")
        return TruthValueComponent.UNDEFINED

    def _parse_confidence_response_unilateral(self, response_text: str) -> float:
        import re
        m = re.search(r'CONFIDENCE:\s*([\d.]+)', response_text, re.IGNORECASE)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
        print(f"WARNING [parse_confidence]: Unexpected: '{response_text[:80]}' [PARSE_FAILURE]")
        return 0.5

    def evaluate_forced_unilateral(self, assertion: Assertion,
                                    context: Optional[str] = None) -> TruthValueComponent:
        """Single-call forced binary evaluation — TRUE or FALSE, no abstention."""
        model_id = getattr(self, "model", "unknown")
        try:
            prompt = self._create_forced_unilateral_prompt(assertion, context=context)
            sys_prompt = "You are an expert fact-checker. You must respond with only TRUE or FALSE."
            text = self._raw_complete(prompt, sys_prompt, max_tokens=10,
                                      call_type=f"forced_unilateral/{model_id}")
            return self._parse_forced_unilateral_response(text)
        except Exception as e:
            print(f"ERROR [forced_unilateral/{model_id}]: {type(e).__name__}: {str(e)[:120]} [API_ERROR]")
            return TruthValueComponent.FALSE

    def evaluate_ternary(self, assertion: Assertion,
                          context: Optional[str] = None) -> TruthValueComponent:
        """Single-call ternary evaluation — TRUE / FALSE / UNCERTAIN."""
        model_id = getattr(self, "model", "unknown")
        try:
            prompt = self._create_ternary_prompt(assertion, context=context)
            sys_prompt = "You are an expert fact-checker evaluating claims based on evidence. Respond with only TRUE, FALSE, or UNCERTAIN."
            text = self._raw_complete(prompt, sys_prompt, max_tokens=10,
                                      call_type=f"ternary/{model_id}")
            return self._parse_ternary_response(text)
        except Exception as e:
            print(f"ERROR [ternary/{model_id}]: {type(e).__name__}: {str(e)[:120]} [API_ERROR]")
            return TruthValueComponent.UNDEFINED

    def evaluate_confidence(self, assertion: Assertion,
                             context: Optional[str] = None) -> float:
        """Single-call numerical confidence rating (0.0–1.0)."""
        model_id = getattr(self, "model", "unknown")
        try:
            prompt = self._create_confidence_prompt_unilateral(assertion, context=context)
            sys_prompt = ("You are an expert evaluator. "
                          "Always respond with 'CONFIDENCE: X.X' where X.X is between 0.0 and 1.0.")
            text = self._raw_complete(prompt, sys_prompt, max_tokens=20,
                                      call_type=f"confidence/{model_id}")
            return self._parse_confidence_response_unilateral(text)
        except Exception as e:
            print(f"ERROR [confidence/{model_id}]: {type(e).__name__}: {str(e)[:120]} [API_ERROR]")
            return 0.5


class OpenAIEvaluator(LLMEvaluator):
    """LLM evaluator using OpenAI's API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize OpenAI evaluator.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY environment variable
            model: Model name to use (default: gpt-4)
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )

        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)

    def evaluate_bilateral(
        self, assertion: Assertion, samples: int = 1, system_prompt: Optional[str] = None, context: Optional[str] = None
    ) -> GeneralizedTruthValue:
        """Evaluate assertion using OpenAI API with optional sampling."""
        if samples > 1:
            return self.evaluate_with_majority_voting(assertion, samples, system_prompt=system_prompt, context=context)
        return self._single_evaluation(assertion, system_prompt=system_prompt, context=context)

    def _raw_complete(self, prompt: str, system_prompt: str, max_tokens: int,
                      call_type: str) -> str:
        """Single OpenAI chat completion call with retry."""
        request_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max_tokens,
        }
        if not self.model.startswith("gpt-5"):
            request_params["temperature"] = 0.0
        return self._call_with_retry(
            lambda: self.client.chat.completions.create(**request_params).choices[0].message.content,
            call_type=call_type,
        )

    def _evaluate_verification(self, assertion: Assertion, system_prompt: Optional[str] = None, context: Optional[str] = None) -> TruthValueComponent:
        """Evaluate verification using OpenAI API with exponential backoff retry."""
        try:
            prompt = self._create_verification_prompt(assertion, context=context)
            sys_prompt = system_prompt or "You are an expert in factual verification. You must respond with only the exact required token sequences."
            response_text = self._raw_complete(
                prompt, sys_prompt, max_tokens=10,
                call_type=f"openai_verification/{self.model}"
            )
            return self._parse_verification_response(response_text)
        except Exception as e:
            print(f"ERROR [openai_verification/{self.model}]: {type(e).__name__}: {str(e)[:120]} [API_ERROR]")
            return TruthValueComponent.UNDEFINED

    def _evaluate_refutation(self, assertion: Assertion, system_prompt: Optional[str] = None, context: Optional[str] = None) -> TruthValueComponent:
        """Evaluate refutation using OpenAI API with exponential backoff retry."""
        try:
            prompt = self._create_refutation_prompt(assertion, context=context)
            sys_prompt = system_prompt or "You are an expert in logical refutation. You must respond with only the exact required token sequences."
            response_text = self._raw_complete(
                prompt, sys_prompt, max_tokens=10,
                call_type=f"openai_refutation/{self.model}"
            )
            return self._parse_refutation_response(response_text)
        except Exception as e:
            print(f"ERROR [openai_refutation/{self.model}]: {type(e).__name__}: {str(e)[:120]} [API_ERROR]")
            return TruthValueComponent.UNDEFINED


class AnthropicEvaluator(LLMEvaluator):
    """LLM evaluator using Anthropic's Claude API."""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"
    ):
        """
        Initialize Anthropic evaluator.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY environment variable
            model: Model name to use (default: claude-sonnet-4-20250514)
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def evaluate_bilateral(
        self, assertion: Assertion, samples: int = 1, system_prompt: Optional[str] = None, context: Optional[str] = None
    ) -> GeneralizedTruthValue:
        """Evaluate assertion using Anthropic API with optional sampling."""
        if samples > 1:
            return self.evaluate_with_majority_voting(assertion, samples, system_prompt=system_prompt, context=context)
        return self._single_evaluation(assertion, system_prompt=system_prompt, context=context)

    def _raw_complete(self, prompt: str, system_prompt: str, max_tokens: int,
                      call_type: str) -> str:
        """Single Anthropic messages call with retry."""
        return self._call_with_retry(
            lambda: self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            ).content[0].text,
            call_type=call_type,
        )

    def _evaluate_verification(self, assertion: Assertion, system_prompt: Optional[str] = None, context: Optional[str] = None) -> TruthValueComponent:
        """Evaluate verification using Anthropic API with exponential backoff retry."""
        try:
            prompt = self._create_verification_prompt(assertion, context=context)
            sys_prompt = system_prompt or "You are an expert in factual verification. You must respond with only the exact required token sequences."
            response_text = self._raw_complete(
                prompt, sys_prompt, max_tokens=10,
                call_type=f"anthropic_verification/{self.model}"
            )
            return self._parse_verification_response(response_text)
        except Exception as e:
            print(f"ERROR [anthropic_verification/{self.model}]: {type(e).__name__}: {str(e)[:120]} [API_ERROR]")
            return TruthValueComponent.UNDEFINED

    def _evaluate_refutation(self, assertion: Assertion, system_prompt: Optional[str] = None, context: Optional[str] = None) -> TruthValueComponent:
        """Evaluate refutation using Anthropic API with exponential backoff retry."""
        try:
            prompt = self._create_refutation_prompt(assertion, context=context)
            sys_prompt = system_prompt or "You are an expert in logical refutation. You must respond with only the exact required token sequences."
            response_text = self._raw_complete(
                prompt, sys_prompt, max_tokens=10,
                call_type=f"anthropic_refutation/{self.model}"
            )
            return self._parse_refutation_response(response_text)
        except Exception as e:
            print(f"ERROR [anthropic_refutation/{self.model}]: {type(e).__name__}: {str(e)[:120]} [API_ERROR]")
            return TruthValueComponent.UNDEFINED


class MockLLMEvaluator(LLMEvaluator):
    """Mock LLM evaluator for testing and demonstration purposes."""

    def __init__(self, responses: Optional[Dict[str, GeneralizedTruthValue]] = None):
        """
        Initialize mock evaluator.

        Args:
            responses: Dictionary mapping assertion strings to predefined responses
        """
        self.responses = responses or {}

    def evaluate_bilateral(
        self, assertion: Assertion, samples: int = 1, system_prompt: Optional[str] = None, context: Optional[str] = None
    ) -> GeneralizedTruthValue:
        """Return predefined response or simulate evaluation with optional sampling."""
        if samples > 1:
            return self.evaluate_with_majority_voting(assertion, samples, system_prompt=system_prompt, context=context)
        return self._single_evaluation(assertion, system_prompt=system_prompt, context=context)

    def _raw_complete(self, prompt: str, system_prompt: str, max_tokens: int,
                      call_type: str) -> str:
        """Mock completion — always returns TRUE."""
        return "TRUE"

    def _evaluate_verification(self, assertion: Assertion, system_prompt: Optional[str] = None, context: Optional[str] = None) -> TruthValueComponent:
        """Mock verification evaluation using predefined logic."""
        assertion_str = str(assertion)

        # Return predefined response if available
        if assertion_str in self.responses:
            return self.responses[assertion_str].u

        # Simulate verification based on assertion content
        predicate = assertion.predicate.lower()

        # Check for weather-related content
        weather_keywords = [
            "sunny",
            "clear",
            "bright",
            "raining",
            "cloudy",
            "stormy",
            "rain",
            "cloud",
            "storm",
        ]
        if any(keyword in predicate for keyword in weather_keywords):
            # Weather statements - generally verifiable
            return TruthValueComponent.TRUE
        elif (
            predicate.startswith("love")
            or predicate.startswith("like")
            or "love" in predicate
            or "like" in predicate
        ):
            # Emotional statements - hard to verify
            return TruthValueComponent.UNDEFINED
        elif any(keyword in predicate for keyword in ["true", "correct", "valid"]):
            # Meta-truth statements
            return TruthValueComponent.TRUE
        elif any(keyword in predicate for keyword in ["false", "incorrect", "invalid"]):
            # Meta-false statements
            return TruthValueComponent.FALSE
        else:
            # Unknown predicates
            return TruthValueComponent.UNDEFINED

    def _evaluate_refutation(self, assertion: Assertion, system_prompt: Optional[str] = None, context: Optional[str] = None) -> TruthValueComponent:
        """Mock refutation evaluation using predefined logic."""
        assertion_str = str(assertion)

        # Return predefined response if available
        if assertion_str in self.responses:
            return self.responses[assertion_str].v

        # Simulate refutation based on assertion content
        predicate = assertion.predicate.lower()

        # Check for different types of weather content
        if any(keyword in predicate for keyword in ["sunny", "clear", "bright"]):
            # Positive weather statements - not always refutable
            return TruthValueComponent.UNDEFINED
        elif any(
            keyword in predicate
            for keyword in ["raining", "cloudy", "stormy", "rain", "cloud", "storm"]
        ):
            # Variable weather statements - refutable
            return TruthValueComponent.TRUE
        elif (
            predicate.startswith("love")
            or predicate.startswith("like")
            or "love" in predicate
            or "like" in predicate
        ):
            # Emotional statements - hard to refute
            return TruthValueComponent.UNDEFINED
        elif any(keyword in predicate for keyword in ["true", "correct", "valid"]):
            # Meta-truth statements
            return TruthValueComponent.FALSE
        elif any(keyword in predicate for keyword in ["false", "incorrect", "invalid"]):
            # Meta-false statements
            return TruthValueComponent.TRUE
        else:
            # Unknown predicates
            return TruthValueComponent.UNDEFINED


def create_llm_evaluator(provider: str = "mock", **kwargs) -> LLMEvaluator:
    """
    Factory function to create LLM evaluators.

    Args:
        provider: LLM provider ('openai', 'anthropic', 'mock')
        **kwargs: Additional arguments passed to the evaluator constructor

    Returns:
        LLMEvaluator instance
    """
    if provider.lower() == "openai":
        return OpenAIEvaluator(**kwargs)
    elif provider.lower() == "anthropic":
        return AnthropicEvaluator(**kwargs)
    elif provider.lower() == "mock":
        return MockLLMEvaluator(**kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: 'openai', 'anthropic', 'mock'"
        )
