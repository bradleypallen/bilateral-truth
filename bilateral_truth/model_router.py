"""
Model routing functionality for the ζ_c CLI.

This module provides routing logic to determine which LLM provider and
configuration to use based on explicit model names.
"""

from typing import Optional, TYPE_CHECKING
from .llm_evaluators import (
    LLMEvaluator,
    OpenAIEvaluator,
    AnthropicEvaluator,
    MockLLMEvaluator,
)

if TYPE_CHECKING:
    from .assertions import Assertion
    from .truth_values import GeneralizedTruthValue, TruthValueComponent


class ModelRouter:
    """
    Routes model names to appropriate LLM evaluators.
    Uses explicit model names only - no patterns or aliases.
    """

    # Map of exact model names to their providers
    MODEL_PROVIDERS = {
        # OpenAI models
        "gpt-5-2025-08-07": "openai",
        "gpt-5-mini-2025-08-07": "openai",
        "gpt-4.1-2025-04-14": "openai",
        "gpt-4.1-mini-2025-04-14": "openai",
        
        # Anthropic models
        "claude-opus-4-6": "anthropic",
        "claude-opus-4-1-20250805": "anthropic",
        "claude-sonnet-4-20250514": "anthropic",
        "claude-3-7-sonnet-20250219": "anthropic",
        "claude-haiku-4-5-20251001": "anthropic",
        
        # OpenRouter models
        "meta-llama/llama-4-scout": "openrouter",
        "meta-llama/llama-4-maverick": "openrouter",
        "google/gemini-2.5-pro": "openrouter",
        "google/gemini-2.5-flash": "openrouter",
        "deepseek/deepseek-chat": "openrouter",       # DeepSeek-V3
        "qwen/qwen-2.5-72b-instruct": "openrouter",   # Qwen2.5-72B
        
        # Mock models for testing
        "mock": "mock",
    }

    @classmethod
    def get_provider(cls, model_name: str) -> str:
        """
        Get the provider for a specific model name.
        
        Args:
            model_name: The exact model name
            
        Returns:
            The provider name
            
        Raises:
            ValueError: If the model name is not recognized
        """
        if model_name not in cls.MODEL_PROVIDERS:
            available_models = ", ".join(sorted(cls.MODEL_PROVIDERS.keys()))
            raise ValueError(
                f"Unknown model: '{model_name}'. Available models: {available_models}"
            )
        return cls.MODEL_PROVIDERS[model_name]

    @classmethod
    def create_evaluator(cls, model_name: str, **kwargs) -> LLMEvaluator:
        """
        Create an LLM evaluator for the specified model.
        
        Args:
            model_name: The exact model name
            **kwargs: Additional arguments passed to the evaluator constructor
            
        Returns:
            An LLMEvaluator instance configured for the specified model
            
        Raises:
            ValueError: If the model name is not recognized
        """
        provider = cls.get_provider(model_name)
        
        if provider == "openai":
            return OpenAIEvaluator(model=model_name, **kwargs)
        elif provider == "anthropic":
            return AnthropicEvaluator(model=model_name, **kwargs)
        elif provider == "openrouter":
            return OpenRouterEvaluator(model=model_name, **kwargs)
        elif provider == "mock":
            return MockLLMEvaluator(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @classmethod
    def list_available_models(cls) -> dict:
        """
        Get all available models organized by provider.
        
        Returns:
            Dictionary mapping provider names to lists of model names
        """
        models_by_provider = {}
        for model, provider in cls.MODEL_PROVIDERS.items():
            if provider not in models_by_provider:
                models_by_provider[provider] = []
            models_by_provider[provider].append(model)
        
        # Sort models within each provider
        for provider in models_by_provider:
            models_by_provider[provider].sort()
            
        return models_by_provider


class OpenRouterEvaluator(LLMEvaluator):
    """LLM evaluator using OpenRouter's API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "openrouter/auto"):
        """
        Initialize OpenRouter evaluator.
        
        Args:
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY environment variable
            model: Model name to use (default: openrouter/auto)
        """
        import os
        
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required for OpenRouter. Install with: pip install openai"
            )
        
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key must be provided or set in OPENROUTER_API_KEY environment variable"
            )
        
        self.model = model
        # OpenRouter uses OpenAI-compatible API
        self.client = openai.OpenAI(
            api_key=self.api_key, base_url="https://openrouter.ai/api/v1"
        )

    def evaluate_bilateral(
        self, assertion: "Assertion", samples: int = 1, system_prompt: Optional[str] = None, context: Optional[str] = None
    ) -> "GeneralizedTruthValue":
        """Evaluate assertion using OpenRouter API with optional sampling."""
        if samples > 1:
            return self.evaluate_with_majority_voting(assertion, samples, system_prompt=system_prompt, context=context)
        return self._single_evaluation(assertion, system_prompt=system_prompt, context=context)

    def _raw_complete(self, prompt: str, system_prompt: str, max_tokens: int,
                      call_type: str) -> str:
        """Single OpenRouter chat completion call with retry."""
        request_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max_tokens,
        }
        if "gpt-5" not in self.model:
            request_params["temperature"] = 0.0
        return self._call_with_retry(
            lambda: self.client.chat.completions.create(**request_params).choices[0].message.content,
            call_type=call_type,
        )

    def _evaluate_verification(self, assertion: "Assertion", system_prompt: Optional[str] = None, context: Optional[str] = None) -> "TruthValueComponent":
        """Evaluate verification using OpenRouter API with exponential backoff retry."""
        from .truth_values import TruthValueComponent
        try:
            prompt = self._create_verification_prompt(assertion, context=context)
            sys_prompt = system_prompt or "You are an expert in factual verification. You must respond with only the exact required token sequences."
            response_text = self._raw_complete(
                prompt, sys_prompt, max_tokens=10,
                call_type=f"openrouter_verification/{self.model}"
            )
            return self._parse_verification_response(response_text)
        except Exception as e:
            print(f"ERROR [openrouter_verification/{self.model}]: {type(e).__name__}: {str(e)[:120]} [API_ERROR]")
            return TruthValueComponent.UNDEFINED

    def _evaluate_refutation(self, assertion: "Assertion", system_prompt: Optional[str] = None, context: Optional[str] = None) -> "TruthValueComponent":
        """Evaluate refutation using OpenRouter API with exponential backoff retry."""
        from .truth_values import TruthValueComponent
        try:
            prompt = self._create_refutation_prompt(assertion, context=context)
            sys_prompt = system_prompt or "You are an expert in logical refutation. You must respond with only the exact required token sequences."
            response_text = self._raw_complete(
                prompt, sys_prompt, max_tokens=10,
                call_type=f"openrouter_refutation/{self.model}"
            )
            return self._parse_refutation_response(response_text)
        except Exception as e:
            print(f"ERROR [openrouter_refutation/{self.model}]: {type(e).__name__}: {str(e)[:120]} [API_ERROR]")
            return TruthValueComponent.UNDEFINED


def get_model_info(model_name: str) -> str:
    """
    Get information about a specific model.
    
    Args:
        model_name: The model name to get information for
        
    Returns:
        A string with information about the model
    """
    try:
        provider = ModelRouter.get_provider(model_name)
        
        info = f"Model: {model_name}\n"
        info += f"Provider: {provider}\n"
        
        # Add provider-specific information
        if provider == "openai":
            info += "API Key: Set OPENAI_API_KEY environment variable\n"
            info += "Documentation: https://platform.openai.com/docs/models\n"
        elif provider == "anthropic":
            info += "API Key: Set ANTHROPIC_API_KEY environment variable\n"
            info += "Documentation: https://docs.anthropic.com/claude/docs/models-overview\n"
        elif provider == "openrouter":
            info += "API Key: Set OPENROUTER_API_KEY environment variable\n"
            info += "Documentation: https://openrouter.ai/docs\n"
            info += "Note: Provides access to many different models through a unified API\n"
        elif provider == "mock":
            info += "Description: Mock evaluator for testing/development\n"
            info += "No API key required\n"
        
        return info
        
    except ValueError as e:
        return f"Error: {e}"