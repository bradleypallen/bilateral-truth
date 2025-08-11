#!/usr/bin/env python3
"""
Integration Test Runner for bilateral-truth LLM Evaluators

This script runs integration tests that make real API calls to verify
that the separate verification/refutation approach works correctly
with actual LLM providers.

Usage:
    python run_integration_tests.py [options]

Options:
    --provider PROVIDER    Run tests for specific provider only (openai, anthropic, openrouter, all)
    --quick               Run only quick tests (single API calls)
    --sampling            Run sampling tests (more API calls)
    --comparison          Run cross-provider comparison tests
    --verbose             Verbose output
    --help               Show this help message

Prerequisites:
    Set up API keys in environment variables or .env file:
    - OPENAI_API_KEY      (for OpenAI tests)
    - ANTHROPIC_API_KEY   (for Anthropic tests) 
    - OPENROUTER_API_KEY  (for OpenRouter tests)

Note: These tests make real API calls and may incur costs.
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

def check_api_keys():
    """Check which API keys are available."""
    keys = {
        'openai': bool(os.getenv('OPENAI_API_KEY')),
        'anthropic': bool(os.getenv('ANTHROPIC_API_KEY')), 
        'openrouter': bool(os.getenv('OPENROUTER_API_KEY'))
    }
    return keys

def load_env_file():
    """Try to load .env file."""
    try:
        from dotenv import load_dotenv
        current_dir = Path.cwd()
        
        # Check current directory and up to 3 parent directories
        for i in range(4):
            env_file = current_dir / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                return True
            if current_dir.parent == current_dir:
                break
            current_dir = current_dir.parent
        
        return False
    except ImportError:
        print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
        return False

def show_api_key_status():
    """Display the status of API keys."""
    print("\n📋 API Key Status:")
    keys = check_api_keys()
    
    for provider, available in keys.items():
        status = "✓ Available" if available else "✗ Missing"
        env_var = f"{provider.upper()}_API_KEY"
        if provider == 'openrouter':
            env_var = "OPENROUTER_API_KEY"
        print(f"   {provider.capitalize()}: {status} ({env_var})")
    
    if not any(keys.values()):
        print("\n⚠️  No API keys found. You can:")
        print("   1. Create a .env file with your API keys")
        print("   2. Set environment variables directly")
        print("   3. Use mock tests instead: pytest tests/test_llm_evaluators.py")
    
    return keys

def build_pytest_command(provider=None, quick=False, sampling=False, comparison=False, verbose=False):
    """Build the pytest command based on options."""
    cmd = ["python", "-m", "pytest", "tests/test_integration.py"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    # Add markers based on what to run
    markers = []
    
    if provider and provider != "all":
        if provider == "openai":
            markers.append("TestOpenAIIntegration")
        elif provider == "anthropic": 
            markers.append("TestAnthropicIntegration")
        elif provider == "openrouter":
            markers.append("TestOpenRouterIntegration")
    
    if quick:
        # Exclude sampling and comparison tests
        markers.extend(["not TestSamplingIntegration", "not TestCrossProviderComparison"])
    elif sampling:
        markers.append("TestSamplingIntegration")
    elif comparison:
        markers.append("TestCrossProviderComparison")
    
    if markers:
        # Join markers with 'and' and 'not' logic
        if any("not" in marker for marker in markers):
            # Complex logic - use multiple -k options
            for marker in markers:
                cmd.extend(["-k", marker])
        else:
            # Simple logic - join with 'or'
            cmd.extend(["-k", " or ".join(markers)])
    
    return cmd

def main():
    parser = argparse.ArgumentParser(
        description="Run integration tests for bilateral-truth LLM evaluators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_integration_tests.py --quick --verbose
  python run_integration_tests.py --provider openai
  python run_integration_tests.py --sampling
  python run_integration_tests.py --comparison --verbose
        """
    )
    
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "openrouter", "all"],
        default="all",
        help="Run tests for specific provider only (default: all)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (single API calls)"
    )
    
    parser.add_argument(
        "--sampling", 
        action="store_true",
        help="Run sampling tests (more API calls, higher cost)"
    )
    
    parser.add_argument(
        "--comparison",
        action="store_true", 
        help="Run cross-provider comparison tests"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output showing API responses"
    )
    
    parser.add_argument(
        "--check-keys",
        action="store_true",
        help="Check API key status and exit"
    )
    
    args = parser.parse_args()
    
    print("🧪 bilateral-truth LLM Evaluator Integration Tests")
    print("=" * 50)
    
    # Try to load .env file
    load_env_file()
    
    # Check API keys
    keys = show_api_key_status()
    
    if args.check_keys:
        return 0
    
    if not any(keys.values()):
        print("\n❌ No API keys available. Cannot run integration tests.")
        return 1
    
    # Warn about costs
    if not args.quick:
        print(f"\n💰 Warning: Integration tests make real API calls which may incur costs.")
        print(f"   Use --quick to run minimal tests.")
        
        if not args.verbose:
            response = input("Continue? (y/N): ").strip().lower()
            if response != 'y':
                print("Aborted.")
                return 0
    
    # Build and run pytest command
    cmd = build_pytest_command(
        provider=args.provider,
        quick=args.quick,
        sampling=args.sampling, 
        comparison=args.comparison,
        verbose=args.verbose
    )
    
    print(f"\n🚀 Running command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())