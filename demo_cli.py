#!/usr/bin/env python3
"""
Demo script showing the bilateral-truth CLI functionality.
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and print its output."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    print("-" * 50)

def main():
    """Run CLI demo."""
    print("bilateral-truth CLI Demo")
    print("=" * 50)
    
    # Show available models
    run_command("python cli.py --list-models | head -20")
    
    # Show model info
    run_command("python cli.py --model-info claude")
    
    # Evaluate some assertions with mock model
    assertions = [
        "The sky is blue",
        "The Earth is flat", 
        "It will rain tomorrow",
        "2 + 2 = 4",
        "Climate change is real"
    ]
    
    print("Evaluating assertions with mock model:")
    print()
    
    for assertion in assertions:
        cmd = f'python cli.py --model mock --verbose "{assertion}"'
        run_command(cmd)
    
    # Show cache size
    run_command("python cli.py --cache-size")
    
    print("\nDemo complete!")
    print("\nTo try interactive mode, run:")
    print("  python cli.py --model mock --interactive")
    print("\nTo use real LLM models, set API keys:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  export ANTHROPIC_API_KEY='your-key'") 
    print("  export OPENROUTER_API_KEY='your-key'")

if __name__ == "__main__":
    main()