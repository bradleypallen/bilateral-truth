#!/usr/bin/env python3
"""
Demo of what interactive mode looks like.
"""

import subprocess
import sys
import os

def main():
    print("Demo: What interactive mode looks like")
    print("=" * 40)
    
    # Show what the interactive mode startup looks like
    cmd = 'echo "quit" | python cli.py -m llama3 -i'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    print("Interactive mode startup:")
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

if __name__ == "__main__":
    main()