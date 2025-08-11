#!/usr/bin/env python3
"""
Standalone CLI script for bilateral-truth bilateral factuality evaluation.

This script can be run directly without installing the package.
"""

import sys
import os

# Add the current directory to the path so we can import bilateral_truth
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bilateral_truth.cli import main

if __name__ == "__main__":
    sys.exit(main())