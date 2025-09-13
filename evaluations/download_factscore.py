#!/usr/bin/env python3
"""
Script to help download and organize FACTScore data files.

The full FACTScore dataset needs to be downloaded from:
https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I

Files to download:
- InstructGPT.jsonl
- ChatGPT.jsonl
- PerplexityAI.jsonl
- Alpaca-7B.jsonl
- Alpaca-13B.jsonl
- Vicuna-7B.jsonl
- Vicuna-13B.jsonl
- OPT-1.3B.jsonl
- OPT-7B.jsonl
- OPT-13B.jsonl
- OPT-30B.jsonl
- human_data.jsonl
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Target directory for FACTScore data
    data_dir = Path("factscore_data/raw/full")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    expected_files = [
        "InstructGPT.jsonl",
        "ChatGPT.jsonl",
        "PerplexityAI.jsonl",
        "Alpaca-7B.jsonl",
        "Alpaca-13B.jsonl",
        "Vicuna-7B.jsonl",
        "Vicuna-13B.jsonl",
        "OPT-1.3B.jsonl",
        "OPT-7B.jsonl",
        "OPT-13B.jsonl",
        "OPT-30B.jsonl",
        "human_data.jsonl"
    ]
    
    print("\n" + "="*60)
    print("FACTScore Data Download Instructions")
    print("="*60)
    print("\n1. Visit the Google Drive folder:")
    print("   https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I")
    print("\n2. Download the following files:")
    for f in expected_files:
        print(f"   - {f}")
    
    print(f"\n3. Place the downloaded files in:")
    print(f"   {data_dir.absolute()}")
    
    print("\n4. Current status:")
    existing_files = []
    missing_files = []
    
    for fname in expected_files:
        fpath = data_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size / (1024 * 1024)  # MB
            existing_files.append(f"   ✓ {fname} ({size:.1f} MB)")
        else:
            missing_files.append(f"   ✗ {fname}")
    
    if existing_files:
        print("\n   Found files:")
        for f in existing_files:
            print(f)
    
    if missing_files:
        print("\n   Missing files:")
        for f in missing_files:
            print(f)
    else:
        print("\n   ✅ All files downloaded successfully!")
        return 0
    
    print("\n" + "="*60)
    return 1 if missing_files else 0

if __name__ == "__main__":
    sys.exit(main())