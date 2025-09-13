#!/usr/bin/env python3
"""
Automated FACTScore data downloader using gdown.
"""

import os
import sys
from pathlib import Path
import gdown
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Direct file IDs from the Google Drive folder
# These IDs are from: https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I
FILE_IDS = {
    "ChatGPT.jsonl": "1LWWbGhqpbGA7dYGsX34pIgqPJLXDEgJT",
    "InstructGPT.jsonl": "1KwwLvLSvTzF7FXlvJx5QVqmhTPr1HQTC",
    "PerplexityAI.jsonl": "1LJ0UocW0x6UwnPzkdra7AvGS76qy4NUb",
    "human_data.jsonl": "1UmA-sJFJq0Fz0bJM7b1OIJBr6DzMdcJ8",
    # Model-specific files
    "Alpaca-7B.jsonl": "1KjdvqmZM7pR3R7dsOPkF66cKJLhFdBT8",
    "Alpaca-13B.jsonl": "1KTLRHV-vPvwLrVGatLOY6dIDvrCrfuXU",
    "Vicuna-7B.jsonl": "1LRJpGj8r0gR5rcGyyjAlwav8FXDKdf3k",
    "Vicuna-13B.jsonl": "1LOSFvLjx8F-EKXE-PZ_pP1IForPC0qUZ",
    # OPT models - these might not have direct IDs, will try folder download
}

def download_file(file_id, output_path):
    """Download a file from Google Drive using its ID."""
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        logger.error(f"Failed to download {output_path}: {e}")
        return False

def main():
    # Target directory
    data_dir = Path("factscore_data/raw/full")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to download the entire folder first
    folder_url = "https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I"
    
    print("Attempting to download FACTScore data from Google Drive...")
    print(f"Target directory: {data_dir.absolute()}")
    print("-" * 60)
    
    # Try folder download
    try:
        print("\nAttempting to download entire folder...")
        gdown.download_folder(folder_url, output=str(data_dir), quiet=False, use_cookies=False)
        print("✅ Folder download successful!")
    except Exception as e:
        print(f"⚠️ Folder download failed: {e}")
        print("\nAttempting individual file downloads...")
        
        # Try individual files
        success_count = 0
        for filename, file_id in FILE_IDS.items():
            output_path = data_dir / filename
            print(f"\nDownloading {filename}...")
            if download_file(file_id, str(output_path)):
                success_count += 1
                print(f"  ✓ {filename} downloaded")
            else:
                print(f"  ✗ {filename} failed")
        
        print(f"\n{success_count}/{len(FILE_IDS)} files downloaded successfully")
    
    # Check what we have
    print("\n" + "="*60)
    print("Current status:")
    files = list(data_dir.glob("*.jsonl"))
    if files:
        print(f"Found {len(files)} JSONL files:")
        for f in sorted(files):
            size = f.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {f.name} ({size:.1f} MB)")
    else:
        print("No JSONL files found yet.")
    
    return 0 if files else 1

if __name__ == "__main__":
    sys.exit(main())