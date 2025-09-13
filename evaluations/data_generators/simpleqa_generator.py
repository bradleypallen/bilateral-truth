#!/usr/bin/env python3
"""
SimpleQA Data Generator

Converts enhanced SimpleQA dataset to standard bilateral truth evaluation format.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class SimpleQAGenerator:
    """Generates standard format dataset from enhanced SimpleQA."""
    
    def __init__(self, source_dataset_path: str):
        """Initialize generator with enhanced SimpleQA dataset path."""
        self.source_path = Path(source_dataset_path)
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source dataset not found: {source_dataset_path}")
    
    def load_enhanced_simpleqa(self) -> Dict[str, Any]:
        """Load enhanced SimpleQA dataset from JSON file."""
        with open(self.source_path, 'r') as f:
            data = json.load(f)
        return data
    
    def generate_standard_dataset(self) -> Dict[str, Any]:
        """Generate standard format dataset from enhanced SimpleQA."""
        
        # Load source data (all assertions)
        enhanced_data = self.load_enhanced_simpleqa()
        source_assertions = enhanced_data['assertions']
        
        assertions = []
        
        for idx, source_assertion in enumerate(source_assertions):
            assertion = {
                "assertion_id": f"simpleqa_{idx:06d}",
                "assertion_text": source_assertion['assertion'],
                "expected_label": source_assertion['label'],  # 'correct' or 'incorrect'
                "context": {
                    "category": source_assertion['topic'],
                    "topic": source_assertion['topic'],
                    "source_question": source_assertion['original_question'],
                    "source_answer": source_assertion.get('original_answer', '')
                },
                "metadata": {
                    "assertion_type": source_assertion.get('assertion_type', 'original'),
                    "generation_method": source_assertion.get('distractor_type', 'simpleqa_original'),
                    "answer_type": source_assertion.get('answer_type', 'unknown')
                }
            }
            assertions.append(assertion)
        
        # Build standard format dataset
        dataset = {
            "metadata": {
                "benchmark": "simpleqa",
                "version": "1.0",
                "total_assertions": len(assertions),
                "generation_timestamp": datetime.now().isoformat() + "Z",
                "source_info": {
                    "original_questions": enhanced_data.get('config', {}).get('total_original_questions', len(assertions)),
                    "original_dataset": str(self.source_path),
                    "generation_params": {
                        "source_config": enhanced_data.get('config', {})
                    }
                }
            },
            "assertions": assertions
        }
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any], output_path: str):
        """Save dataset to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✅ Generated SimpleQA standard dataset: {output_path}")
        print(f"   📊 {dataset['metadata']['total_assertions']} assertions from {dataset['metadata']['source_info']['original_questions']} questions")


def main():
    """Main CLI for SimpleQA data generation."""
    parser = argparse.ArgumentParser(description="Generate standard format dataset from enhanced SimpleQA")
    parser.add_argument("--source", required=True, help="Path to enhanced SimpleQA JSON file")
    parser.add_argument("--output", required=True, help="Output path for standard dataset JSON")
    
    args = parser.parse_args()
    
    # Generate complete dataset
    generator = SimpleQAGenerator(args.source)
    dataset = generator.generate_standard_dataset()
    generator.save_dataset(dataset, args.output)


if __name__ == "__main__":
    main()