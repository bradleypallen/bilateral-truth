#!/usr/bin/env python3
"""
TruthfulQA Data Generator

Converts TruthfulQA dataset to standard bilateral truth evaluation format.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TruthfulQAGenerator:
    """Generates standard format dataset from TruthfulQA."""
    
    def __init__(self, source_dataset_path: str):
        """Initialize generator with source dataset path."""
        self.source_path = Path(source_dataset_path)
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source dataset not found: {source_dataset_path}")
    
    def load_truthfulqa_data(self, max_samples: int = None) -> pd.DataFrame:
        """Load TruthfulQA data from CSV file."""
        df = pd.read_csv(self.source_path)
        
        if max_samples:
            df = df.head(max_samples)
            
        return df
    
    def generate_standard_dataset(self) -> Dict[str, Any]:
        """Generate standard format dataset from TruthfulQA."""
        
        # Load source data (all questions)
        df = self.load_truthfulqa_data()
        
        assertions = []
        assertion_id_counter = 0
        
        for idx, row in df.iterrows():
            question = row['Question']
            correct_answers = row['Correct Answers'].split('; ') if pd.notna(row['Correct Answers']) else []
            incorrect_answers = row['Incorrect Answers'].split('; ') if pd.notna(row['Incorrect Answers']) else []
            category = row['Category'] if 'Category' in df.columns else 'Unknown'
            
            # Generate correct assertion (use first correct answer)
            if correct_answers:
                correct_assertion = {
                    "assertion_id": f"truthfulqa_{assertion_id_counter:06d}",
                    "assertion_text": correct_answers[0].strip(),
                    "expected_label": "correct",
                    "context": {
                        "category": category,
                        "topic": category,  # TruthfulQA uses category as topic
                        "source_question": question,
                        "source_answer": correct_answers[0].strip()
                    },
                    "metadata": {
                        "assertion_type": "original",
                        "generation_method": "truthfulqa_correct"
                    }
                }
                assertions.append(correct_assertion)
                assertion_id_counter += 1
            
            # Generate incorrect assertion (use first incorrect answer)
            if incorrect_answers:
                incorrect_assertion = {
                    "assertion_id": f"truthfulqa_{assertion_id_counter:06d}",
                    "assertion_text": incorrect_answers[0].strip(),
                    "expected_label": "incorrect", 
                    "context": {
                        "category": category,
                        "topic": category,
                        "source_question": question,
                        "source_answer": incorrect_answers[0].strip()
                    },
                    "metadata": {
                        "assertion_type": "distractor",
                        "generation_method": "truthfulqa_incorrect"
                    }
                }
                assertions.append(incorrect_assertion)
                assertion_id_counter += 1
        
        # Build standard format dataset
        dataset = {
            "metadata": {
                "benchmark": "truthfulqa",
                "version": "1.0",
                "total_assertions": len(assertions),
                "generation_timestamp": datetime.now().isoformat() + "Z",
                "source_info": {
                    "original_questions": len(df),
                    "original_dataset": str(self.source_path),
                    "generation_params": {}
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
        
        print(f"✅ Generated TruthfulQA standard dataset: {output_path}")
        print(f"   📊 {dataset['metadata']['total_assertions']} assertions from {dataset['metadata']['source_info']['original_questions']} questions")


def main():
    """Main CLI for TruthfulQA data generation."""
    parser = argparse.ArgumentParser(description="Generate standard format dataset from TruthfulQA")
    parser.add_argument("--source", required=True, help="Path to TruthfulQA CSV file")
    parser.add_argument("--output", required=True, help="Output path for standard dataset JSON")
    
    args = parser.parse_args()
    
    # Generate complete dataset
    generator = TruthfulQAGenerator(args.source)
    dataset = generator.generate_standard_dataset()
    generator.save_dataset(dataset, args.output)


if __name__ == "__main__":
    main()