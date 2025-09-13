#!/usr/bin/env python3
"""
MMLU-Pro Data Generator

Converts MMLU-Pro dataset to standard bilateral truth evaluation format.
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

# Import MMLU-Pro components
sys.path.append(str(Path(__file__).parent.parent / "mmlu_pro"))
from data_loader import MMLUProLoader
from assertion_generator import MMLUProAssertionGenerator


class MMLUProGenerator:
    """Generates standard format dataset from MMLU-Pro."""
    
    def __init__(self):
        """Initialize generator with MMLU-Pro loader and assertion generator."""
        self.loader = MMLUProLoader()
        self.generator = MMLUProAssertionGenerator()
    
    def generate_standard_dataset(self, subjects: List[str] = None) -> Dict[str, Any]:
        """Generate standard format dataset from MMLU-Pro."""
        
        # Load MMLU-Pro data
        print("🔄 Loading MMLU-Pro dataset...")
        self.loader.load_dataset()
        
        if subjects is None:
            subjects = [s for s, count in self.loader.get_question_stats("test").items() if count > 0]
        
        assertions = []
        assertion_id_counter = 0
        total_questions = 0
        
        for subject in subjects:
            print(f"📚 Processing subject: {subject}")
            questions = self.loader.get_questions_by_subject(subject, "test")
            
            for question_data in questions:
                total_questions += 1
                question_id = question_data.get('question_id', f"{subject}_{len(assertions)}")
                question_text = question_data['question']
                category = question_data.get('category', subject)
                
                # Generate assertions from question
                question_assertions = self.generator.generate_assertions(question_data)
                
                for assertion, is_correct in question_assertions:
                    standard_assertion = {
                        "assertion_id": f"mmlupro_{assertion_id_counter:06d}",
                        "assertion_text": assertion.predicate,
                        "expected_label": "correct" if is_correct else "incorrect",
                        "context": {
                            "category": category,
                            "topic": subject,
                            "source_question": question_text,
                            "source_answer": assertion.predicate.split(" is ")[-1] if " is " in assertion.predicate else "N/A"
                        },
                        "metadata": {
                            "assertion_type": "original" if is_correct else "distractor", 
                            "generation_method": "mmlupro_correct" if is_correct else "mmlupro_distractor",
                            "question_id": question_id
                        }
                    }
                    assertions.append(standard_assertion)
                    assertion_id_counter += 1
        
        # Build standard format dataset
        dataset = {
            "metadata": {
                "benchmark": "mmlu-pro",
                "version": "1.0", 
                "total_assertions": len(assertions),
                "generation_timestamp": datetime.now().isoformat() + "Z",
                "source_info": {
                    "original_questions": total_questions,
                    "original_dataset": "MMLU-Pro",
                    "generation_params": {
                        "subjects": subjects
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
        
        print(f"✅ Generated MMLU-Pro standard dataset: {output_path}")
        print(f"   📊 {dataset['metadata']['total_assertions']} assertions from {dataset['metadata']['source_info']['original_questions']} questions")


def main():
    """Main CLI for MMLU-Pro data generation."""
    parser = argparse.ArgumentParser(description="Generate standard format dataset from MMLU-Pro")
    parser.add_argument("--output", required=True, help="Output path for standard dataset JSON")
    parser.add_argument("--subjects", nargs="+", help="Specific subjects to include (default: all)")
    
    args = parser.parse_args()
    
    # Generate complete dataset
    generator = MMLUProGenerator()
    dataset = generator.generate_standard_dataset(args.subjects)
    generator.save_dataset(dataset, args.output)


if __name__ == "__main__":
    main()