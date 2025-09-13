#!/usr/bin/env python3
"""
FACTScore data generator for bilateral truth evaluation.

This module downloads and processes the FACTScore biography dataset,
converting atomic facts into a format suitable for bilateral truth evaluation.

Since the FACTScore package has incompatible dependencies (torch < 2.0),
we'll manually process the data files from their Google Drive.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FACTScoreDataGenerator:
    """Generator for converting FACTScore data to bilateral truth format."""
    
    # Known URLs for FACTScore data files (from their repository)
    # These need to be manually downloaded from Google Drive:
    # https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I
    
    EXPECTED_FILES = [
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
        "human_data.jsonl",
        # Sample files for testing
        "sample_ChatGPT.jsonl",
        "sample_InstructGPT.jsonl"
    ]
    
    def __init__(self, data_dir: str = "evaluations/factscore_data"):
        """Initialize the generator with a data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)
        self.output_file = self.data_dir.parent / "standard_datasets" / "factscore_complete.json"
        
    def check_data_files(self) -> List[str]:
        """Check which FACTScore data files are available."""
        available = []
        
        # Check for full dataset in extracted folder
        full_data_dir = self.raw_dir / "full" / "data" / "labeled"
        if full_data_dir.exists():
            for filename in full_data_dir.glob("*.jsonl"):
                available.append(str(filename))
            if available:
                logger.info(f"Using full dataset from {full_data_dir}")
                return available
        
        # Fall back to checking individual files in raw directory
        for filename in self.EXPECTED_FILES:
            if (self.raw_dir / filename).exists():
                available.append(str(self.raw_dir / filename))
        return available
    
    def load_factscore_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load a FACTScore JSONL file."""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def convert_to_standard_format(self) -> Dict[str, Any]:
        """
        Convert FACTScore data to standard bilateral truth format.
        
        Uses the standard format required by generic_evaluator.py
        """
        available_files = self.check_data_files()
        
        if not available_files:
            logger.warning(f"No FACTScore data files found in {self.raw_dir}")
            logger.info("Please download the data files from:")
            logger.info("https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I")
            logger.info(f"And place them in: {self.raw_dir}")
            return None
        
        logger.info(f"Found {len(available_files)} FACTScore data files")
        
        all_assertions = []
        assertion_counter = 0
        
        for filepath in available_files:
            filepath = Path(filepath)
            model_name = filepath.stem  # Get filename without extension
            
            logger.info(f"Processing {filepath.name}...")
            data = self.load_factscore_file(filepath)
            
            for entry_idx, entry in enumerate(data):
                # Handle new format with annotations
                if "annotations" in entry and entry.get("annotations"):
                    # New format from labeled data
                    topic = entry.get("topic", "")
                    input_prompt = entry.get("input", "")
                    output_text = entry.get("output", "")
                    categories = entry.get("cat", [])
                    
                    annotations = entry.get("annotations", [])
                    if annotations is None:
                        annotations = []
                    
                    for ann_idx, annotation in enumerate(annotations):
                        # Each annotation contains atomic facts
                        human_facts = annotation.get("human-atomic-facts", []) if annotation else []
                        model_facts = annotation.get("model-atomic-facts", []) if annotation else []
                        
                        # Process human-annotated facts
                        for fact_data in human_facts or []:
                            assertion_counter += 1
                            fact_text = fact_data.get("text", "")
                            label = fact_data.get("label", "NS")  # S=Supported, NS=Not Supported
                            
                            expected_label = "correct" if label == "S" else "incorrect"
                            
                            assertion = {
                                "assertion_id": f"factscore_{model_name}_{entry_idx}_{ann_idx}_h{assertion_counter}",
                                "assertion_text": fact_text,
                                "expected_label": expected_label,
                                "context": {
                                    "category": "Biography",
                                    "topic": topic,
                                    "difficulty": categories[0] if categories else "standard",
                                    "source_question": input_prompt,
                                    "source_answer": output_text[:200] + "..." if len(output_text) > 200 else output_text
                                },
                                "metadata": {
                                    "assertion_type": "human_annotated",
                                    "generation_method": model_name,
                                    "annotation_label": label,
                                    "annotation_source": "human",
                                    "categories": categories
                                }
                            }
                            all_assertions.append(assertion)
                        
                        # Process model-generated facts (if needed)
                        for fact_idx, fact_data in enumerate(model_facts or []):
                            if isinstance(fact_data, dict):
                                fact_text = fact_data.get("text", "")
                            else:
                                fact_text = str(fact_data)
                            
                            if fact_text:
                                assertion_counter += 1
                                assertion = {
                                    "assertion_id": f"factscore_{model_name}_{entry_idx}_{ann_idx}_m{fact_idx}",
                                    "assertion_text": fact_text,
                                    "expected_label": "correct",  # Default, as these don't have labels
                                    "context": {
                                        "category": "Biography",
                                        "topic": topic,
                                        "difficulty": categories[0] if categories else "standard",
                                        "source_question": input_prompt,
                                        "source_answer": output_text[:200] + "..." if len(output_text) > 200 else output_text
                                    },
                                    "metadata": {
                                        "assertion_type": "model_generated",
                                        "generation_method": model_name,
                                        "annotation_source": "model",
                                        "categories": categories
                                    }
                                }
                                all_assertions.append(assertion)
                
                else:
                    # Old format (for sample files)
                    prompt = entry.get("prompt", "")
                    facts = entry.get("facts", [])
                    llama_labels = entry.get("LLAMA+NP_labels", [])
                    chatgpt_labels = entry.get("ChatGPT_labels", [])
                    
                    # Extract person name from prompt
                    person = prompt.replace("Tell me a bio of ", "").rstrip(".")
                    
                    # Process each atomic fact
                    for fact_idx, fact in enumerate(facts):
                        assertion_counter += 1
                        
                        # Determine expected label based on ground truth labels
                        llama_label = llama_labels[fact_idx] if fact_idx < len(llama_labels) else None
                        chatgpt_label = chatgpt_labels[fact_idx] if fact_idx < len(chatgpt_labels) else None
                        
                        # Use majority voting or ChatGPT label as ground truth
                        if chatgpt_label is not None:
                            expected_label = "correct" if chatgpt_label else "incorrect"
                        elif llama_label is not None:
                            expected_label = "correct" if llama_label else "incorrect"
                        else:
                            expected_label = "correct"  # Default to correct if no labels
                        
                        assertion = {
                            "assertion_id": f"factscore_{model_name}_{entry_idx}_{fact_idx}",
                            "assertion_text": fact,
                            "expected_label": expected_label,
                            "context": {
                                "category": "Biography",
                                "topic": person,
                                "difficulty": "factual",
                                "source_question": prompt,
                                "source_answer": f"Generated biography by {model_name}"
                            },
                            "metadata": {
                                "assertion_type": "original",
                                "generation_method": model_name,
                                "llama_label": llama_label,
                                "chatgpt_label": chatgpt_label,
                                "fact_index": fact_idx,
                                "total_facts": len(facts)
                            }
                        }
                        all_assertions.append(assertion)
        
        logger.info(f"Processed {len(all_assertions)} atomic facts from {len(available_files)} models")
        
        from datetime import datetime
        
        return {
            "metadata": {
                "benchmark": "factscore",
                "version": "2023_biographies",
                "total_assertions": len(all_assertions),
                "generation_timestamp": datetime.now().isoformat() + "Z",
                "source_info": {
                    "original_questions": len(available_files) * 500,  # Max 500 per model
                    "original_dataset": "FACTScore biography generation",
                    "generation_params": {
                        "models": [f.replace(".jsonl", "") for f in available_files],
                        "description": "Atomic facts from LLM-generated biographies"
                    }
                }
            },
            "assertions": all_assertions
        }
    
    def save_standard_format(self, data: Dict[str, Any]):
        """Save the converted data in standard format."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {data['metadata']['total_assertions']} assertions to {self.output_file}")
    
    def generate(self):
        """Main generation pipeline."""
        logger.info("Starting FACTScore data generation...")
        
        # Check for existing files
        available = self.check_data_files()
        if not available:
            print("\n" + "="*60)
            print("FACTScore Data Download Instructions")
            print("="*60)
            print("\n1. Visit the Google Drive link:")
            print("   https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I")
            print("\n2. Download the following JSONL files:")
            for f in self.EXPECTED_FILES:
                print(f"   - {f}")
            print(f"\n3. Place them in: {self.raw_dir.absolute()}")
            print("\n4. Re-run this script")
            print("="*60 + "\n")
            return
        
        # Convert to standard format
        data = self.convert_to_standard_format()
        
        if data:
            # Save the converted data
            self.save_standard_format(data)
            
            # Print summary statistics
            print("\n" + "="*60)
            print("FACTScore Data Generation Summary")
            print("="*60)
            print(f"Total atomic facts: {data['metadata']['total_assertions']}")
            print(f"Models included: {', '.join(data['metadata']['source_info']['generation_params']['models'])}")
            
            # Count facts with labels
            with_llama = sum(1 for item in data['assertions'] if item['metadata'].get('llama_label') is not None)
            with_chatgpt = sum(1 for item in data['assertions'] if item['metadata'].get('chatgpt_label') is not None)
            print(f"Facts with LLAMA+NP labels: {with_llama}")
            print(f"Facts with ChatGPT labels: {with_chatgpt}")
            
            # Sample fact
            if data['assertions']:
                sample = data['assertions'][0]
                print(f"\nSample fact:")
                print(f"  Model: {sample['metadata']['generation_method']}")
                print(f"  Person: {sample['context']['topic']}")
                print(f"  Fact: {sample['assertion_text'][:100]}...")
                print(f"  Expected label: {sample['expected_label']}")
                print(f"  LLAMA label: {sample['metadata'].get('llama_label')}")
                print(f"  ChatGPT label: {sample['metadata'].get('chatgpt_label')}")
            print("="*60 + "\n")


if __name__ == "__main__":
    generator = FACTScoreDataGenerator()
    generator.generate()