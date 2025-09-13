#!/usr/bin/env python3
"""
Forced-Choice Unilateral Evaluator

Implements true unilateral evaluation with forced binary choice (CORRECT/INCORRECT)
as described in the ArXiv paper 2507.09751v2. This provides a baseline for comparison
with bilateral evaluation results.

Key differences from bilateral evaluation:
- Single prompt asking "Is this correct?"
- Forced binary choice (no uncertainty allowed)
- 100% coverage (always returns a judgment)
"""

import json
import time
import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bilateral_truth.model_router import ModelRouter
from bilateral_truth.assertions import Assertion

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"🔑 Loaded environment variables from {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed")


class UnilateralEvaluator:
    """Forced-choice unilateral evaluator for direct comparison with bilateral evaluation."""
    
    def __init__(self, model_name: str, dataset_path: str,
                 checkpoint_dir: str = "checkpoints",
                 prompt_style: str = "direct"):
        """Initialize unilateral evaluator.
        
        Args:
            model_name: Name of the model to evaluate
            dataset_path: Path to standard format dataset JSON
            checkpoint_dir: Directory to save checkpoints for recovery
            prompt_style: Style of prompt ('direct', 'zero-shot', 'few-shot')
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.prompt_style = prompt_style
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.evaluator = ModelRouter.create_evaluator(model_name)
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Results tracking
        self.results = {
            'model': model_name,
            'dataset': self.dataset['metadata']['benchmark'],
            'evaluation_type': 'unilateral_forced_choice' if prompt_style != 'uncertain' else 'unilateral_with_uncertainty',
            'prompt_style': prompt_style,
            'total_samples': 0,
            'correct': 0,
            'incorrect': 0,
            'uncertain': 0,  # Track uncertain responses
            'category_analysis': {},
            'detailed_results': []
        }
        
        # Generate checkpoint ID
        self.checkpoint_id = self._generate_checkpoint_id()
        self.checkpoint_path = self.checkpoint_dir / f"unilateral_{self.checkpoint_id}.json"
    
    def _load_dataset(self) -> Dict:
        """Load standard format dataset."""
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)
            
        # Validate dataset format - EXACTLY like generic_evaluator.py
        if 'metadata' not in dataset or 'assertions' not in dataset:
            raise ValueError(f"Invalid dataset format. Missing 'metadata' or 'assertions'.")
        
        print(f"📊 Loaded dataset: {dataset['metadata']['benchmark']}")
        print(f"   📝 {dataset['metadata']['total_assertions']} assertions")
        print(f"   📅 Generated: {dataset['metadata']['generation_timestamp']}")
        
        return dataset
    
    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID based on evaluation parameters."""
        params = f"{self.model_name}_{self.dataset_path}_{self.prompt_style}"
        return hashlib.md5(params.encode()).hexdigest()[:8]
    
    def _save_checkpoint(self, processed_indices: set):
        """Save checkpoint for recovery."""
        checkpoint_data = {
            'processed_indices': list(processed_indices),
            'results': self.results,
            'timestamp': time.time()
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
    
    def _load_checkpoint(self) -> Optional[set]:
        """Load checkpoint if it exists."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)
                self.results = data['results']
                print(f"📥 Loaded checkpoint with {len(data['processed_indices'])} processed items")
                return set(data['processed_indices'])
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
        return None
    
    def _create_unilateral_prompt(self, assertion: str, context: str = "") -> str:
        """Create unilateral prompt based on style."""
        
        if self.prompt_style == "direct":
            return f"""Determine whether the following statement is correct.
Conclude with a single line containing ONLY one of these two words:
CORRECT
INCORRECT

Statement: {assertion}"""

        elif self.prompt_style == "uncertain":
            return f"""Determine whether the following statement is correct.

You have three options:
1. CORRECT - if you are confident the statement is true
2. INCORRECT - if you are confident the statement is false
3. UNCERTAIN - if you lack sufficient confidence to determine correctness

Important: Only respond CORRECT or INCORRECT if you are reasonably confident.
If you have doubts or the statement is ambiguous, respond UNCERTAIN.

Conclude with a single line containing ONLY one of these three words:
CORRECT
INCORRECT
UNCERTAIN

Statement: {assertion}"""

        elif self.prompt_style == "zero-shot":
            return f"""I'll provide you with a statement.
Your task is to determine whether this statement is correct by following these steps:

1. Analyze the statement carefully
2. Consider what makes it correct or incorrect
3. Check for factual accuracy
4. Make your determination

After your analysis, conclude with a single line containing ONLY one of these two words:
CORRECT
INCORRECT

Statement: {assertion}"""

        else:  # few-shot
            return f"""Here are examples of evaluating statements:

Example 1:
Statement: Paris is the capital of France.
Analysis: This is a factual statement about geography. Paris is indeed the capital city of France.
CORRECT

Example 2:
Statement: The Earth orbits around Mars.
Analysis: This is incorrect. The Earth orbits around the Sun, not Mars.
INCORRECT

Now evaluate this statement:
Statement: {assertion}

Conclude with a single line containing ONLY one of these two words:
CORRECT
INCORRECT"""
    
    def _evaluate_single_unilateral(self, assertion: str, context: str = "") -> str:
        """Evaluate a single assertion with forced binary choice."""
        prompt = self._create_unilateral_prompt(assertion, context)
        
        # Use a simple system prompt for unilateral evaluation
        system_prompt = "You are an expert evaluator determining the correctness of statements."
        
        # Retry logic for API failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get model response directly from the LLM client
                # We need to access the underlying client since evaluators only have bilateral methods
                if hasattr(self.evaluator, 'client'):  # OpenAI/Anthropic evaluators
                    if hasattr(self.evaluator.client, 'chat'):  # OpenAI
                        response = self.evaluator.client.chat.completions.create(
                            model=self.evaluator.model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.1,
                            max_tokens=100
                        ).choices[0].message.content
                        break  # Success, exit retry loop
                    else:  # Anthropic
                        response = self.evaluator.client.messages.create(
                            model=self.evaluator.model,
                            system=system_prompt,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1,
                            max_tokens=100
                        ).content[0].text
                        break  # Success, exit retry loop
                elif hasattr(self.evaluator, '_call_api'):  # OpenRouter evaluator
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    response = self.evaluator._call_api(messages, temperature=0.1, max_tokens=100)
                    break  # Success, exit retry loop
                else:  # MockLLMEvaluator
                    # For mock, just return a deterministic response based on assertion hash
                    import hashlib
                    hash_val = int(hashlib.md5(assertion.encode()).hexdigest(), 16)
                    response = "CORRECT" if hash_val % 2 == 0 else "INCORRECT"
                    break  # Success, exit retry loop
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  API error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print(f"❌ Failed after {max_retries} attempts: {str(e)[:100]}")
                    # Default to INCORRECT on persistent failure
                    response = "INCORRECT"
        
        # Parse response to extract CORRECT/INCORRECT/UNCERTAIN
        response_upper = response.upper()
        if "UNCERTAIN" in response_upper:
            return "UNCERTAIN"
        elif "CORRECT" in response_upper and "INCORRECT" not in response_upper:
            return "CORRECT"
        elif "INCORRECT" in response_upper:
            return "INCORRECT"
        else:
            # Default handling based on prompt style
            if self.prompt_style == "uncertain":
                # For uncertain style, default to UNCERTAIN for unparseable
                print(f"⚠️  Unparseable response, defaulting to UNCERTAIN: {response[:100]}")
                return "UNCERTAIN"
            else:
                # For forced-choice styles, default to INCORRECT
                print(f"⚠️  Unparseable response, defaulting to INCORRECT: {response[:100]}")
                return "INCORRECT"
    
    def evaluate_dataset(self, sample_size: Optional[int] = None) -> Dict:
        """Evaluate dataset with forced-choice unilateral evaluation.
        
        Args:
            sample_size: Number of samples to evaluate (None for all)
            
        Returns:
            Evaluation results dictionary
        """
        # Load checkpoint if exists
        processed_indices = self._load_checkpoint() or set()
        
        # Get items to process - use 'assertions' like generic_evaluator.py
        items = self.dataset['assertions']
        if sample_size:
            items = items[:sample_size]
        
        total_items = len(items)
        print(f"\n🎯 Evaluating {total_items} items with forced-choice unilateral evaluation")
        print(f"📊 Model: {self.model_name}")
        print(f"📝 Prompt style: {self.prompt_style}")
        
        # Process each item
        for idx, item in enumerate(items):
            if idx in processed_indices:
                continue
            
            # Progress indicator
            if idx % 10 == 0:
                print(f"Evaluating {idx + 1}/{total_items}")
            
            # Get assertion and ground truth - match generic_evaluator.py field names
            assertion = item['assertion_text']
            ground_truth = item['expected_label']  # Will be 'correct' or 'incorrect'
            category = item.get('context', {}).get('category', 'uncategorized')
            
            # Evaluate with unilateral prompt
            prediction = self._evaluate_single_unilateral(assertion)
            
            # Determine if correct - ground_truth is 'correct' or 'incorrect'
            # UNCERTAIN is treated as abstention (counts as incorrect for accuracy)
            if prediction == "UNCERTAIN":
                is_correct = False
                abstained = True
            else:
                is_correct = (
                    (prediction == "CORRECT" and ground_truth == "correct") or
                    (prediction == "INCORRECT" and ground_truth == "incorrect")
                )
                abstained = False
            
            # Store detailed result
            detailed_result = {
                'index': idx,
                'assertion': assertion,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'is_correct': is_correct,
                'abstained': abstained if self.prompt_style == "uncertain" else False,
                'category': category
            }
            self.results['detailed_results'].append(detailed_result)
            
            # Update counters
            self.results['total_samples'] += 1
            if prediction == "UNCERTAIN":
                self.results['uncertain'] += 1
            elif is_correct:
                self.results['correct'] += 1
            else:
                self.results['incorrect'] += 1
            
            # Update category analysis
            if category not in self.results['category_analysis']:
                self.results['category_analysis'][category] = {
                    'total': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'uncertain': 0
                }
            
            self.results['category_analysis'][category]['total'] += 1
            if prediction == "UNCERTAIN":
                self.results['category_analysis'][category]['uncertain'] += 1
            elif is_correct:
                self.results['category_analysis'][category]['correct'] += 1
            else:
                self.results['category_analysis'][category]['incorrect'] += 1
            
            # Update processed set and save checkpoint
            processed_indices.add(idx)
            if idx % 10 == 0:  # Save checkpoint every 10 items
                self._save_checkpoint(processed_indices)
        
        # Calculate final metrics
        self._calculate_metrics()
        
        # Clean up checkpoint
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            print(f"🧹 Cleaned up checkpoint")
        
        return self.results
    
    def _calculate_metrics(self):
        """Calculate evaluation metrics."""
        total = self.results['total_samples']
        if total == 0:
            return
        
        # Overall metrics
        self.results['accuracy'] = self.results['correct'] / total
        
        # Coverage depends on prompt style
        if self.prompt_style == "uncertain":
            # Coverage = percentage of non-uncertain responses
            self.results['coverage'] = (total - self.results['uncertain']) / total
            self.results['uncertainty_rate'] = self.results['uncertain'] / total
            # Accuracy on answered questions only
            answered = total - self.results['uncertain']
            if answered > 0:
                self.results['accuracy_on_answered'] = self.results['correct'] / answered
            else:
                self.results['accuracy_on_answered'] = 0
        else:
            # Always 100% for forced choice
            self.results['coverage'] = 1.0
        
        # Calculate precision, recall, F1 for binary classification
        # True positives: predicted CORRECT when ground truth is correct
        # False positives: predicted CORRECT when ground truth is incorrect
        # True negatives: predicted INCORRECT when ground truth is incorrect
        # False negatives: predicted INCORRECT when ground truth is correct
        
        tp = sum(1 for r in self.results['detailed_results'] 
                 if r['prediction'] == 'CORRECT' and r['ground_truth'] == 'correct')
        fp = sum(1 for r in self.results['detailed_results']
                 if r['prediction'] == 'CORRECT' and r['ground_truth'] == 'incorrect')
        tn = sum(1 for r in self.results['detailed_results']
                 if r['prediction'] == 'INCORRECT' and r['ground_truth'] == 'incorrect')
        fn = sum(1 for r in self.results['detailed_results']
                 if r['prediction'] == 'INCORRECT' and r['ground_truth'] == 'correct')
        
        # Precision, recall, F1 for "true" class
        precision_true = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_true = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_true = 2 * precision_true * recall_true / (precision_true + recall_true) \
                  if (precision_true + recall_true) > 0 else 0
        
        # Precision, recall, F1 for "false" class  
        precision_false = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_false = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_false = 2 * precision_false * recall_false / (precision_false + recall_false) \
                   if (precision_false + recall_false) > 0 else 0
        
        # Macro F1 (average of F1 scores)
        self.results['f1_macro'] = (f1_true + f1_false) / 2
        self.results['precision_macro'] = (precision_true + precision_false) / 2
        self.results['recall_macro'] = (recall_true + recall_false) / 2
        
        # Store detailed metrics
        self.results['metrics'] = {
            'confusion_matrix': {
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            },
            'true_class': {
                'precision': precision_true,
                'recall': recall_true,
                'f1': f1_true
            },
            'false_class': {
                'precision': precision_false,
                'recall': recall_false,
                'f1': f1_false
            }
        }
        
        # Category-specific metrics
        for category, stats in self.results['category_analysis'].items():
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']
    
    def save_results(self, output_dir: str = "results"):
        """Save evaluation results to file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create filename
        dataset_name = Path(self.dataset_path).stem
        model_safe = self.model_name.replace('/', '_').replace(':', '_')
        output_file = output_path / f"{dataset_name}_{model_safe}_unilateral_{self.prompt_style}_results.json"
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to {output_file}")
        return output_file


def main():
    """Main entry point for unilateral evaluation."""
    parser = argparse.ArgumentParser(description='Forced-choice unilateral evaluator')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to standard format dataset JSON')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name for evaluation')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--prompt-style', type=str, default='direct',
                       choices=['direct', 'zero-shot', 'few-shot', 'uncertain'],
                       help='Prompt style for unilateral evaluation')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = UnilateralEvaluator(
        model_name=args.model,
        dataset_path=args.dataset,
        prompt_style=args.prompt_style
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(sample_size=args.samples)
    
    # Save results
    evaluator.save_results(output_dir=args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("UNILATERAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {Path(args.dataset).stem}")
    print(f"Prompt Style: {args.prompt_style}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Coverage: {results['coverage']:.3f} (always 1.0 for forced choice)")
    print(f"Macro F1: {results['f1_macro']:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()