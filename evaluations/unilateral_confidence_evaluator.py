#!/usr/bin/env python3
"""
Confidence-Based Unilateral Evaluator

Implements unilateral evaluation with numerical confidence ratings (0.0-1.0)
for comparison with bilateral evaluation results at different threshold levels.

Key features:
- Prompts for numerical confidence rating (0.0-1.0)
- Supports multiple thresholds for CORRECT/INCORRECT classification
- Provides comparison metrics at different confidence levels
"""

import json
import time
import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

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


class UnilateralConfidenceEvaluator:
    """Confidence-based unilateral evaluator with threshold analysis."""
    
    def __init__(self, model_name: str, dataset_path: str,
                 checkpoint_dir: str = "checkpoints",
                 thresholds: List[float] = None):
        """Initialize confidence evaluator.
        
        Args:
            model_name: Name of the model to evaluate
            dataset_path: Path to standard format dataset JSON
            checkpoint_dir: Directory to save checkpoints for recovery
            thresholds: List of confidence thresholds to evaluate (default: [0.5, 0.7, 0.9])
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.thresholds = thresholds or [0.5, 0.7, 0.9]
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.evaluator = ModelRouter.create_evaluator(model_name)
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Results tracking - store raw confidence values
        self.results = {
            'model': model_name,
            'dataset': self.dataset['metadata']['benchmark'],
            'evaluation_type': 'unilateral_confidence',
            'thresholds': self.thresholds,
            'total_samples': 0,
            'confidence_distribution': {
                '0.0-0.1': 0,
                '0.1-0.2': 0,
                '0.2-0.3': 0,
                '0.3-0.4': 0,
                '0.4-0.5': 0,
                '0.5-0.6': 0,
                '0.6-0.7': 0,
                '0.7-0.8': 0,
                '0.8-0.9': 0,
                '0.9-1.0': 0
            },
            'threshold_analysis': {},  # Results at each threshold
            'detailed_results': []
        }
        
        # Initialize threshold analysis
        for threshold in self.thresholds:
            self.results['threshold_analysis'][str(threshold)] = {
                'correct': 0,
                'incorrect': 0,
                'abstained': 0,  # Below threshold
                'accuracy': 0,
                'coverage': 0,
                'category_analysis': {}
            }
        
        # Generate checkpoint ID
        self.checkpoint_id = self._generate_checkpoint_id()
        self.checkpoint_path = self.checkpoint_dir / f"confidence_{self.checkpoint_id}.json"
    
    def _load_dataset(self) -> Dict:
        """Load standard format dataset."""
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)
            
        # Validate dataset format
        if 'metadata' not in dataset or 'assertions' not in dataset:
            raise ValueError(f"Invalid dataset format. Missing 'metadata' or 'assertions'.")
        
        print(f"📊 Loaded dataset: {dataset['metadata']['benchmark']}")
        print(f"   📝 {dataset['metadata']['total_assertions']} assertions")
        print(f"   📅 Generated: {dataset['metadata']['generation_timestamp']}")
        
        return dataset
    
    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID based on evaluation parameters."""
        params = f"{self.model_name}_{self.dataset_path}_confidence"
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
    
    def _create_confidence_prompt(self, assertion: str) -> str:
        """Create confidence-based evaluation prompt."""
        return f"""Rate your confidence that the following statement is correct.

Provide a single number between 0.0 and 1.0:
- 0.0 = Definitely incorrect
- 0.5 = Completely uncertain  
- 1.0 = Definitely correct

Statement: {assertion}

Respond with ONLY this format:
CONFIDENCE: [number]"""
    
    def _evaluate_single_confidence(self, assertion: str) -> float:
        """Evaluate a single assertion and return confidence value."""
        prompt = self._create_confidence_prompt(assertion)
        
        # System prompt for confidence evaluation
        system_prompt = """You are an expert evaluator providing confidence ratings for statement correctness.
Always conclude with 'CONFIDENCE: X.X' where X.X is a number between 0.0 and 1.0."""
        
        # Retry logic for API failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get model response directly from the LLM client
                if hasattr(self.evaluator, 'client'):  # OpenAI/Anthropic evaluators
                    if hasattr(self.evaluator.client, 'chat'):  # OpenAI
                        response = self.evaluator.client.chat.completions.create(
                            model=self.evaluator.model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.1,
                            max_tokens=50
                        ).choices[0].message.content
                        break
                    else:  # Anthropic
                        response = self.evaluator.client.messages.create(
                            model=self.evaluator.model,
                            system=system_prompt,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1,
                            max_tokens=50
                        ).content[0].text
                        break
                elif hasattr(self.evaluator, '_call_api'):  # OpenRouter evaluator
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    response = self.evaluator._call_api(messages, temperature=0.1, max_tokens=50)
                    break
                else:  # MockLLMEvaluator
                    # For mock, return a deterministic confidence based on assertion hash
                    import hashlib
                    hash_val = int(hashlib.md5(assertion.encode()).hexdigest(), 16)
                    # Generate confidence between 0.0 and 1.0
                    response = f"CONFIDENCE: {(hash_val % 100) / 100.0:.2f}"
                    break
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  API error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print(f"❌ Failed after {max_retries} attempts: {str(e)[:100]}")
                    # Default to 0.5 (uncertain) on persistent failure
                    response = "CONFIDENCE: 0.5"
        
        # Parse confidence value from response
        try:
            # Look for "CONFIDENCE:" pattern
            import re
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response, re.IGNORECASE)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                # Clamp to valid range
                confidence = max(0.0, min(1.0, confidence))
            else:
                print(f"⚠️  Could not parse confidence from response: {response[:200]}")
                confidence = 0.5  # Default to uncertain
        except Exception as e:
            print(f"⚠️  Error parsing confidence: {e}")
            confidence = 0.5  # Default to uncertain
        
        return confidence
    
    def evaluate_dataset(self, sample_size: Optional[int] = None) -> Dict:
        """Evaluate dataset with confidence-based evaluation.
        
        Args:
            sample_size: Number of samples to evaluate (None for all)
            
        Returns:
            Evaluation results dictionary with threshold analysis
        """
        # Load checkpoint if exists
        processed_indices = self._load_checkpoint() or set()
        
        # Get items to process
        items = self.dataset['assertions']
        if sample_size:
            items = items[:sample_size]
        
        total_items = len(items)
        print(f"\n🎯 Evaluating {total_items} items with confidence-based evaluation")
        print(f"📊 Model: {self.model_name}")
        print(f"📈 Thresholds: {self.thresholds}")
        
        # Process each item
        for idx, item in enumerate(items):
            if idx in processed_indices:
                continue
            
            # Progress indicator
            if idx % 10 == 0:
                print(f"Evaluating {idx + 1}/{total_items}")
            
            # Get assertion and ground truth
            assertion = item['assertion_text']
            ground_truth = item['expected_label']  # 'correct' or 'incorrect'
            category = item.get('context', {}).get('category', 'uncategorized')
            
            # Get confidence rating
            confidence = self._evaluate_single_confidence(assertion)
            
            # Update confidence distribution
            bucket_idx = min(int(confidence * 10), 9)
            bucket_key = f"{bucket_idx/10:.1f}-{(bucket_idx+1)/10:.1f}"
            self.results['confidence_distribution'][bucket_key] += 1
            
            # Store detailed result
            detailed_result = {
                'index': idx,
                'assertion': assertion,
                'ground_truth': ground_truth,
                'confidence': confidence,
                'category': category
            }
            
            # Evaluate at each threshold
            threshold_predictions = {}
            for threshold in self.thresholds:
                # A threshold defines the minimum confidence needed to make a prediction
                # For threshold 0.7: need confidence >= 0.7 for CORRECT, <= 0.3 for INCORRECT
                # Between 0.3 and 0.7: abstain due to insufficient confidence
                if confidence >= threshold:
                    prediction = "CORRECT"
                elif confidence <= (1.0 - threshold):
                    prediction = "INCORRECT"
                else:
                    # Confidence is in the middle zone - abstain
                    prediction = "ABSTAINED"
                
                threshold_predictions[str(threshold)] = prediction
                
                # Update threshold-specific metrics
                threshold_key = str(threshold)
                threshold_stats = self.results['threshold_analysis'][threshold_key]
                
                if prediction == "ABSTAINED":
                    threshold_stats['abstained'] += 1
                    is_correct = False  # Abstentions count as incorrect for accuracy
                else:
                    is_correct = (
                        (prediction == "CORRECT" and ground_truth == "correct") or
                        (prediction == "INCORRECT" and ground_truth == "incorrect")
                    )
                    if is_correct:
                        threshold_stats['correct'] += 1
                    else:
                        threshold_stats['incorrect'] += 1
                
                # Update category analysis for this threshold
                if category not in threshold_stats['category_analysis']:
                    threshold_stats['category_analysis'][category] = {
                        'total': 0,
                        'correct': 0,
                        'incorrect': 0,
                        'abstained': 0
                    }
                
                cat_stats = threshold_stats['category_analysis'][category]
                cat_stats['total'] += 1
                if prediction == "ABSTAINED":
                    cat_stats['abstained'] += 1
                elif is_correct:
                    cat_stats['correct'] += 1
                else:
                    cat_stats['incorrect'] += 1
            
            detailed_result['threshold_predictions'] = threshold_predictions
            self.results['detailed_results'].append(detailed_result)
            
            # Update total samples
            self.results['total_samples'] += 1
            
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
        """Calculate evaluation metrics for each threshold."""
        total = self.results['total_samples']
        if total == 0:
            return
        
        # Calculate metrics for each threshold
        for threshold_key, stats in self.results['threshold_analysis'].items():
            # Overall metrics
            stats['accuracy'] = stats['correct'] / total if total > 0 else 0
            stats['coverage'] = (total - stats['abstained']) / total if total > 0 else 0
            
            # Accuracy on answered questions only
            answered = total - stats['abstained']
            if answered > 0:
                stats['accuracy_on_answered'] = stats['correct'] / answered
            else:
                stats['accuracy_on_answered'] = 0
            
            # Calculate confusion matrix metrics
            threshold = float(threshold_key)
            tp = sum(1 for r in self.results['detailed_results']
                    if r['threshold_predictions'][threshold_key] == 'CORRECT' 
                    and r['ground_truth'] == 'correct')
            fp = sum(1 for r in self.results['detailed_results']
                    if r['threshold_predictions'][threshold_key] == 'CORRECT'
                    and r['ground_truth'] == 'incorrect')
            tn = sum(1 for r in self.results['detailed_results']
                    if r['threshold_predictions'][threshold_key] == 'INCORRECT'
                    and r['ground_truth'] == 'incorrect')
            fn = sum(1 for r in self.results['detailed_results']
                    if r['threshold_predictions'][threshold_key] == 'INCORRECT'
                    and r['ground_truth'] == 'correct')
            
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
            
            # Store metrics
            stats['f1_macro'] = (f1_true + f1_false) / 2
            stats['precision_macro'] = (precision_true + precision_false) / 2
            stats['recall_macro'] = (recall_true + recall_false) / 2
            
            stats['metrics'] = {
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
            for category, cat_stats in stats['category_analysis'].items():
                if cat_stats['total'] > 0:
                    cat_stats['accuracy'] = cat_stats['correct'] / cat_stats['total']
                    cat_stats['coverage'] = (cat_stats['total'] - cat_stats['abstained']) / cat_stats['total']
        
        # Calculate average confidence
        total_confidence = sum(r['confidence'] for r in self.results['detailed_results'])
        self.results['average_confidence'] = total_confidence / total if total > 0 else 0
    
    def save_results(self, output_dir: str = "results"):
        """Save evaluation results to file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create filename
        dataset_name = Path(self.dataset_path).stem
        model_safe = self.model_name.replace('/', '_').replace(':', '_')
        output_file = output_path / f"{dataset_name}_{model_safe}_unilateral_confidence_results.json"
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to {output_file}")
        return output_file


def main():
    """Main entry point for confidence-based evaluation."""
    parser = argparse.ArgumentParser(description='Confidence-based unilateral evaluator')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to standard format dataset JSON')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name for evaluation')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--thresholds', type=float, nargs='+',
                       default=[0.5, 0.7, 0.9],
                       help='Confidence thresholds for classification (default: 0.5 0.7 0.9)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = UnilateralConfidenceEvaluator(
        model_name=args.model,
        dataset_path=args.dataset,
        thresholds=args.thresholds
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(sample_size=args.samples)
    
    # Save results
    evaluator.save_results(output_dir=args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("CONFIDENCE-BASED UNILATERAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {Path(args.dataset).stem}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Average Confidence: {results['average_confidence']:.3f}")
    print("\nResults by Threshold:")
    print("-"*60)
    
    for threshold in args.thresholds:
        threshold_key = str(threshold)
        stats = results['threshold_analysis'][threshold_key]
        print(f"\nThreshold: {threshold}")
        print(f"  Accuracy: {stats['accuracy']:.3f}")
        print(f"  Coverage: {stats['coverage']:.3f}")
        print(f"  Accuracy on Answered: {stats['accuracy_on_answered']:.3f}")
        print(f"  Macro F1: {stats['f1_macro']:.3f}")
        print(f"  Abstained: {stats['abstained']}/{results['total_samples']}")
    
    print("\nConfidence Distribution:")
    print("-"*60)
    for bucket, count in sorted(results['confidence_distribution'].items()):
        percentage = (count / results['total_samples'] * 100) if results['total_samples'] > 0 else 0
        bar = '█' * int(percentage / 2)
        print(f"{bucket}: {bar} {count} ({percentage:.1f}%)")
    
    print("="*60)


if __name__ == "__main__":
    main()