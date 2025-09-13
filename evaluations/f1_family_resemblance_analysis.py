#!/usr/bin/env python3
"""
Test the family resemblance hypothesis for F1 macro performance patterns:
Do models from the same family (OpenAI, Anthropic, Llama, Google) exhibit
similar strengths and weaknesses in F1 macro performance across categories?
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine, euclidean, correlation
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Model families mapping
MODEL_FAMILIES = {
    'gpt-5-2025-08-07': 'OpenAI',
    'gpt-4.1-2025-04-14': 'OpenAI',
    'gpt-4.1-mini-2025-04-14': 'OpenAI',
    'claude-opus-4-1-20250805': 'Anthropic',
    'claude-sonnet-4-20250514': 'Anthropic',
    'claude-3-5-haiku-20241022': 'Anthropic',
    'meta-llama-llama-4-scout': 'Meta/Llama',
    'meta-llama-llama-4-maverick': 'Meta/Llama',
    'google-gemini-2.5-pro': 'Google',
    'google-gemini-2.5-flash': 'Google'
}

def calculate_f1_for_category(results: List[Dict], category: str) -> float:
    """Calculate F1 macro score for a specific category from detailed results."""
    
    # Filter results for this category
    category_results = [r for r in results if r.get('category') == category]
    
    if not category_results:
        return np.nan
    
    # Separate by expected label and projected value
    correct_true = sum(1 for r in category_results 
                       if r.get('expected_label') == 'correct' and r.get('projected_value') == 't')
    correct_false = sum(1 for r in category_results 
                        if r.get('expected_label') == 'correct' and r.get('projected_value') == 'f')
    incorrect_true = sum(1 for r in category_results 
                         if r.get('expected_label') == 'incorrect' and r.get('projected_value') == 't')
    incorrect_false = sum(1 for r in category_results 
                          if r.get('expected_label') == 'incorrect' and r.get('projected_value') == 'f')
    
    # Calculate precision and recall for each class
    # Class "correct" (expected_label == 'correct')
    correct_precision = correct_true / (correct_true + incorrect_true) if (correct_true + incorrect_true) > 0 else 0
    correct_recall = correct_true / (correct_true + correct_false) if (correct_true + correct_false) > 0 else 0
    correct_f1 = 2 * correct_precision * correct_recall / (correct_precision + correct_recall) if (correct_precision + correct_recall) > 0 else 0
    
    # Class "incorrect" (expected_label == 'incorrect')
    incorrect_precision = incorrect_false / (incorrect_false + correct_false) if (incorrect_false + correct_false) > 0 else 0
    incorrect_recall = incorrect_false / (incorrect_false + incorrect_true) if (incorrect_false + incorrect_true) > 0 else 0
    incorrect_f1 = 2 * incorrect_precision * incorrect_recall / (incorrect_precision + incorrect_recall) if (incorrect_precision + incorrect_recall) > 0 else 0
    
    # Macro F1
    return (correct_f1 + incorrect_f1) / 2

def load_category_performance(results_dir: str = "results") -> Dict:
    """Load F1 macro performance by category for each model and benchmark."""
    
    results_path = Path(results_dir)
    performance_data = defaultdict(lambda: defaultdict(dict))
    
    for file_path in results_path.glob("*.json"):
        if "summary" in file_path.name:
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Parse filename to get model and benchmark
            parts = file_path.stem.split('_')
            
            # Identify benchmark
            benchmark = None
            if parts[0] == 'truthfulqa' and parts[1] == 'complete':
                benchmark = 'TruthfulQA'
                model_parts = parts[2:-2]
            elif parts[0] == 'simpleqa' and parts[1] == 'complete':
                benchmark = 'SimpleQA'
                model_parts = parts[2:-2]
            elif parts[0] == 'mmlupro' and parts[1] == 'complete':
                benchmark = 'MMLU-Pro'
                model_parts = parts[2:-2]
            elif parts[0] == 'factscore' and parts[1] == 'complete':
                benchmark = 'FACTScore'
                model_parts = parts[2:-2]
            else:
                continue
                
            model = '-'.join(model_parts)
            
            # Extract category performance
            if 'category_analysis' in data and data['category_analysis']:
                # Use pre-calculated category analysis if available
                for category, stats in data['category_analysis'].items():
                    if isinstance(stats, dict):
                        # Try to get F1 if available, otherwise calculate from accuracy
                        f1 = stats.get('f1_macro', stats.get('accuracy', 0))
                        performance_data[model][benchmark][category] = f1
            else:
                # Calculate F1 from detailed results
                detailed_results = data.get('detailed_results', [])
                
                # Get all categories
                categories = set(r.get('category') for r in detailed_results if r.get('category'))
                
                for category in categories:
                    f1 = calculate_f1_for_category(detailed_results, category)
                    if not np.isnan(f1):
                        performance_data[model][benchmark][category] = f1
            
            # Also store overall F1
            performance_data[model][benchmark]['_overall'] = data.get('f1_macro', 0)
            
        except Exception as e:
            print(f"Warning: Could not load {file_path.name}: {e}")
    
    return performance_data

def create_performance_profiles(performance_data: Dict) -> Tuple[np.ndarray, List[str], List[str]]:
    """Create performance profile vectors for each model."""
    
    models = sorted(performance_data.keys())
    
    # Collect all unique benchmark-category pairs
    all_features = set()
    for model_data in performance_data.values():
        for benchmark, categories in model_data.items():
            for category in categories:
                if category != '_overall':  # Skip overall scores for now
                    all_features.add(f"{benchmark}:{category}")
    
    features = sorted(all_features)
    
    # Create performance matrix
    performance_matrix = np.zeros((len(models), len(features)))
    
    for i, model in enumerate(models):
        for j, feature in enumerate(features):
            benchmark, category = feature.split(':', 1)
            if benchmark in performance_data[model] and category in performance_data[model][benchmark]:
                performance_matrix[i, j] = performance_data[model][benchmark][category]
            else:
                performance_matrix[i, j] = np.nan
    
    # Handle missing values - use mean imputation
    for j in range(performance_matrix.shape[1]):
        col = performance_matrix[:, j]
        if np.all(np.isnan(col)):
            performance_matrix[:, j] = 0
        else:
            mean_val = np.nanmean(col)
            performance_matrix[np.isnan(performance_matrix[:, j]), j] = mean_val
    
    return performance_matrix, models, features

def calculate_similarity_distances(performance_matrix: np.ndarray) -> np.ndarray:
    """Calculate pairwise similarity distances between model performance profiles."""
    
    n_models = performance_matrix.shape[0]
    distances = np.zeros((n_models, n_models))
    
    # Standardize features for fair comparison
    scaler = StandardScaler()
    normalized_matrix = scaler.fit_transform(performance_matrix)
    
    for i in range(n_models):
        for j in range(n_models):
            # Use correlation distance (1 - correlation)
            # This captures similarity in performance patterns regardless of absolute values
            if i != j:
                corr, _ = pearsonr(normalized_matrix[i], normalized_matrix[j])
                distances[i, j] = 1 - corr
            else:
                distances[i, j] = 0
    
    return distances

def test_family_resemblance_hypothesis(distances: np.ndarray, models: List[str]) -> Dict:
    """Test if models from the same family have more similar performance patterns."""
    
    # Map models to families
    model_to_family = {}
    for model in models:
        family_found = False
        for pattern, family in MODEL_FAMILIES.items():
            if pattern in model:
                model_to_family[model] = family
                family_found = True
                break
        
        if not family_found:
            # Try to infer from model name
            if 'gpt' in model.lower():
                model_to_family[model] = 'OpenAI'
            elif 'claude' in model.lower():
                model_to_family[model] = 'Anthropic'
            elif 'llama' in model.lower():
                model_to_family[model] = 'Meta/Llama'
            elif 'gemini' in model.lower():
                model_to_family[model] = 'Google'
            else:
                model_to_family[model] = 'Unknown'
    
    # Collect within-family and between-family distances
    within_family = []
    between_family = []
    family_specific = defaultdict(lambda: defaultdict(list))
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:  # Upper triangle only
                dist = distances[i, j]
                family1 = model_to_family.get(model1, 'Unknown')
                family2 = model_to_family.get(model2, 'Unknown')
                
                if family1 == family2 and family1 != 'Unknown':
                    within_family.append(dist)
                    family_specific[family1]['within'].append(dist)
                elif family1 != 'Unknown' and family2 != 'Unknown':
                    between_family.append(dist)
                    pair_key = tuple(sorted([family1, family2]))
                    family_specific[pair_key]['between'].append(dist)
    
    results = {
        'within_family_distances': within_family,
        'between_family_distances': between_family,
        'within_mean': np.mean(within_family) if within_family else np.nan,
        'within_std': np.std(within_family) if within_family else np.nan,
        'between_mean': np.mean(between_family) if between_family else np.nan,
        'between_std': np.std(between_family) if between_family else np.nan,
        'family_specific': dict(family_specific),
        'model_families': model_to_family
    }
    
    # Statistical test
    if within_family and between_family:
        statistic, p_value = mannwhitneyu(within_family, between_family, alternative='less')
        results['mann_whitney_statistic'] = statistic
        results['mann_whitney_p_value'] = p_value
        results['hypothesis_supported'] = p_value < 0.05
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(within_family) + np.var(between_family)) / 2)
        if pooled_std > 0:
            results['cohens_d'] = (np.mean(between_family) - np.mean(within_family)) / pooled_std
        else:
            results['cohens_d'] = 0
    
    return results

def identify_family_strengths(performance_data: Dict, models: List[str], 
                             model_families: Dict) -> Dict:
    """Identify which categories each family excels or struggles with."""
    
    family_strengths = defaultdict(lambda: defaultdict(list))
    
    # Group models by family
    families = defaultdict(list)
    for model in models:
        family = model_families.get(model, 'Unknown')
        if family != 'Unknown':
            families[family].append(model)
    
    # Analyze each benchmark-category combination
    all_categories = set()
    for model_data in performance_data.values():
        for benchmark, cats in model_data.items():
            for cat in cats:
                if cat != '_overall':
                    all_categories.add((benchmark, cat))
    
    for benchmark, category in all_categories:
        # Get performance for each family
        family_performances = {}
        
        for family, family_models in families.items():
            perfs = []
            for model in family_models:
                if benchmark in performance_data[model] and category in performance_data[model][benchmark]:
                    perfs.append(performance_data[model][benchmark][category])
            
            if perfs:
                family_performances[family] = np.mean(perfs)
        
        if len(family_performances) > 1:
            # Find best and worst performing families
            sorted_families = sorted(family_performances.items(), key=lambda x: x[1], reverse=True)
            
            # Record strengths (top performer)
            best_family = sorted_families[0][0]
            best_score = sorted_families[0][1]
            
            # Record weaknesses (bottom performer)
            worst_family = sorted_families[-1][0]
            worst_score = sorted_families[-1][1]
            
            # Calculate relative advantage
            if worst_score > 0:
                advantage = (best_score - worst_score) / worst_score * 100
            else:
                advantage = 100
            
            family_strengths[best_family]['strengths'].append({
                'benchmark': benchmark,
                'category': category,
                'score': best_score,
                'advantage': advantage
            })
            
            family_strengths[worst_family]['weaknesses'].append({
                'benchmark': benchmark,
                'category': category,
                'score': worst_score,
                'disadvantage': -advantage
            })
    
    return dict(family_strengths)

def visualize_results(performance_matrix: np.ndarray, distances: np.ndarray, 
                      models: List[str], features: List[str], results: Dict,
                      performance_data: Dict, family_strengths: Dict):
    """Create comprehensive visualizations of the analysis."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Performance heatmap
    ax1 = plt.subplot(3, 3, 1)
    
    # Sample features for visualization (too many to show all)
    n_features_to_show = min(30, len(features))
    feature_indices = np.linspace(0, len(features)-1, n_features_to_show, dtype=int)
    sampled_features = [features[i] for i in feature_indices]
    sampled_matrix = performance_matrix[:, feature_indices]
    
    sns.heatmap(sampled_matrix, xticklabels=[f.split(':')[-1][:15] for f in sampled_features],
                yticklabels=[m.split('-')[0][:12] for m in models],
                cmap='RdYlGn', center=0.5, vmin=0, vmax=1,
                cbar_kws={'label': 'F1 Score'})
    ax1.set_title('F1 Performance Matrix (Sample Categories)')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. PCA projection of performance profiles
    ax2 = plt.subplot(3, 3, 2)
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    normalized = scaler.fit_transform(performance_matrix)
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(normalized)
    
    # Color by family
    families = [results['model_families'].get(m, 'Unknown') for m in models]
    unique_families = list(set(families))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_families)))
    family_colors = {fam: colors[i] for i, fam in enumerate(unique_families)}
    
    for model, coord, family in zip(models, pca_coords, families):
        ax2.scatter(coord[0], coord[1], c=[family_colors[family]], s=100, alpha=0.7)
        ax2.annotate(model.split('-')[0][:10], (coord[0], coord[1]), fontsize=8)
    
    ax2.set_title(f'Performance Profile Clustering (PCA)\nExplained Var: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    
    # Add legend
    for family, color in family_colors.items():
        ax2.scatter([], [], c=[color], label=family, s=100)
    ax2.legend(loc='best', title='Family')
    
    # 3. Within vs Between family distances
    ax3 = plt.subplot(3, 3, 3)
    
    data_to_plot = [results['within_family_distances'], results['between_family_distances']]
    bp = ax3.boxplot(data_to_plot, labels=['Within Family', 'Between Family'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax3.set_ylabel('Performance Pattern Distance')
    ax3.set_title('F1 Pattern Similarity: Within vs Between Families')
    
    # Add statistical annotation
    if 'mann_whitney_p_value' in results:
        p_val = results['mann_whitney_p_value']
        effect = results.get('cohens_d', 0)
        ax3.text(0.5, 0.95, f'p-value: {p_val:.4f}', 
                transform=ax3.transAxes, ha='center', va='top')
        ax3.text(0.5, 0.90, f"Cohen's d: {effect:.2f}", 
                transform=ax3.transAxes, ha='center', va='top')
        
        if results['hypothesis_supported']:
            ax3.text(0.5, 0.85, '✓ Family resemblance supported', 
                    transform=ax3.transAxes, ha='center', va='top', color='green', fontweight='bold')
        else:
            ax3.text(0.5, 0.85, '✗ No significant family resemblance', 
                    transform=ax3.transAxes, ha='center', va='top', color='red')
    
    # 4. Distance matrix heatmap
    ax4 = plt.subplot(3, 3, 4)
    sns.heatmap(distances, xticklabels=[m.split('-')[0][:10] for m in models],
                yticklabels=[m.split('-')[0][:10] for m in models],
                cmap='coolwarm_r', vmin=0, vmax=2,
                cbar_kws={'label': 'Pattern Distance'})
    ax4.set_title('Pairwise Performance Pattern Distances')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Family-specific performance profiles
    ax5 = plt.subplot(3, 3, 5)
    
    # Average performance by family across benchmarks
    family_avg_performance = defaultdict(list)
    
    for model in models:
        family = results['model_families'].get(model, 'Unknown')
        if family != 'Unknown':
            model_perfs = []
            for benchmark_data in performance_data[model].values():
                if '_overall' in benchmark_data:
                    model_perfs.append(benchmark_data['_overall'])
            if model_perfs:
                family_avg_performance[family].append(np.mean(model_perfs))
    
    families_to_plot = sorted(family_avg_performance.keys())
    positions = range(len(families_to_plot))
    
    bp = ax5.boxplot([family_avg_performance[f] for f in families_to_plot],
                     labels=families_to_plot, positions=positions, patch_artist=True)
    
    for patch, family in zip(bp['boxes'], families_to_plot):
        patch.set_facecolor(family_colors[family])
    
    ax5.set_ylabel('F1 Macro Score')
    ax5.set_title('Overall F1 Performance by Family')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. t-SNE visualization
    ax6 = plt.subplot(3, 3, 6)
    
    # Apply t-SNE to performance profiles
    tsne = TSNE(n_components=2, perplexity=min(5, len(models)-1), random_state=42)
    tsne_coords = tsne.fit_transform(performance_matrix)
    
    for model, coord, family in zip(models, tsne_coords, families):
        ax6.scatter(coord[0], coord[1], c=[family_colors[family]], s=100, alpha=0.7)
        ax6.annotate(model.split('-')[0][:10], (coord[0], coord[1]), fontsize=8)
    
    ax6.set_title('Performance Profile Clustering (t-SNE)')
    ax6.set_xlabel('t-SNE 1')
    ax6.set_ylabel('t-SNE 2')
    
    # 7-9. Family strengths and weaknesses
    for idx, (family, data) in enumerate(list(family_strengths.items())[:3]):
        ax = plt.subplot(3, 3, 7 + idx)
        
        strengths = data.get('strengths', [])[:5]  # Top 5 strengths
        weaknesses = data.get('weaknesses', [])[:5]  # Top 5 weaknesses
        
        if strengths or weaknesses:
            categories = []
            scores = []
            colors_list = []
            
            for s in strengths:
                categories.append(f"{s['benchmark'][:4]}:{s['category'][:15]}")
                scores.append(s['score'])
                colors_list.append('green')
            
            for w in weaknesses:
                categories.append(f"{w['benchmark'][:4]}:{w['category'][:15]}")
                scores.append(w['score'])
                colors_list.append('red')
            
            y_pos = np.arange(len(categories))
            ax.barh(y_pos, scores, color=colors_list, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories, fontsize=8)
            ax.set_xlabel('F1 Score')
            ax.set_title(f'{family} Family:\nStrengths (green) & Weaknesses (red)')
            ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig('f1_family_resemblance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_detailed_results(results: Dict, family_strengths: Dict):
    """Print detailed analysis results."""
    
    print("\n" + "="*80)
    print("F1 PERFORMANCE FAMILY RESEMBLANCE ANALYSIS")
    print("="*80)
    
    print(f"\nWithin-family distance: {results['within_mean']:.4f} ± {results['within_std']:.4f}")
    print(f"Between-family distance: {results['between_mean']:.4f} ± {results['between_std']:.4f}")
    
    if not np.isnan(results['within_mean']) and not np.isnan(results['between_mean']):
        if results['within_mean'] < results['between_mean']:
            reduction = (1 - results['within_mean'] / results['between_mean']) * 100
            print(f"\n→ Within-family distances are {reduction:.1f}% smaller")
            print("  (Models from same family have more similar performance patterns)")
        else:
            increase = (results['within_mean'] / results['between_mean'] - 1) * 100
            print(f"\n→ Within-family distances are {increase:.1f}% larger")
            print("  (Models from same family have less similar performance patterns)")
    
    if 'mann_whitney_p_value' in results:
        print(f"\nStatistical Test Results:")
        print(f"  Mann-Whitney U p-value: {results['mann_whitney_p_value']:.6f}")
        print(f"  Cohen's d effect size: {results.get('cohens_d', 0):.3f}")
        
        if results['hypothesis_supported']:
            print("\n✓ HYPOTHESIS SUPPORTED:")
            print("  Models from the same family exhibit significantly more similar")
            print("  F1 performance patterns across categories (p < 0.05)")
        else:
            print("\n✗ HYPOTHESIS NOT SUPPORTED:")
            print("  No significant difference in performance pattern similarity")
            print("  between within-family and between-family model pairs")
    
    # Print family-specific patterns
    print("\n" + "-"*80)
    print("FAMILY-SPECIFIC PERFORMANCE CHARACTERISTICS")
    print("-"*80)
    
    for family, data in family_strengths.items():
        if family == 'Unknown':
            continue
            
        print(f"\n{family} Family:")
        
        strengths = sorted(data.get('strengths', []), key=lambda x: x['advantage'], reverse=True)[:5]
        if strengths:
            print("  Top Strengths:")
            for s in strengths:
                print(f"    • {s['benchmark']}: {s['category'][:30]:30} (F1: {s['score']:.3f}, +{s['advantage']:.1f}% advantage)")
        
        weaknesses = sorted(data.get('weaknesses', []), key=lambda x: x['disadvantage'])[:5]
        if weaknesses:
            print("  Top Weaknesses:")
            for w in weaknesses:
                print(f"    • {w['benchmark']}: {w['category'][:30]:30} (F1: {w['score']:.3f}, {w['disadvantage']:.1f}% disadvantage)")
    
    # Print pairwise family comparisons
    print("\n" + "-"*80)
    print("BETWEEN-FAMILY DISTANCE STATISTICS")
    print("-"*80)
    
    family_pairs = defaultdict(list)
    for key, data in results.get('family_specific', {}).items():
        if isinstance(key, tuple) and 'between' in data:
            family_pairs[key] = data['between']
    
    for (fam1, fam2), distances in sorted(family_pairs.items()):
        if distances:
            print(f"\n{fam1} <-> {fam2}:")
            print(f"  Mean distance: {np.mean(distances):.4f}")
            print(f"  Std deviation: {np.std(distances):.4f}")

def main():
    """Main analysis function."""
    print("Loading F1 performance data from evaluation results...")
    performance_data = load_category_performance()
    
    if not performance_data:
        print("No results found!")
        return
    
    print(f"Loaded performance data for {len(performance_data)} models")
    
    # Create performance profiles
    print("\nCreating performance profile vectors...")
    performance_matrix, models, features = create_performance_profiles(performance_data)
    print(f"Performance matrix shape: {performance_matrix.shape}")
    print(f"Features (benchmark:category pairs): {len(features)}")
    
    # Calculate similarity distances
    print("\nCalculating performance pattern distances...")
    distances = calculate_similarity_distances(performance_matrix)
    
    # Test hypothesis
    print("\nTesting family resemblance hypothesis...")
    results = test_family_resemblance_hypothesis(distances, models)
    
    # Identify family strengths and weaknesses
    print("\nIdentifying family-specific strengths and weaknesses...")
    family_strengths = identify_family_strengths(performance_data, models, 
                                                 results['model_families'])
    
    # Print results
    print_detailed_results(results, family_strengths)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_results(performance_matrix, distances, models, features, 
                     results, performance_data, family_strengths)
    
    print("\nAnalysis complete! Visualizations saved to 'f1_family_resemblance_analysis.png'")
    
    # Save results to JSON for further analysis
    output_data = {
        'hypothesis_supported': bool(results.get('hypothesis_supported', False)),  # Convert numpy bool to Python bool
        'within_family_mean': float(results.get('within_mean', 0)),  # Convert to Python float
        'between_family_mean': float(results.get('between_mean', 0)),
        'p_value': float(results.get('mann_whitney_p_value', 1.0)),
        'cohens_d': float(results.get('cohens_d', 0)),
        'model_families': results.get('model_families', {}),
        'family_strengths': {
            k: {
                'strengths': [
                    {
                        'benchmark': s['benchmark'],
                        'category': s['category'],
                        'score': float(s['score']),  # Convert numpy floats
                        'advantage': float(s['advantage'])
                    }
                    for s in v.get('strengths', [])
                ],
                'weaknesses': [
                    {
                        'benchmark': w['benchmark'],
                        'category': w['category'],
                        'score': float(w['score']),
                        'disadvantage': float(w['disadvantage'])
                    }
                    for w in v.get('weaknesses', [])
                ]
            }
            for k, v in family_strengths.items()
        }
    }
    
    with open('f1_family_resemblance_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Results saved to 'f1_family_resemblance_results.json'")

if __name__ == "__main__":
    main()