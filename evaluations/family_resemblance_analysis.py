#!/usr/bin/env python3
"""
Test the family resemblance hypothesis: Do models from the same family 
(OpenAI, Anthropic, Llama, Google) have more similar bilateral truth value 
distributions than models from different families?
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
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

def load_results(results_dir: str = "results") -> Dict:
    """Load all results and extract bilateral truth value distributions."""
    results_path = Path(results_dir)
    
    # Store distributions by model and benchmark
    distributions = defaultdict(lambda: defaultdict(dict))
    
    for file_path in results_path.glob("*.json"):
        if "summary" in file_path.name:
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Parse filename to get model and benchmark
            parts = file_path.stem.split('_')
            
            # Find benchmark name
            benchmark = None
            if parts[0] == 'truthfulqa' and parts[1] == 'complete':
                benchmark = 'TruthfulQA'
                model_parts = parts[2:-2]  # Remove benchmark and policy parts
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
            
            # Count bilateral truth values
            tv_counts = defaultdict(int)
            total = 0
            
            for item in data.get('detailed_results', []):
                bilateral_value = item.get('bilateral_value', '<e,e>')
                tv_counts[bilateral_value] += 1
                total += 1
            
            # Convert to probability distribution
            if total > 0:
                tv_labels = ['<t,f>', '<f,t>', '<t,t>', '<f,f>', '<t,e>', '<e,t>', '<f,e>', '<e,f>', '<e,e>']
                distribution = [tv_counts[tv] / total for tv in tv_labels]
                
                distributions[model][benchmark] = {
                    'distribution': distribution,
                    'counts': dict(tv_counts),
                    'total': total,
                    'accuracy': data.get('accuracy', 0),
                    'coverage': data.get('coverage', 0)
                }
                
        except Exception as e:
            print(f"Warning: Could not load {file_path.name}: {e}")
    
    return distributions

def calculate_distances(distributions: Dict) -> Tuple[np.ndarray, List[str], Dict]:
    """Calculate pairwise distances between all models across all benchmarks."""
    
    models = sorted(distributions.keys())
    n_models = len(models)
    
    # Store distances for each benchmark
    benchmark_distances = {}
    
    # Get all benchmarks
    benchmarks = set()
    for model_data in distributions.values():
        benchmarks.update(model_data.keys())
    benchmarks = sorted(benchmarks)
    
    # Calculate distances for each benchmark
    for benchmark in benchmarks:
        distance_matrix = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if model1 in distributions and model2 in distributions:
                    if benchmark in distributions[model1] and benchmark in distributions[model2]:
                        dist1 = distributions[model1][benchmark]['distribution']
                        dist2 = distributions[model2][benchmark]['distribution']
                        
                        # Use Jensen-Shannon divergence (symmetric KL divergence)
                        dist1_array = np.array(dist1) + 1e-10  # Add small value to avoid log(0)
                        dist2_array = np.array(dist2) + 1e-10
                        
                        # Jensen-Shannon divergence
                        m = 0.5 * (dist1_array + dist2_array)
                        js_div = 0.5 * np.sum(dist1_array * np.log(dist1_array / m)) + \
                                 0.5 * np.sum(dist2_array * np.log(dist2_array / m))
                        
                        distance_matrix[i, j] = np.sqrt(js_div)  # JS distance
                    else:
                        distance_matrix[i, j] = np.nan
        
        benchmark_distances[benchmark] = distance_matrix
    
    # Calculate average distance across benchmarks
    avg_distance = np.nanmean(list(benchmark_distances.values()), axis=0)
    
    return avg_distance, models, benchmark_distances

def test_family_resemblance(distances: np.ndarray, models: List[str]) -> Dict:
    """Test if within-family distances are smaller than between-family distances."""
    
    # Map models to families
    model_to_family = {}
    for model in models:
        for pattern, family in MODEL_FAMILIES.items():
            if pattern in model:
                model_to_family[model] = family
                break
        if model not in model_to_family:
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
    family_pairs = defaultdict(list)
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:  # Only consider upper triangle
                dist = distances[i, j]
                if not np.isnan(dist):
                    family1 = model_to_family.get(model1, 'Unknown')
                    family2 = model_to_family.get(model2, 'Unknown')
                    
                    if family1 == family2 and family1 != 'Unknown':
                        within_family.append(dist)
                        family_pairs[family1].append(dist)
                    elif family1 != 'Unknown' and family2 != 'Unknown':
                        between_family.append(dist)
                        family_pairs[f"{family1}-{family2}"].append(dist)
    
    # Statistical tests
    results = {
        'within_family_mean': np.mean(within_family) if within_family else np.nan,
        'within_family_std': np.std(within_family) if within_family else np.nan,
        'between_family_mean': np.mean(between_family) if between_family else np.nan,
        'between_family_std': np.std(between_family) if between_family else np.nan,
        'within_family_distances': within_family,
        'between_family_distances': between_family,
        'family_pairs': dict(family_pairs),
        'model_families': model_to_family
    }
    
    # Perform statistical test if we have data
    if within_family and between_family:
        from scipy.stats import mannwhitneyu
        statistic, p_value = mannwhitneyu(within_family, between_family, alternative='less')
        results['mann_whitney_statistic'] = statistic
        results['mann_whitney_p_value'] = p_value
        results['hypothesis_supported'] = p_value < 0.05
    
    return results

def visualize_results(distances: np.ndarray, models: List[str], results: Dict, 
                      benchmark_distances: Dict, distributions: Dict):
    """Create visualizations of the family resemblance analysis."""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Distance heatmap
    ax1 = plt.subplot(3, 3, 1)
    sns.heatmap(distances, xticklabels=models, yticklabels=models, 
                cmap='coolwarm_r', annot=False, fmt='.2f', cbar_kws={'label': 'JS Distance'})
    ax1.set_title('Pairwise Model Distances (Average Across Benchmarks)')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)
    
    # 2. MDS visualization
    ax2 = plt.subplot(3, 3, 2)
    # Use MDS to project distances to 2D
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(np.nan_to_num(distances, nan=np.nanmax(distances)))
    
    # Color by family
    families = [results['model_families'].get(m, 'Unknown') for m in models]
    unique_families = list(set(families))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_families)))
    family_colors = {fam: colors[i] for i, fam in enumerate(unique_families)}
    
    for model, coord, family in zip(models, coords, families):
        ax2.scatter(coord[0], coord[1], c=[family_colors[family]], s=100, alpha=0.7)
        ax2.annotate(model.split('-')[0][:10], (coord[0], coord[1]), fontsize=8)
    
    ax2.set_title('Model Similarity Map (MDS Projection)')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    
    # Add legend for families
    for family, color in family_colors.items():
        ax2.scatter([], [], c=[color], label=family, s=100)
    ax2.legend(loc='best', title='Model Family')
    
    # 3. Within vs Between family distances
    ax3 = plt.subplot(3, 3, 3)
    data_to_plot = [results['within_family_distances'], results['between_family_distances']]
    labels = ['Within Family', 'Between Family']
    
    bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax3.set_ylabel('JS Distance')
    ax3.set_title('Within vs Between Family Distances')
    
    # Add statistical annotation
    if 'mann_whitney_p_value' in results:
        p_val = results['mann_whitney_p_value']
        ax3.text(0.5, 0.95, f'p-value: {p_val:.4f}', 
                transform=ax3.transAxes, ha='center', va='top')
        if results['hypothesis_supported']:
            ax3.text(0.5, 0.90, '✓ Family resemblance supported', 
                    transform=ax3.transAxes, ha='center', va='top', color='green')
        else:
            ax3.text(0.5, 0.90, '✗ No significant family resemblance', 
                    transform=ax3.transAxes, ha='center', va='top', color='red')
    
    # 4-7. Distance heatmaps for each benchmark
    for idx, (benchmark, dist_matrix) in enumerate(benchmark_distances.items()):
        ax = plt.subplot(3, 3, 4 + idx)
        sns.heatmap(dist_matrix, xticklabels=[m.split('-')[0][:8] for m in models], 
                   yticklabels=[m.split('-')[0][:8] for m in models],
                   cmap='coolwarm_r', annot=False, cbar_kws={'label': 'JS Dist'})
        ax.set_title(f'{benchmark} Distances')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
    
    # 8. Family-specific distance distributions
    ax8 = plt.subplot(3, 3, 8)
    family_data = defaultdict(list)
    for pair, dists in results['family_pairs'].items():
        if '-' not in pair:  # Within-family
            family_data[pair] = dists
    
    if family_data:
        positions = range(1, len(family_data) + 1)
        bp = ax8.boxplot(family_data.values(), labels=family_data.keys(), 
                         positions=positions, patch_artist=True)
        for patch, pos in zip(bp['boxes'], positions):
            patch.set_facecolor(plt.cm.Set3(pos / len(family_data)))
        ax8.set_ylabel('JS Distance')
        ax8.set_title('Within-Family Distance Distributions')
        plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 9. Truth value distribution comparison for selected models
    ax9 = plt.subplot(3, 3, 9)
    
    # Select one model from each family for comparison
    family_representatives = {}
    for model in models:
        family = results['model_families'].get(model, 'Unknown')
        if family != 'Unknown' and family not in family_representatives:
            family_representatives[family] = model
    
    tv_labels = ['<t,f>', '<f,t>', '<t,t>', '<f,f>', '<t,e>', '<e,t>', '<f,e>', '<e,f>', '<e,e>']
    x = np.arange(len(tv_labels))
    width = 0.8 / len(family_representatives)
    
    for i, (family, model) in enumerate(family_representatives.items()):
        if model in distributions:
            # Average across benchmarks
            avg_dist = np.zeros(9)
            count = 0
            for benchmark_data in distributions[model].values():
                avg_dist += np.array(benchmark_data['distribution'])
                count += 1
            if count > 0:
                avg_dist /= count
                ax9.bar(x + i * width, avg_dist, width, label=f'{family}', 
                       color=family_colors[family], alpha=0.7)
    
    ax9.set_xlabel('Truth Value')
    ax9.set_ylabel('Probability')
    ax9.set_title('Average Truth Value Distributions by Family')
    ax9.set_xticks(x + width * (len(family_representatives) - 1) / 2)
    ax9.set_xticklabels(tv_labels, rotation=45, ha='right')
    ax9.legend()
    
    plt.tight_layout()
    plt.savefig('family_resemblance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_detailed_results(results: Dict):
    """Print detailed analysis results."""
    
    print("\n" + "="*80)
    print("FAMILY RESEMBLANCE HYPOTHESIS TEST RESULTS")
    print("="*80)
    
    print("\nBilateral Truth Value Semantics:")
    print("-" * 40)
    print("Core truth values (no evaluation failures):")
    print("  <t,f>: Classical true (verifiable, not refutable)")
    print("  <f,t>: Classical false (not verifiable, refutable)")
    print("  <t,t>: Contradiction (both verifiable and refutable)")
    print("  <f,f>: Knowledge gap/uncertainty (neither verifiable nor refutable)")
    print("\nValues with evaluation failures (e in one or both components):")
    print("  <t,e>: Verifiable but refutability evaluation failed")
    print("  <e,t>: Verifiability evaluation failed but refutable")
    print("  <f,e>: Not verifiable but refutability evaluation failed")
    print("  <e,f>: Verifiability evaluation failed but not refutable")
    print("  <e,e>: Complete evaluation failure (both dimensions failed)")
    
    print(f"\nWithin-family distance: {results['within_family_mean']:.4f} ± {results['within_family_std']:.4f}")
    print(f"Between-family distance: {results['between_family_mean']:.4f} ± {results['between_family_std']:.4f}")
    
    if results['within_family_mean'] < results['between_family_mean']:
        reduction = (1 - results['within_family_mean'] / results['between_family_mean']) * 100
        print(f"Within-family distances are {reduction:.1f}% smaller than between-family")
    
    if 'mann_whitney_p_value' in results:
        print(f"\nMann-Whitney U test p-value: {results['mann_whitney_p_value']:.6f}")
        if results['hypothesis_supported']:
            print("✓ HYPOTHESIS SUPPORTED: Models from the same family have significantly")
            print("  more similar bilateral truth value distributions (p < 0.05)")
            print("\nInterpretation: Family resemblance indicates shared:")
            print("  • Knowledge boundaries (similar <f,f> patterns for uncertainty)")
            print("  • Evaluation failure patterns (similar occurrences of 'e' components)")
            print("  • Contradiction detection (similar <t,t> patterns)")
            print("  • Processing capabilities (which assertions cause evaluation failures)")
        else:
            print("✗ HYPOTHESIS NOT SUPPORTED: No significant difference between")
            print("  within-family and between-family distances (p >= 0.05)")
            print("\nInterpretation: Individual model characteristics dominate over family traits")
    
    # Print family-specific statistics
    print("\n" + "-"*80)
    print("FAMILY-SPECIFIC STATISTICS")
    print("-"*80)
    
    for pair, distances in results['family_pairs'].items():
        if '-' not in pair and distances:  # Within-family
            print(f"\n{pair} (within-family):")
            print(f"  Mean distance: {np.mean(distances):.4f}")
            print(f"  Std deviation: {np.std(distances):.4f}")
            print(f"  N pairs: {len(distances)}")
    
    print("\n" + "-"*80)
    print("BETWEEN-FAMILY STATISTICS")
    print("-"*80)
    
    between_stats = defaultdict(list)
    for pair, distances in results['family_pairs'].items():
        if '-' in pair and distances:
            families = pair.split('-')
            key = tuple(sorted(families))
            between_stats[key].extend(distances)
    
    for families, distances in sorted(between_stats.items()):
        print(f"\n{families[0]} <-> {families[1]}:")
        print(f"  Mean distance: {np.mean(distances):.4f}")
        print(f"  Std deviation: {np.std(distances):.4f}")
        print(f"  N pairs: {len(distances)}")

def main():
    """Main analysis function."""
    print("Loading evaluation results...")
    distributions = load_results()
    
    if not distributions:
        print("No results found!")
        return
    
    print(f"Loaded results for {len(distributions)} models")
    
    # Calculate distances
    print("\nCalculating pairwise distances...")
    avg_distances, models, benchmark_distances = calculate_distances(distributions)
    
    # Test hypothesis
    print("Testing family resemblance hypothesis...")
    results = test_family_resemblance(avg_distances, models)
    
    # Print results
    print_detailed_results(results)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_results(avg_distances, models, results, benchmark_distances, distributions)
    
    print("\nAnalysis complete! Visualizations saved to 'family_resemblance_analysis.png'")

if __name__ == "__main__":
    main()