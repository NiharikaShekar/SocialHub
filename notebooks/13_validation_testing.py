#!/usr/bin/env python3
"""
Validation and Testing Framework
Implements cross-validation and statistical significance testing for recommendation models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
import warnings
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'notebooks'))

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
VALIDATION_RESULTS_DIR = PROJECT_ROOT / 'results' / 'validation'
VALIDATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def cross_validate_model(model_func, data, n_splits=5, random_state=42):
    """
    Perform k-fold cross-validation on a model.
    
    Args:
        model_func: Function that trains and evaluates a model
        data: Dataset to split
        n_splits: Number of folds
        random_state: Random seed
    
    Returns:
        Dictionary of metrics across folds
    """
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
        print(f"  Fold {fold}/{n_splits}...")
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        metrics = model_func(train_data, test_data)
        metrics['fold'] = fold
        fold_metrics.append(metrics)
    
    # Aggregate results
    aggregated = {}
    metric_names = [k for k in fold_metrics[0].keys() if k != 'fold']
    
    for metric in metric_names:
        values = [m[metric] for m in fold_metrics]
        aggregated[f'{metric}_mean'] = np.mean(values)
        aggregated[f'{metric}_std'] = np.std(values)
        aggregated[f'{metric}_min'] = np.min(values)
        aggregated[f'{metric}_max'] = np.max(values)
        aggregated[f'{metric}_values'] = values
    
    return aggregated, fold_metrics


def paired_t_test(metric1_values, metric2_values, metric_name):
    """
    Perform paired t-test between two sets of metric values.
    
    Args:
        metric1_values: First set of metric values
        metric2_values: Second set of metric values
        metric_name: Name of the metric
    
    Returns:
        Dictionary with test results
    """
    if len(metric1_values) != len(metric2_values):
        return None
    
    t_stat, p_value = stats.ttest_rel(metric1_values, metric2_values)
    
    return {
        'metric': metric_name,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_diff': np.mean(metric1_values) - np.mean(metric2_values),
        'mean1': np.mean(metric1_values),
        'mean2': np.mean(metric2_values)
    }


def validate_friend_finder_baseline():
    """Validate Friend Finder baseline model."""
    print("\n" + "="*60)
    print("VALIDATING FRIEND FINDER BASELINE")
    print("="*60)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("friend_finder", PROJECT_ROOT / 'notebooks' / '04_friend_finder_baseline_embedding.py')
    friend_finder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(friend_finder_module)
    FriendFinderBaseline = friend_finder_module.FriendFinderBaseline
    load_optimal_k = friend_finder_module.load_optimal_k
    
    df_marketing = pd.read_csv(PROCESSED_DATA_DIR / 'marketing_processed.csv')
    
    def train_and_evaluate(train_data, test_data):
        optimal_k = load_optimal_k(default_k=8)
        model = FriendFinderBaseline(n_clusters=optimal_k)
        model.train(train_data, force_recreate_embeddings=False)
        
        # Evaluate on TRAINING set students (can't recommend for unseen users)
        # This tests if model can make recommendations for known users
        eval_students = train_data['student_id'].sample(n=min(100, len(train_data)), random_state=42).tolist()
        
        correct = 0
        total = 0
        
        for student_id in eval_students:
            recs = model.recommend_friends(student_id, top_k=5)
            if len(recs) > 0:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {'accuracy': accuracy, 'coverage': len(eval_students) / len(train_data)}
    
    results, fold_results = cross_validate_model(train_and_evaluate, df_marketing, n_splits=5)
    
    print("\nCross-Validation Results:")
    for key, value in results.items():
        if not isinstance(value, list):
            print(f"  {key}: {value:.4f}")
    
    return results, fold_results


def validate_study_buddy_baseline():
    """Validate Study Buddy baseline model."""
    print("\n" + "="*60)
    print("VALIDATING STUDY BUDDY BASELINE")
    print("="*60)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("study_buddy", PROJECT_ROOT / 'notebooks' / '05_study_buddy_baseline_embedding.py')
    study_buddy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study_buddy_module)
    StudyBuddyBaseline = study_buddy_module.StudyBuddyBaseline
    
    df_profiles = pd.read_csv(PROCESSED_DATA_DIR / 'profiles_processed.csv')
    
    def train_and_evaluate(train_data, test_data):
        model = StudyBuddyBaseline()
        model.train(train_data, force_recreate_embeddings=False)
        
        # Evaluate on TRAINING set students (can't recommend for unseen users)
        # This tests if model can make recommendations for known users
        eval_students = train_data['student_id'].sample(n=min(100, len(train_data)), random_state=42).tolist()
        
        correct = 0
        total = 0
        
        for student_id in eval_students:
            recs = model.recommend_study_buddies(student_id, top_k=5)
            if len(recs) > 0:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {'accuracy': accuracy, 'coverage': len(eval_students) / len(train_data)}
    
    results, fold_results = cross_validate_model(train_and_evaluate, df_profiles, n_splits=5)
    
    print("\nCross-Validation Results:")
    for key, value in results.items():
        if not isinstance(value, list):
            print(f"  {key}: {value:.4f}")
    
    return results, fold_results


def create_validation_visualizations(ff_results, sb_results, save_path):
    """Create visualizations for validation results."""
    print("\nCreating validation visualizations...")
    
    # 1. Cross-validation accuracy across folds
    if ff_results and 'accuracy_values' in ff_results:
        plt.figure(figsize=(10, 6))
        folds = range(1, len(ff_results['accuracy_values']) + 1)
        plt.plot(folds, ff_results['accuracy_values'], marker='o', label='Friend Finder', linewidth=2, markersize=8)
        if sb_results and 'accuracy_values' in sb_results:
            plt.plot(folds, sb_results['accuracy_values'], marker='s', label='Study Buddy', linewidth=2, markersize=8)
        plt.xlabel('Fold', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Cross-Validation Accuracy Across Folds', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(folds)
        plt.tight_layout()
        plt.savefig(save_path / 'cv_accuracy_across_folds.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Metric distribution with confidence intervals
    if ff_results:
        metrics_to_plot = ['accuracy', 'coverage']
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, metric in enumerate(metrics_to_plot):
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            
            if mean_key in ff_results:
                means = [ff_results[mean_key]]
                stds = [ff_results[std_key]]
                labels = ['Friend Finder']
                
                if sb_results and mean_key in sb_results:
                    means.append(sb_results[mean_key])
                    stds.append(sb_results[std_key])
                    labels.append('Study Buddy')
                
                x_pos = np.arange(len(means))
                axes[idx].bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5, 
                             color=['orange', 'green'][:len(means)], edgecolor='black')
                axes[idx].set_xticks(x_pos)
                axes[idx].set_xticklabels(labels)
                axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
                axes[idx].set_title(f'{metric.capitalize()} with Std Dev', fontsize=12, fontweight='bold')
                axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path / 'cv_metrics_with_confidence.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Validation visualizations saved to: {save_path}")


def main():
    """Main function for validation testing."""
    print("="*60)
    print("VALIDATION AND TESTING FRAMEWORK")
    print("="*60)
    
    # Validate models
    ff_results, ff_folds = validate_friend_finder_baseline()
    sb_results, sb_folds = validate_study_buddy_baseline()
    
    # Save results
    validation_results = {
        'friend_finder': ff_results,
        'study_buddy': sb_results,
        'friend_finder_folds': ff_folds,
        'study_buddy_folds': sb_folds
    }
    
    with open(VALIDATION_RESULTS_DIR / 'validation_results.pkl', 'wb') as f:
        pickle.dump(validation_results, f)
    
    # Create summary DataFrame
    summary_data = []
    if ff_results:
        for metric in ['accuracy', 'coverage']:
            summary_data.append({
                'Model': 'Friend Finder',
                'Metric': metric,
                'Mean': ff_results.get(f'{metric}_mean', 0),
                'Std': ff_results.get(f'{metric}_std', 0),
                'Min': ff_results.get(f'{metric}_min', 0),
                'Max': ff_results.get(f'{metric}_max', 0)
            })
    
    if sb_results:
        for metric in ['accuracy', 'coverage']:
            summary_data.append({
                'Model': 'Study Buddy',
                'Metric': metric,
                'Mean': sb_results.get(f'{metric}_mean', 0),
                'Std': sb_results.get(f'{metric}_std', 0),
                'Min': sb_results.get(f'{metric}_min', 0),
                'Max': sb_results.get(f'{metric}_max', 0)
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(VALIDATION_RESULTS_DIR / 'validation_summary.csv', index=False)
    
    # Create visualizations
    create_validation_visualizations(ff_results, sb_results, VALIDATION_RESULTS_DIR)
    
    print("\n" + "="*60)
    print("VALIDATION TESTING COMPLETED!")
    print("="*60)
    print(f"Results saved to: {VALIDATION_RESULTS_DIR}")


if __name__ == "__main__":
    main()

