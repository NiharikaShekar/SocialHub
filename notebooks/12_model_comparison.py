#!/usr/bin/env python3
"""
Model Comparison: Baseline vs Advanced (GNN)
Compares K-Means baseline with GraphSAGE GNN models using standard evaluation metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
import warnings
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'notebooks'))
from utils.create_embeddings import load_marketing_embeddings, load_profiles_embeddings
# Import evaluation functions
import importlib.util
eval_spec = importlib.util.spec_from_file_location("eval_metrics", PROJECT_ROOT / 'notebooks' / '11_evaluation_metrics.py')
eval_module = importlib.util.module_from_spec(eval_spec)
eval_spec.loader.exec_module(eval_module)
RecommendationEvaluator = eval_module.RecommendationEvaluator
create_ground_truth_friend_finder = eval_module.create_ground_truth_friend_finder
create_ground_truth_study_buddy = eval_module.create_ground_truth_study_buddy

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'baseline'
COMPARISON_RESULTS_DIR = PROJECT_ROOT / 'results' / 'model_comparison'
COMPARISON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_baseline_friend_finder():
    """Load baseline Friend Finder model."""
    print("Loading baseline Friend Finder model...")
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("friend_finder", PROJECT_ROOT / 'notebooks' / '04_friend_finder_baseline_embedding.py')
    friend_finder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(friend_finder_module)
    FriendFinderBaseline = friend_finder_module.FriendFinderBaseline
    load_optimal_k = friend_finder_module.load_optimal_k
    
    df_marketing = pd.read_csv(PROCESSED_DATA_DIR / 'marketing_processed.csv')
    optimal_k = load_optimal_k(default_k=8)
    model = FriendFinderBaseline(n_clusters=optimal_k)
    model.train(df_marketing, force_recreate_embeddings=False)
    
    return model, df_marketing


def load_baseline_study_buddy():
    """Load baseline Study Buddy model."""
    print("Loading baseline Study Buddy model...")
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("study_buddy", PROJECT_ROOT / 'notebooks' / '05_study_buddy_baseline_embedding.py')
    study_buddy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study_buddy_module)
    StudyBuddyBaseline = study_buddy_module.StudyBuddyBaseline
    
    df_profiles = pd.read_csv(PROCESSED_DATA_DIR / 'profiles_processed.csv')
    model = StudyBuddyBaseline()
    model.train(df_profiles, force_recreate_embeddings=False)
    
    return model, df_profiles


def load_gnn_study_buddy():
    """Load GNN Study Buddy model."""
    print("Loading GNN Study Buddy model...")
    
    model_files_dir = PROJECT_ROOT / 'study_buddy_api' / 'model_files'
    
    # Load embeddings
    gnn_embeddings = np.load(model_files_dir / 'final_user_vectors.npy')
    
    # Load artifacts
    with open(model_files_dir / 'api_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    return gnn_embeddings, artifacts


def load_gnn_friend_finder():
    """Load GNN Friend Finder model if available."""
    print("Loading GNN Friend Finder model...")
    
    # Check if friend_finder_api has model files
    friend_finder_model_dir = PROJECT_ROOT / 'friend_finder_api' / 'model_files'
    
    if (friend_finder_model_dir / 'user_embeddings.npy').exists():
        gnn_embeddings = np.load(friend_finder_model_dir / 'user_embeddings.npy')
        with open(friend_finder_model_dir / 'artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return gnn_embeddings, artifacts
    else:
        print("  Friend Finder GNN files not found. Skipping...")
        return None, None


def gnn_recommend_friends(gnn_embeddings, student_id, student_ids, top_k=10):
    """Get recommendations from GNN embeddings using cosine similarity."""
    if student_id not in student_ids:
        return []
    
    student_idx = student_ids.index(student_id)
    student_embedding = gnn_embeddings[student_idx].reshape(1, -1)
    
    similarities = cosine_similarity(student_embedding, gnn_embeddings)[0]
    
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    recommendations = [student_ids[i] for i in top_indices]
    
    return recommendations


def compare_models_friend_finder():
    """Compare baseline vs GNN for Friend Finder."""
    print("\n" + "="*60)
    print("MODEL COMPARISON - FRIEND FINDER")
    print("="*60)
    
    # Load baseline
    baseline_model, df_marketing = load_baseline_friend_finder()
    
    # Load GNN
    gnn_embeddings, gnn_artifacts = load_gnn_friend_finder()
    
    if gnn_embeddings is None:
        print("GNN model not available for Friend Finder. Skipping comparison.")
        return None
    
    # Get test students
    test_students = df_marketing['student_id'].sample(n=min(500, len(df_marketing)), random_state=42).tolist()
    student_ids = df_marketing['student_id'].tolist()
    
    # Generate recommendations
    print("\nGenerating baseline recommendations...")
    baseline_recs = {}
    for student_id in test_students:
        recs = baseline_model.recommend_friends(student_id, top_k=10)
        baseline_recs[student_id] = recs
    
    print("Generating GNN recommendations...")
    gnn_recs = {}
    for student_id in test_students:
        recs = gnn_recommend_friends(gnn_embeddings, student_id, student_ids, top_k=10)
        gnn_recs[student_id] = recs
    
    # Create ground truth
    print("Creating ground truth...")
    baseline_embeddings = load_marketing_embeddings()
    ground_truth = create_ground_truth_friend_finder(df_marketing, baseline_embeddings)
    
    # Evaluate both models
    print("Evaluating models...")
    evaluator = RecommendationEvaluator()
    
    baseline_metrics = evaluator.evaluate_recommendations(baseline_recs, ground_truth, k_values=[5, 10])
    gnn_metrics = evaluator.evaluate_recommendations(gnn_recs, ground_truth, k_values=[5, 10])
    
    # Coverage
    all_items = student_ids
    baseline_metrics['Coverage'] = evaluator.coverage(baseline_recs, all_items)
    gnn_metrics['Coverage'] = evaluator.coverage(gnn_recs, all_items)
    
    return {
        'baseline': baseline_metrics,
        'gnn': gnn_metrics,
        'model_name': 'Friend Finder'
    }


def compare_models_study_buddy():
    """Compare baseline vs GNN for Study Buddy."""
    print("\n" + "="*60)
    print("MODEL COMPARISON - STUDY BUDDY")
    print("="*60)
    
    # Load baseline
    baseline_model, df_profiles = load_baseline_study_buddy()
    
    # Load GNN
    gnn_embeddings, gnn_artifacts = load_gnn_study_buddy()
    
    # Get test students
    test_students = df_profiles['student_id'].sample(n=min(500, len(df_profiles)), random_state=42).tolist()
    student_ids = df_profiles['student_id'].tolist()
    
    # Generate recommendations
    print("\nGenerating baseline recommendations...")
    baseline_recs = {}
    for student_id in test_students:
        recs = baseline_model.recommend_study_buddies(student_id, top_k=10)
        baseline_recs[student_id] = recs
    
    print("Generating GNN recommendations...")
    gnn_recs = {}
    for student_id in test_students:
        recs = gnn_recommend_friends(gnn_embeddings, student_id, student_ids, top_k=10)
        gnn_recs[student_id] = recs
    
    # Create ground truth
    print("Creating ground truth...")
    baseline_embeddings = load_profiles_embeddings()
    ground_truth = create_ground_truth_study_buddy(df_profiles, baseline_embeddings)
    
    # Evaluate both models
    print("Evaluating models...")
    evaluator = RecommendationEvaluator()
    
    baseline_metrics = evaluator.evaluate_recommendations(baseline_recs, ground_truth, k_values=[5, 10])
    gnn_metrics = evaluator.evaluate_recommendations(gnn_recs, ground_truth, k_values=[5, 10])
    
    # Coverage
    all_items = student_ids
    baseline_metrics['Coverage'] = evaluator.coverage(baseline_recs, all_items)
    gnn_metrics['Coverage'] = evaluator.coverage(gnn_recs, all_items)
    
    return {
        'baseline': baseline_metrics,
        'gnn': gnn_metrics,
        'model_name': 'Study Buddy'
    }


def statistical_significance_test(baseline_scores, gnn_scores, metric_name):
    """Perform statistical significance test (t-test)."""
    if len(baseline_scores) < 2 or len(gnn_scores) < 2:
        return None, None
    
    t_stat, p_value = stats.ttest_ind(baseline_scores, gnn_scores)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'metric': metric_name
    }


def create_comparison_visualizations(ff_comparison, sb_comparison, save_path):
    """Create visualizations comparing baseline vs GNN."""
    print("\nCreating comparison visualizations...")
    
    metrics_to_compare = ['Precision@5', 'Precision@10', 'Recall@5', 'Recall@10', 
                          'F1@5', 'F1@10', 'MRR', 'Coverage']
    
    # 1. Friend Finder Comparison
    if ff_comparison:
        print("  Creating Friend Finder comparison charts...")
        ff_baseline = ff_comparison['baseline']
        ff_gnn = ff_comparison['gnn']
        
        for metric in metrics_to_compare:
            if metric in ff_baseline and metric in ff_gnn:
                plt.figure(figsize=(8, 6))
                models = ['Baseline (K-Means)', 'GNN (GraphSAGE)']
                values = [ff_baseline[metric], ff_gnn[metric]]
                colors = ['orange', 'blue']
                
                bars = plt.bar(models, values, alpha=0.8, color=colors, edgecolor='black')
                plt.ylabel(metric, fontsize=12)
                plt.title(f'Friend Finder - {metric} Comparison', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3, axis='y')
                
                for i, (bar, val) in enumerate(zip(bars, values)):
                    plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                            f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(save_path / f'ff_comparison_{metric.lower().replace("@", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    # 2. Study Buddy Comparison
    if sb_comparison:
        print("  Creating Study Buddy comparison charts...")
        sb_baseline = sb_comparison['baseline']
        sb_gnn = sb_comparison['gnn']
        
        for metric in metrics_to_compare:
            if metric in sb_baseline and metric in sb_gnn:
                plt.figure(figsize=(8, 6))
                models = ['Baseline (Embedding)', 'GNN (GraphSAGE)']
                values = [sb_baseline[metric], sb_gnn[metric]]
                colors = ['green', 'purple']
                
                bars = plt.bar(models, values, alpha=0.8, color=colors, edgecolor='black')
                plt.ylabel(metric, fontsize=12)
                plt.title(f'Study Buddy - {metric} Comparison', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3, axis='y')
                
                for i, (bar, val) in enumerate(zip(bars, values)):
                    plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                            f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(save_path / f'sb_comparison_{metric.lower().replace("@", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    # 3. Side-by-side comparison
    if ff_comparison and sb_comparison:
        print("  Creating comprehensive comparison chart...")
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Baseline vs GNN - Comprehensive Comparison', fontsize=16, fontweight='bold')
        
        metrics_subset = ['Precision@5', 'Recall@5', 'F1@5', 'MRR']
        
        for idx, metric in enumerate(metrics_subset):
            # Friend Finder
            ff_b = ff_comparison['baseline'].get(metric, 0)
            ff_g = ff_comparison['gnn'].get(metric, 0)
            axes[0, idx].bar(['Baseline', 'GNN'], [ff_b, ff_g], alpha=0.8, color=['orange', 'blue'])
            axes[0, idx].set_title(f'Friend Finder - {metric}')
            axes[0, idx].set_ylabel('Score')
            axes[0, idx].grid(True, alpha=0.3, axis='y')
            
            # Study Buddy
            sb_b = sb_comparison['baseline'].get(metric, 0)
            sb_g = sb_comparison['gnn'].get(metric, 0)
            axes[1, idx].bar(['Baseline', 'GNN'], [sb_b, sb_g], alpha=0.8, color=['green', 'purple'])
            axes[1, idx].set_title(f'Study Buddy - {metric}')
            axes[1, idx].set_ylabel('Score')
            axes[1, idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path / 'comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  All comparison visualizations saved to: {save_path}")


def main():
    """Main function for model comparison."""
    print("="*60)
    print("MODEL COMPARISON: BASELINE vs GNN")
    print("="*60)
    
    # Compare models
    ff_comparison = compare_models_friend_finder()
    sb_comparison = compare_models_study_buddy()
    
    # Print results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    if ff_comparison:
        print("\nFRIEND FINDER:")
        print("Baseline Metrics:")
        for metric, value in ff_comparison['baseline'].items():
            print(f"  {metric:20s}: {value:.4f}")
        print("\nGNN Metrics:")
        for metric, value in ff_comparison['gnn'].items():
            print(f"  {metric:20s}: {value:.4f}")
    
    if sb_comparison:
        print("\nSTUDY BUDDY:")
        print("Baseline Metrics:")
        for metric, value in sb_comparison['baseline'].items():
            print(f"  {metric:20s}: {value:.4f}")
        print("\nGNN Metrics:")
        for metric, value in sb_comparison['gnn'].items():
            print(f"  {metric:20s}: {value:.4f}")
    
    # Save results
    results = {
        'friend_finder': ff_comparison,
        'study_buddy': sb_comparison
    }
    
    with open(COMPARISON_RESULTS_DIR / 'model_comparison_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Create comparison DataFrame
    if ff_comparison and sb_comparison:
        comparison_data = []
        for metric in ['Precision@5', 'Precision@10', 'Recall@5', 'Recall@10', 'F1@5', 'F1@10', 'MRR', 'Coverage']:
            comparison_data.append({
                'Metric': metric,
                'FF_Baseline': ff_comparison['baseline'].get(metric, 0),
                'FF_GNN': ff_comparison['gnn'].get(metric, 0),
                'SB_Baseline': sb_comparison['baseline'].get(metric, 0),
                'SB_GNN': sb_comparison['gnn'].get(metric, 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(COMPARISON_RESULTS_DIR / 'model_comparison_results.csv', index=False)
    
    # Create visualizations
    create_comparison_visualizations(ff_comparison, sb_comparison, COMPARISON_RESULTS_DIR)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON COMPLETED!")
    print("="*60)
    print(f"Results saved to: {COMPARISON_RESULTS_DIR}")


if __name__ == "__main__":
    main()

