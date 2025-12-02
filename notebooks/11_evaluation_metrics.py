#!/usr/bin/env python3
"""
Evaluation Metrics for Recommendation Systems
Implements comprehensive evaluation metrics for Friend Finder and Study Buddy models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
import warnings
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'notebooks'))
from utils.create_embeddings import load_marketing_embeddings, load_profiles_embeddings

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'baseline'
EVAL_RESULTS_DIR = PROJECT_ROOT / 'results' / 'evaluation'
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class RecommendationEvaluator:
    """
    Comprehensive evaluation framework for recommendation systems.
    Implements Precision@k, Recall@k, F1-Score, MRR, Coverage, and Diversity metrics.
    """
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
    
    def precision_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate Precision@k.
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant (ground truth) item IDs
            k: Number of top recommendations to consider
        
        Returns:
            Precision@k score
        """
        if k == 0:
            return 0.0
        
        top_k_recommended = recommended[:k]
        if len(top_k_recommended) == 0:
            return 0.0
        
        relevant_recommended = len(set(top_k_recommended) & set(relevant))
        return relevant_recommended / len(top_k_recommended)
    
    def recall_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate Recall@k.
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant (ground truth) item IDs
            k: Number of top recommendations to consider
        
        Returns:
            Recall@k score
        """
        if len(relevant) == 0:
            return 0.0
        
        top_k_recommended = recommended[:k]
        relevant_recommended = len(set(top_k_recommended) & set(relevant))
        return relevant_recommended / len(relevant)
    
    def f1_score_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate F1-Score@k.
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant (ground truth) item IDs
            k: Number of top recommendations to consider
        
        Returns:
            F1-Score@k
        """
        precision = self.precision_at_k(recommended, relevant, k)
        recall = self.recall_at_k(recommended, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mean_reciprocal_rank(self, recommended: List[int], relevant: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant (ground truth) item IDs
        
        Returns:
            MRR score
        """
        if len(relevant) == 0:
            return 0.0
        
        for rank, item in enumerate(recommended, 1):
            if item in relevant:
                return 1.0 / rank
        
        return 0.0
    
    def coverage(self, all_recommendations: Dict[int, List[int]], all_items: List[int]) -> float:
        """
        Calculate Coverage: proportion of items that appear in at least one recommendation.
        
        Args:
            all_recommendations: Dictionary mapping user_id to list of recommended item IDs
            all_items: List of all possible item IDs
        
        Returns:
            Coverage score (0.0 to 1.0)
        """
        if len(all_items) == 0:
            return 0.0
        
        recommended_items = set()
        for recommendations in all_recommendations.values():
            recommended_items.update(recommendations)
        
        return len(recommended_items) / len(all_items)
    
    def intra_list_diversity(self, recommended: List[int], embeddings: np.ndarray, 
                            item_to_idx: Dict[int, int]) -> float:
        """
        Calculate intra-list diversity using cosine distance between recommended items.
        
        Args:
            recommended: List of recommended item IDs
            embeddings: Embedding matrix
            item_to_idx: Mapping from item ID to embedding index
        
        Returns:
            Diversity score (higher = more diverse)
        """
        if len(recommended) < 2:
            return 0.0
        
        indices = [item_to_idx[item] for item in recommended if item in item_to_idx]
        if len(indices) < 2:
            return 0.0
        
        rec_embeddings = embeddings[indices]
        similarities = cosine_similarity(rec_embeddings)
        
        n = len(similarities)
        if n == 0:
            return 0.0
        
        upper_triangle = np.triu(similarities, k=1)
        avg_similarity = np.sum(upper_triangle) / (n * (n - 1) / 2)
        
        diversity = 1.0 - avg_similarity
        return diversity
    
    def evaluate_recommendations(self, recommendations: Dict[int, List[int]], 
                                 ground_truth: Dict[int, List[int]], 
                                 k_values: List[int] = [5, 10]) -> Dict[str, float]:
        """
        Evaluate recommendations using multiple metrics.
        
        Args:
            recommendations: Dictionary mapping user_id to list of recommended item IDs
            ground_truth: Dictionary mapping user_id to list of relevant item IDs
            k_values: List of k values for Precision@k and Recall@k
        
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        all_precisions = {k: [] for k in k_values}
        all_recalls = {k: [] for k in k_values}
        all_f1s = {k: [] for k in k_values}
        all_mrrs = []
        
        for user_id in recommendations.keys():
            if user_id not in ground_truth:
                continue
            
            rec = recommendations[user_id]
            rel = ground_truth[user_id]
            
            for k in k_values:
                all_precisions[k].append(self.precision_at_k(rec, rel, k))
                all_recalls[k].append(self.recall_at_k(rec, rel, k))
                all_f1s[k].append(self.f1_score_at_k(rec, rel, k))
            
            all_mrrs.append(self.mean_reciprocal_rank(rec, rel))
        
        for k in k_values:
            results[f'Precision@{k}'] = np.mean(all_precisions[k]) if all_precisions[k] else 0.0
            results[f'Recall@{k}'] = np.mean(all_recalls[k]) if all_recalls[k] else 0.0
            results[f'F1@{k}'] = np.mean(all_f1s[k]) if all_f1s[k] else 0.0
        
        results['MRR'] = np.mean(all_mrrs) if all_mrrs else 0.0
        
        return results


def create_ground_truth_friend_finder(df_students: pd.DataFrame, 
                                     embeddings: np.ndarray,
                                     similarity_threshold: float = 0.7,
                                     max_relevant: int = 20) -> Dict[int, List[int]]:
    """
    Create ground truth for Friend Finder based on high similarity.
    
    Args:
        df_students: DataFrame with student data
        embeddings: Embedding matrix
        similarity_threshold: Minimum similarity to be considered relevant
        max_relevant: Maximum number of relevant items per user
    
    Returns:
        Dictionary mapping student_id to list of relevant student IDs
    """
    print("Creating ground truth for Friend Finder...")
    
    similarities = cosine_similarity(embeddings)
    ground_truth = {}
    
    student_ids = df_students['student_id'].values
    
    for i, student_id in enumerate(student_ids):
        similar_students = []
        for j, other_id in enumerate(student_ids):
            if i != j and similarities[i, j] >= similarity_threshold:
                similar_students.append((other_id, similarities[i, j]))
        
        similar_students.sort(key=lambda x: x[1], reverse=True)
        ground_truth[student_id] = [sid for sid, _ in similar_students[:max_relevant]]
    
    print(f"Created ground truth for {len(ground_truth)} students")
    return ground_truth


def create_ground_truth_study_buddy(df_students: pd.DataFrame,
                                   embeddings: np.ndarray,
                                   similarity_threshold: float = 0.6,
                                   gpa_complementary: bool = True,
                                   max_relevant: int = 20) -> Dict[int, List[int]]:
    """
    Create ground truth for Study Buddy based on similarity and course overlap.
    
    Args:
        df_students: DataFrame with student data
        embeddings: Embedding matrix
        similarity_threshold: Minimum similarity to be considered relevant
        gpa_complementary: Whether to consider GPA complementarity
        max_relevant: Maximum number of relevant items per user
    
    Returns:
        Dictionary mapping student_id to list of relevant student IDs
    """
    print("Creating ground truth for Study Buddy...")
    
    similarities = cosine_similarity(embeddings)
    ground_truth = {}
    
    student_ids = df_students['student_id'].values
    gpas = df_students['GPA'].values
    
    for i, student_id in enumerate(student_ids):
        candidate_students = []
        
        for j, other_id in enumerate(student_ids):
            if i == j:
                continue
            
            similarity = similarities[i, j]
            
            if similarity >= similarity_threshold:
                score = similarity
                
                if gpa_complementary:
                    gpa_diff = abs(gpas[i] - gpas[j])
                    if gpa_diff > 0.5:
                        score += 0.1
                
                candidate_students.append((other_id, score))
        
        candidate_students.sort(key=lambda x: x[1], reverse=True)
        ground_truth[student_id] = [sid for sid, _ in candidate_students[:max_relevant]]
    
    print(f"Created ground truth for {len(ground_truth)} students")
    return ground_truth


def evaluate_friend_finder_baseline():
    """
    Evaluate Friend Finder baseline model.
    """
    print("\n" + "="*60)
    print("EVALUATING FRIEND FINDER BASELINE MODEL")
    print("="*60)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("friend_finder", PROJECT_ROOT / 'notebooks' / '04_friend_finder_baseline_embedding.py')
    friend_finder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(friend_finder_module)
    FriendFinderBaseline = friend_finder_module.FriendFinderBaseline
    load_optimal_k = friend_finder_module.load_optimal_k
    
    print("\nLoading data and model...")
    df_marketing = pd.read_csv(PROCESSED_DATA_DIR / 'marketing_processed.csv')
    print(f"Loaded {len(df_marketing)} students")
    
    optimal_k = load_optimal_k(default_k=8)
    model = FriendFinderBaseline(n_clusters=optimal_k)
    model.train(df_marketing, force_recreate_embeddings=False)
    
    print("\nGenerating recommendations for evaluation...")
    test_students = df_marketing['student_id'].sample(n=min(500, len(df_marketing)), random_state=42).tolist()
    
    recommendations = {}
    for student_id in test_students:
        recs = model.recommend_friends(student_id, top_k=10)
        recommendations[student_id] = recs
    
    print("\nCreating ground truth...")
    embeddings = load_marketing_embeddings()
    if embeddings is None:
        print("Error: Could not load embeddings")
        return None, None, None, None
    ground_truth = create_ground_truth_friend_finder(df_marketing, embeddings)
    
    print("\nCalculating metrics...")
    evaluator = RecommendationEvaluator()
    metrics = evaluator.evaluate_recommendations(recommendations, ground_truth, k_values=[5, 10])
    
    all_items = df_marketing['student_id'].tolist()
    coverage = evaluator.coverage(recommendations, all_items)
    metrics['Coverage'] = coverage
    
    print("\n" + "="*60)
    print("FRIEND FINDER BASELINE - EVALUATION RESULTS")
    print("="*60)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    
    return metrics, model, recommendations, ground_truth


def evaluate_study_buddy_baseline():
    """
    Evaluate Study Buddy baseline model.
    """
    print("\n" + "="*60)
    print("EVALUATING STUDY BUDDY BASELINE MODEL")
    print("="*60)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("study_buddy", PROJECT_ROOT / 'notebooks' / '05_study_buddy_baseline_embedding.py')
    study_buddy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study_buddy_module)
    StudyBuddyBaseline = study_buddy_module.StudyBuddyBaseline
    
    print("\nLoading data and model...")
    df_profiles = pd.read_csv(PROCESSED_DATA_DIR / 'profiles_processed.csv')
    print(f"Loaded {len(df_profiles)} students")
    
    model = StudyBuddyBaseline()
    model.train(df_profiles, force_recreate_embeddings=False)
    
    print("\nGenerating recommendations for evaluation...")
    test_students = df_profiles['student_id'].sample(n=min(500, len(df_profiles)), random_state=42).tolist()
    
    recommendations = {}
    for student_id in test_students:
        recs = model.recommend_study_buddies(student_id, top_k=10)
        recommendations[student_id] = recs
    
    print("\nCreating ground truth...")
    embeddings = load_profiles_embeddings()
    if embeddings is None:
        print("Error: Could not load embeddings")
        return None, None, None, None
    ground_truth = create_ground_truth_study_buddy(df_profiles, embeddings)
    
    print("\nCalculating metrics...")
    evaluator = RecommendationEvaluator()
    metrics = evaluator.evaluate_recommendations(recommendations, ground_truth, k_values=[5, 10])
    
    all_items = df_profiles['student_id'].tolist()
    coverage = evaluator.coverage(recommendations, all_items)
    metrics['Coverage'] = coverage
    
    print("\n" + "="*60)
    print("STUDY BUDDY BASELINE - EVALUATION RESULTS")
    print("="*60)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    
    return metrics, model, recommendations, ground_truth


def calculate_diversity_metrics(model, recommendations: Dict[int, List[int]], 
                               embeddings: np.ndarray, student_ids: pd.Series):
    """
    Calculate diversity metrics for recommendations.
    
    Args:
        model: Trained model
        recommendations: Dictionary of recommendations
        embeddings: Embedding matrix
        student_ids: Series of student IDs
    
    Returns:
        Dictionary of diversity metrics
    """
    print("\nCalculating diversity metrics...")
    
    evaluator = RecommendationEvaluator()
    item_to_idx = {sid: idx for idx, sid in enumerate(student_ids)}
    
    diversities = []
    for student_id, recs in recommendations.items():
        if len(recs) > 1:
            diversity = evaluator.intra_list_diversity(recs, embeddings, item_to_idx)
            diversities.append(diversity)
    
    avg_diversity = np.mean(diversities) if diversities else 0.0
    
    return {'Average_Intra_List_Diversity': avg_diversity}


def create_evaluation_visualizations(friend_finder_metrics: Dict, 
                                     study_buddy_metrics: Dict,
                                     save_path: Path):
    """
    Create multiple separate visualizations for evaluation metrics.
    """
    print("\nCreating evaluation visualizations...")
    
    models = ['Friend Finder', 'Study Buddy']
    precision_5 = [friend_finder_metrics.get('Precision@5', 0), study_buddy_metrics.get('Precision@5', 0)]
    precision_10 = [friend_finder_metrics.get('Precision@10', 0), study_buddy_metrics.get('Precision@10', 0)]
    recall_5 = [friend_finder_metrics.get('Recall@5', 0), study_buddy_metrics.get('Recall@5', 0)]
    recall_10 = [friend_finder_metrics.get('Recall@10', 0), study_buddy_metrics.get('Recall@10', 0)]
    mrr_values = [friend_finder_metrics.get('MRR', 0), study_buddy_metrics.get('MRR', 0)]
    coverage_values = [friend_finder_metrics.get('Coverage', 0), study_buddy_metrics.get('Coverage', 0)]
    f1_5 = [friend_finder_metrics.get('F1@5', 0), study_buddy_metrics.get('F1@5', 0)]
    f1_10 = [friend_finder_metrics.get('F1@10', 0), study_buddy_metrics.get('F1@10', 0)]
    diversity_ff = friend_finder_metrics.get('Average_Intra_List_Diversity', 0)
    diversity_sb = study_buddy_metrics.get('Average_Intra_List_Diversity', 0)
    
    x = np.arange(len(models))
    width = 0.35
    
    # 1. Precision Metrics Comparison
    print("  Creating precision metrics visualization...")
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, precision_5, width, label='Precision@5', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, precision_10, width, label='Precision@10', alpha=0.8, color='steelblue')
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision Metrics Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    for i, (p5, p10) in enumerate(zip(precision_5, precision_10)):
        plt.text(i - width/2, p5 + 0.01, f'{p5:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, p10 + 0.01, f'{p10:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path / 'eval_precision_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Recall Metrics Comparison
    print("  Creating recall metrics visualization...")
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, recall_5, width, label='Recall@5', alpha=0.8, color='lightcoral')
    plt.bar(x + width/2, recall_10, width, label='Recall@10', alpha=0.8, color='crimson')
    plt.ylabel('Recall', fontsize=12)
    plt.title('Recall Metrics Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    for i, (r5, r10) in enumerate(zip(recall_5, recall_10)):
        plt.text(i - width/2, r5 + 0.01, f'{r5:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, r10 + 0.01, f'{r10:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path / 'eval_recall_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. F1-Score Comparison
    print("  Creating F1-score visualization...")
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, f1_5, width, label='F1@5', alpha=0.8, color='lightgreen')
    plt.bar(x + width/2, f1_10, width, label='F1@10', alpha=0.8, color='darkgreen')
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('F1-Score Metrics Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    for i, (f5, f10) in enumerate(zip(f1_5, f1_10)):
        plt.text(i - width/2, f5 + 0.01, f'{f5:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, f10 + 0.01, f'{f10:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path / 'eval_f1_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. MRR Comparison
    print("  Creating MRR visualization...")
    plt.figure(figsize=(8, 6))
    plt.bar(models, mrr_values, alpha=0.8, color='green', edgecolor='black')
    plt.ylabel('Mean Reciprocal Rank (MRR)', fontsize=12)
    plt.title('Mean Reciprocal Rank Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    for i, mrr in enumerate(mrr_values):
        plt.text(i, mrr + 0.01, f'{mrr:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'eval_mrr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Coverage Comparison
    print("  Creating coverage visualization...")
    plt.figure(figsize=(8, 6))
    plt.bar(models, coverage_values, alpha=0.8, color='orange', edgecolor='black')
    plt.ylabel('Coverage', fontsize=12)
    plt.title('Coverage Metric Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    for i, cov in enumerate(coverage_values):
        plt.text(i, cov + 0.01, f'{cov:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'eval_coverage_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Diversity Comparison
    print("  Creating diversity visualization...")
    plt.figure(figsize=(8, 6))
    diversity_values = [diversity_ff, diversity_sb]
    plt.bar(models, diversity_values, alpha=0.8, color='purple', edgecolor='black')
    plt.ylabel('Average Intra-List Diversity', fontsize=12)
    plt.title('Recommendation Diversity Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    for i, div in enumerate(diversity_values):
        plt.text(i, div + 0.01, f'{div:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'eval_diversity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. All Metrics Radar Chart
    print("  Creating comprehensive metrics radar chart...")
    try:
        categories = ['Precision@5', 'Recall@5', 'F1@5', 'Precision@10', 'Recall@10', 'F1@10', 'MRR', 'Coverage']
        ff_values = [
            friend_finder_metrics.get('Precision@5', 0),
            friend_finder_metrics.get('Recall@5', 0),
            friend_finder_metrics.get('F1@5', 0),
            friend_finder_metrics.get('Precision@10', 0),
            friend_finder_metrics.get('Recall@10', 0),
            friend_finder_metrics.get('F1@10', 0),
            friend_finder_metrics.get('MRR', 0),
            friend_finder_metrics.get('Coverage', 0)
        ]
        sb_values = [
            study_buddy_metrics.get('Precision@5', 0),
            study_buddy_metrics.get('Recall@5', 0),
            study_buddy_metrics.get('F1@5', 0),
            study_buddy_metrics.get('Precision@10', 0),
            study_buddy_metrics.get('Recall@10', 0),
            study_buddy_metrics.get('F1@10', 0),
            study_buddy_metrics.get('MRR', 0),
            study_buddy_metrics.get('Coverage', 0)
        ]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        x_pos = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x_pos - width/2, ff_values, width, label='Friend Finder', alpha=0.8, color='orange')
        ax.bar(x_pos + width/2, sb_values, width, label='Study Buddy', alpha=0.8, color='green')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Comprehensive Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_path / 'eval_comprehensive_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  Skipping radar chart: {e}")
    
    # 8. Metric Summary Table Visualization
    print("  Creating metrics summary table...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    data = [
        ['Precision@5', f"{precision_5[0]:.4f}", f"{precision_5[1]:.4f}"],
        ['Precision@10', f"{precision_10[0]:.4f}", f"{precision_10[1]:.4f}"],
        ['Recall@5', f"{recall_5[0]:.4f}", f"{recall_5[1]:.4f}"],
        ['Recall@10', f"{recall_10[0]:.4f}", f"{recall_10[1]:.4f}"],
        ['F1@5', f"{f1_5[0]:.4f}", f"{f1_5[1]:.4f}"],
        ['F1@10', f"{f1_10[0]:.4f}", f"{f1_10[1]:.4f}"],
        ['MRR', f"{mrr_values[0]:.4f}", f"{mrr_values[1]:.4f}"],
        ['Coverage', f"{coverage_values[0]:.4f}", f"{coverage_values[1]:.4f}"],
        ['Diversity', f"{diversity_ff:.4f}", f"{diversity_sb:.4f}"]
    ]
    
    table = ax.table(cellText=data, colLabels=['Metric', 'Friend Finder', 'Study Buddy'],
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title('Evaluation Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path / 'eval_metrics_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  All visualizations saved to: {save_path}")


def main():
    """
    Main function to run comprehensive evaluation.
    """
    print("="*60)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("="*60)
    
    ff_metrics, ff_model, ff_recs, ff_gt = evaluate_friend_finder_baseline()
    sb_metrics, sb_model, sb_recs, sb_gt = evaluate_study_buddy_baseline()
    
    if ff_metrics is None or sb_metrics is None:
        print("Evaluation failed. Please check errors above.")
        return
    
    print("\n" + "="*60)
    print("DIVERSITY METRICS")
    print("="*60)
    
    df_marketing = pd.read_csv(PROCESSED_DATA_DIR / 'marketing_processed.csv')
    marketing_embeddings = load_marketing_embeddings()
    if marketing_embeddings is not None and ff_model is not None:
        ff_diversity = calculate_diversity_metrics(
            ff_model, ff_recs, marketing_embeddings, df_marketing['student_id']
        )
        ff_metrics.update(ff_diversity)
        print(f"Friend Finder Diversity: {ff_diversity['Average_Intra_List_Diversity']:.4f}")
    
    df_profiles = pd.read_csv(PROCESSED_DATA_DIR / 'profiles_processed.csv')
    profiles_embeddings = load_profiles_embeddings()
    if profiles_embeddings is not None and sb_model is not None:
        sb_diversity = calculate_diversity_metrics(
            sb_model, sb_recs, profiles_embeddings, df_profiles['student_id']
        )
        sb_metrics.update(sb_diversity)
        print(f"Study Buddy Diversity: {sb_diversity['Average_Intra_List_Diversity']:.4f}")
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    results = {
        'friend_finder': ff_metrics,
        'study_buddy': sb_metrics
    }
    
    with open(EVAL_RESULTS_DIR / 'evaluation_metrics.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    results_df = pd.DataFrame([ff_metrics, sb_metrics], index=['Friend_Finder', 'Study_Buddy'])
    results_df.to_csv(EVAL_RESULTS_DIR / 'evaluation_metrics.csv')
    
    print(f"Results saved to: {EVAL_RESULTS_DIR}")
    
    create_evaluation_visualizations(ff_metrics, sb_metrics, EVAL_RESULTS_DIR)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()

