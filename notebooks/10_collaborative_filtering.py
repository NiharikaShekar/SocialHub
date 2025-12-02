#!/usr/bin/env python3
"""
Collaborative Filtering Implementation
Implements matrix factorization-based collaborative filtering for friend and study buddy recommendations.
Generates synthetic interaction data from embeddings and similarity scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Any
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'notebooks'))
from utils.create_embeddings import load_marketing_embeddings, load_profiles_embeddings

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'baseline'
CF_RESULTS_DIR = PROJECT_ROOT / 'results' / 'collaborative_filtering'
CF_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class CollaborativeFiltering:
    """
    Collaborative Filtering using Matrix Factorization (NMF).
    Learns latent factors from user-user interaction matrix.
    """
    
    def __init__(self, n_components=50, random_state=42):
        """
        Initialize Collaborative Filtering model.
        
        Args:
            n_components: Number of latent factors
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, interaction_matrix: np.ndarray, user_ids: List, item_ids: List):
        """
        Fit the collaborative filtering model.
        
        Args:
            interaction_matrix: User-item interaction matrix (sparse, 0-1 or ratings)
            user_ids: List of user IDs corresponding to rows
            item_ids: List of item IDs corresponding to columns
        """
        print(f"Fitting Collaborative Filtering model with {self.n_components} components...")
        
        self.user_ids = user_ids
        self.item_ids = item_ids
        
        # Use Non-negative Matrix Factorization
        self.model = NMF(n_components=self.n_components, 
                        random_state=self.random_state,
                        max_iter=500,
                        alpha_W=0.01,
                        alpha_H=0.01)
        
        # Fit model
        self.user_factors = self.model.fit_transform(interaction_matrix)
        self.item_factors = self.model.components_.T
        
        print(f"Model fitted. User factors shape: {self.user_factors.shape}")
        print(f"Item factors shape: {self.item_factors.shape}")
    
    def predict(self, user_id: int, top_k: int = 10) -> List[int]:
        """
        Predict top-k recommendations for a user.
        
        Args:
            user_id: User ID to get recommendations for
            top_k: Number of recommendations
        
        Returns:
            List of recommended item IDs
        """
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.index(user_id)
        user_vector = self.user_factors[user_idx]
        
        # Predict scores for all items
        scores = np.dot(self.item_factors, user_vector)
        
        # Get top-k items (excluding self if user_id == item_id)
        top_indices = np.argsort(scores)[::-1]
        
        recommendations = []
        for idx in top_indices:
            item_id = self.item_ids[idx]
            if item_id != user_id:  # Don't recommend self
                recommendations.append(item_id)
            if len(recommendations) >= top_k:
                break
        
        return recommendations
    
    def get_user_similarity(self, user_id1: int, user_id2: int) -> float:
        """
        Get similarity between two users based on latent factors.
        
        Args:
            user_id1: First user ID
            user_id2: Second user ID
        
        Returns:
            Similarity score (0-1)
        """
        if user_id1 not in self.user_ids or user_id2 not in self.user_ids:
            return 0.0
        
        idx1 = self.user_ids.index(user_id1)
        idx2 = self.user_ids.index(user_id2)
        
        vec1 = self.user_factors[idx1]
        vec2 = self.user_factors[idx2]
        
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        return max(0.0, similarity)  # Ensure non-negative
    
    def save_model(self, filepath: Path):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_ids': self.user_ids,
            'item_ids': self.item_ids,
            'n_components': self.n_components
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Path):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(n_components=model_data['n_components'])
        instance.model = model_data['model']
        instance.user_factors = model_data['user_factors']
        instance.item_factors = model_data['item_factors']
        instance.user_ids = model_data['user_ids']
        instance.item_ids = model_data['item_ids']
        
        return instance


def generate_synthetic_interactions(df_students: pd.DataFrame,
                                   embeddings: np.ndarray,
                                   interaction_rate: float = 0.1,
                                   similarity_threshold: float = 0.6) -> np.ndarray:
    """
    Generate synthetic user-user interaction matrix based on embeddings.
    
    Args:
        df_students: DataFrame with student data
        embeddings: Embedding matrix for students
        interaction_rate: Proportion of possible interactions to generate (0-1)
        similarity_threshold: Minimum similarity to generate interaction
    
    Returns:
        Binary interaction matrix (n_users x n_users)
    """
    print("Generating synthetic interaction data...")
    
    n_students = len(df_students)
    student_ids = df_students['student_id'].values
    
    # Calculate similarity matrix
    print("  Computing similarity matrix...")
    similarities = cosine_similarity(embeddings)
    
    # Initialize interaction matrix (binary: 1 = interaction, 0 = no interaction)
    interaction_matrix = np.zeros((n_students, n_students), dtype=np.float32)
    
    # Generate interactions based on similarity
    print(f"  Generating interactions (rate={interaction_rate}, threshold={similarity_threshold})...")
    n_interactions = 0
    
    for i in range(n_students):
        for j in range(i + 1, n_students):
            similarity = similarities[i, j]
            
            # Higher similarity = higher probability of interaction
            if similarity >= similarity_threshold:
                # Probability increases with similarity
                prob = min(1.0, similarity * interaction_rate * 2)
                
                # Sample interaction
                if np.random.random() < prob:
                    interaction_matrix[i, j] = 1.0
                    interaction_matrix[j, i] = 1.0  # Symmetric
                    n_interactions += 1
    
    # Add some random interactions for diversity (cold-start handling)
    n_random = int(n_students * n_students * interaction_rate * 0.1)
    random_added = 0
    
    for _ in range(n_random):
        i, j = np.random.randint(0, n_students, size=2)
        if i != j and interaction_matrix[i, j] == 0:
            interaction_matrix[i, j] = 1.0
            interaction_matrix[j, i] = 1.0
            random_added += 1
    
    print(f"  Generated {n_interactions} similarity-based interactions")
    print(f"  Added {random_added} random interactions for diversity")
    print(f"  Total interactions: {np.sum(interaction_matrix) / 2:.0f}")
    print(f"  Interaction density: {np.sum(interaction_matrix) / (n_students * n_students):.4f}")
    
    return interaction_matrix


def create_hybrid_recommendations(cf_model: CollaborativeFiltering,
                                 embeddings: np.ndarray,
                                 student_ids: List[int],
                                 user_id: int,
                                 top_k: int = 10,
                                 cf_weight: float = 0.6,
                                 content_weight: float = 0.4) -> List[int]:
    """
    Create hybrid recommendations combining CF and content-based (embedding) approaches.
    
    Args:
        cf_model: Trained collaborative filtering model
        embeddings: Content-based embeddings
        student_ids: List of student IDs
        user_id: User to get recommendations for
        top_k: Number of recommendations
        cf_weight: Weight for CF scores
        content_weight: Weight for content-based scores
    
    Returns:
        List of recommended student IDs
    """
    # Get CF recommendations
    cf_recs = cf_model.predict(user_id, top_k=top_k * 2)
    cf_scores = {}
    for rec_id in cf_recs:
        cf_scores[rec_id] = cf_model.get_user_similarity(user_id, rec_id)
    
    # Get content-based recommendations
    if user_id in student_ids:
        user_idx = student_ids.index(user_id)
        user_embedding = embeddings[user_idx]
        
        content_similarities = cosine_similarity([user_embedding], embeddings)[0]
        content_scores = {}
        
        for i, sid in enumerate(student_ids):
            if sid != user_id:
                content_scores[sid] = content_similarities[i]
    else:
        content_scores = {}
    
    # Combine scores
    all_candidates = set(cf_scores.keys()) | set(content_scores.keys())
    combined_scores = {}
    
    for candidate_id in all_candidates:
        cf_score = cf_scores.get(candidate_id, 0.0)
        content_score = content_scores.get(candidate_id, 0.0)
        
        # Normalize scores to 0-1
        combined_score = (cf_weight * cf_score) + (content_weight * content_score)
        combined_scores[candidate_id] = combined_score
    
    # Get top-k
    sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = [sid for sid, _ in sorted_candidates[:top_k]]
    
    return recommendations


def evaluate_cf_model(cf_model: CollaborativeFiltering,
                     interaction_matrix: np.ndarray,
                     test_ratio: float = 0.2) -> Dict[str, float]:
    """
    Evaluate CF model using train/test split.
    
    Args:
        cf_model: Trained CF model
        interaction_matrix: Full interaction matrix
        test_ratio: Proportion of interactions to use for testing
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating CF model...")
    
    n_users, n_items = interaction_matrix.shape
    
    # Create train/test split
    test_mask = np.random.random((n_users, n_items)) < test_ratio
    train_matrix = interaction_matrix.copy()
    train_matrix[test_mask] = 0
    
    # Retrain on training set
    cf_model.fit(train_matrix, cf_model.user_ids, cf_model.item_ids)
    
    # Evaluate on test set
    test_interactions = interaction_matrix[test_mask]
    test_users, test_items = np.where(test_mask)
    
    # Predict and evaluate
    correct = 0
    total = len(test_users)
    
    for i in range(min(1000, total)):  # Sample for speed
        user_id = cf_model.user_ids[test_users[i]]
        item_id = cf_model.item_ids[test_items[i]]
        
        recs = cf_model.predict(user_id, top_k=20)
        if item_id in recs:
            correct += 1
    
    accuracy = correct / min(1000, total) if total > 0 else 0.0
    
    return {
        'test_accuracy': accuracy,
        'test_samples': min(1000, total)
    }


def implement_friend_finder_cf():
    """
    Implement Collaborative Filtering for Friend Finder.
    """
    print("\n" + "="*60)
    print("COLLABORATIVE FILTERING - FRIEND FINDER")
    print("="*60)
    
    print("\nLoading data...")
    df_marketing = pd.read_csv(PROCESSED_DATA_DIR / 'marketing_processed.csv')
    print(f"Loaded {len(df_marketing)} students")
    
    print("\nLoading embeddings...")
    embeddings = load_marketing_embeddings()
    if embeddings is None:
        print("Error: Could not load embeddings")
        return None
    
    print("\nGenerating synthetic interactions...")
    interaction_matrix = generate_synthetic_interactions(
        df_marketing, embeddings, interaction_rate=0.15, similarity_threshold=0.65
    )
    
    student_ids = df_marketing['student_id'].tolist()
    
    print("\nTraining Collaborative Filtering model...")
    cf_model = CollaborativeFiltering(n_components=50, random_state=42)
    cf_model.fit(interaction_matrix, student_ids, student_ids)
    
    print("\nEvaluating model...")
    eval_results = evaluate_cf_model(cf_model, interaction_matrix)
    print(f"Test Accuracy: {eval_results['test_accuracy']:.4f}")
    
    print("\nTesting recommendations...")
    test_students = df_marketing['student_id'].sample(n=min(5, len(df_marketing)), random_state=42).tolist()
    
    for student_id in test_students:
        recs = cf_model.predict(student_id, top_k=5)
        print(f"\nStudent {student_id} - CF Recommendations: {recs[:5]}")
        
        # Hybrid recommendations
        hybrid_recs = create_hybrid_recommendations(
            cf_model, embeddings, student_ids, student_id, top_k=5
        )
        print(f"Student {student_id} - Hybrid Recommendations: {hybrid_recs[:5]}")
    
    print("\nSaving model...")
    cf_model.save_model(CF_RESULTS_DIR / 'friend_finder_cf_model.pkl')
    
    # Save interaction matrix
    np.save(CF_RESULTS_DIR / 'friend_finder_interactions.npy', interaction_matrix)
    
    return cf_model, interaction_matrix


def implement_study_buddy_cf():
    """
    Implement Collaborative Filtering for Study Buddy.
    """
    print("\n" + "="*60)
    print("COLLABORATIVE FILTERING - STUDY BUDDY")
    print("="*60)
    
    print("\nLoading data...")
    df_profiles = pd.read_csv(PROCESSED_DATA_DIR / 'profiles_processed.csv')
    print(f"Loaded {len(df_profiles)} students")
    
    print("\nLoading embeddings...")
    embeddings = load_profiles_embeddings()
    if embeddings is None:
        print("Error: Could not load embeddings")
        return None
    
    print("\nGenerating synthetic interactions...")
    interaction_matrix = generate_synthetic_interactions(
        df_profiles, embeddings, interaction_rate=0.12, similarity_threshold=0.6
    )
    
    student_ids = df_profiles['student_id'].tolist()
    
    print("\nTraining Collaborative Filtering model...")
    cf_model = CollaborativeFiltering(n_components=50, random_state=42)
    cf_model.fit(interaction_matrix, student_ids, student_ids)
    
    print("\nEvaluating model...")
    eval_results = evaluate_cf_model(cf_model, interaction_matrix)
    print(f"Test Accuracy: {eval_results['test_accuracy']:.4f}")
    
    print("\nTesting recommendations...")
    test_students = df_profiles['student_id'].sample(n=min(5, len(df_profiles)), random_state=42).tolist()
    
    for student_id in test_students:
        recs = cf_model.predict(student_id, top_k=5)
        print(f"\nStudent {student_id} - CF Recommendations: {recs[:5]}")
        
        # Hybrid recommendations
        hybrid_recs = create_hybrid_recommendations(
            cf_model, embeddings, student_ids, student_id, top_k=5
        )
        print(f"Student {student_id} - Hybrid Recommendations: {hybrid_recs[:5]}")
    
    print("\nSaving model...")
    cf_model.save_model(CF_RESULTS_DIR / 'study_buddy_cf_model.pkl')
    
    # Save interaction matrix
    np.save(CF_RESULTS_DIR / 'study_buddy_interactions.npy', interaction_matrix)
    
    return cf_model, interaction_matrix


def create_cf_visualizations(ff_cf_model, sb_cf_model, 
                             ff_interactions, sb_interactions,
                             ff_student_ids, sb_student_ids):
    """
    Create multiple separate visualizations for CF models.
    """
    print("\nCreating visualizations...")
    
    # 1. Friend Finder - Interaction Matrix Heatmap
    print("  Creating Friend Finder interaction matrix heatmap...")
    sample_size = min(200, len(ff_interactions))
    plt.figure(figsize=(12, 10))
    plt.imshow(ff_interactions[:sample_size, :sample_size], cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Interaction')
    plt.title('Friend Finder - Interaction Matrix (Sample)', fontsize=14, fontweight='bold')
    plt.xlabel('Student ID', fontsize=12)
    plt.ylabel('Student ID', fontsize=12)
    plt.tight_layout()
    plt.savefig(CF_RESULTS_DIR / 'cf_friend_finder_interaction_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Study Buddy - Interaction Matrix Heatmap
    print("  Creating Study Buddy interaction matrix heatmap...")
    sample_size = min(200, len(sb_interactions))
    plt.figure(figsize=(12, 10))
    plt.imshow(sb_interactions[:sample_size, :sample_size], cmap='YlGnBu', aspect='auto')
    plt.colorbar(label='Interaction')
    plt.title('Study Buddy - Interaction Matrix (Sample)', fontsize=14, fontweight='bold')
    plt.xlabel('Student ID', fontsize=12)
    plt.ylabel('Student ID', fontsize=12)
    plt.tight_layout()
    plt.savefig(CF_RESULTS_DIR / 'cf_study_buddy_interaction_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Friend Finder - Interactions per Student Distribution
    print("  Creating Friend Finder interaction distribution...")
    ff_interaction_counts = np.sum(ff_interactions, axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(ff_interaction_counts, bins=50, alpha=0.7, edgecolor='black', color='orange')
    plt.title('Friend Finder - Interactions per Student', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Interactions', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axvline(np.mean(ff_interaction_counts), color='red', linestyle='--', 
                label=f'Mean: {np.mean(ff_interaction_counts):.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(CF_RESULTS_DIR / 'cf_friend_finder_interaction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Study Buddy - Interactions per Student Distribution
    print("  Creating Study Buddy interaction distribution...")
    sb_interaction_counts = np.sum(sb_interactions, axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(sb_interaction_counts, bins=50, alpha=0.7, edgecolor='black', color='green')
    plt.title('Study Buddy - Interactions per Student', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Interactions', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axvline(np.mean(sb_interaction_counts), color='red', linestyle='--', 
                label=f'Mean: {np.mean(sb_interaction_counts):.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(CF_RESULTS_DIR / 'cf_study_buddy_interaction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Friend Finder - Latent Factor Analysis (Top Factors)
    print("  Creating Friend Finder latent factors visualization...")
    if ff_cf_model.user_factors is not None:
        n_factors = min(10, ff_cf_model.user_factors.shape[1])
        factor_importance = np.mean(np.abs(ff_cf_model.user_factors), axis=0)
        top_factors = np.argsort(factor_importance)[-n_factors:][::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_factors), factor_importance[top_factors], alpha=0.7, color='purple')
        plt.title('Friend Finder - Top Latent Factors Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Factor Index', fontsize=12)
        plt.ylabel('Average Absolute Value', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(CF_RESULTS_DIR / 'cf_friend_finder_latent_factors.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Study Buddy - Latent Factor Analysis (Top Factors)
    print("  Creating Study Buddy latent factors visualization...")
    if sb_cf_model.user_factors is not None:
        n_factors = min(10, sb_cf_model.user_factors.shape[1])
        factor_importance = np.mean(np.abs(sb_cf_model.user_factors), axis=0)
        top_factors = np.argsort(factor_importance)[-n_factors:][::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_factors), factor_importance[top_factors], alpha=0.7, color='teal')
        plt.title('Study Buddy - Top Latent Factors Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Factor Index', fontsize=12)
        plt.ylabel('Average Absolute Value', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(CF_RESULTS_DIR / 'cf_study_buddy_latent_factors.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Interaction Density Comparison
    print("  Creating interaction density comparison...")
    ff_density = np.sum(ff_interactions) / (len(ff_interactions) * len(ff_interactions))
    sb_density = np.sum(sb_interactions) / (len(sb_interactions) * len(sb_interactions))
    
    plt.figure(figsize=(8, 6))
    models = ['Friend Finder', 'Study Buddy']
    densities = [ff_density, sb_density]
    colors = ['orange', 'green']
    plt.bar(models, densities, alpha=0.7, color=colors, edgecolor='black')
    plt.title('Interaction Matrix Density Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Density (Interactions / Total Possible)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    for i, (model, density) in enumerate(zip(models, densities)):
        plt.text(i, density + 0.001, f'{density:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(CF_RESULTS_DIR / 'cf_interaction_density_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. User Clustering Based on Latent Factors (t-SNE)
    print("  Creating user clustering visualization...")
    try:
        from sklearn.manifold import TSNE
        
        # Friend Finder clustering
        if ff_cf_model.user_factors is not None and len(ff_cf_model.user_factors) > 100:
            sample_size = min(1000, len(ff_cf_model.user_factors))
            sample_indices = np.random.choice(len(ff_cf_model.user_factors), sample_size, replace=False)
            sample_factors = ff_cf_model.user_factors[sample_indices]
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            factors_2d = tsne.fit_transform(sample_factors)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(factors_2d[:, 0], factors_2d[:, 1], alpha=0.6, s=20, c='orange')
            plt.title('Friend Finder - User Clustering (t-SNE of Latent Factors)', fontsize=14, fontweight='bold')
            plt.xlabel('t-SNE Component 1', fontsize=12)
            plt.ylabel('t-SNE Component 2', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(CF_RESULTS_DIR / 'cf_friend_finder_user_clustering.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Study Buddy clustering
        if sb_cf_model.user_factors is not None and len(sb_cf_model.user_factors) > 100:
            sample_size = min(1000, len(sb_cf_model.user_factors))
            sample_indices = np.random.choice(len(sb_cf_model.user_factors), sample_size, replace=False)
            sample_factors = sb_cf_model.user_factors[sample_indices]
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            factors_2d = tsne.fit_transform(sample_factors)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(factors_2d[:, 0], factors_2d[:, 1], alpha=0.6, s=20, c='green')
            plt.title('Study Buddy - User Clustering (t-SNE of Latent Factors)', fontsize=14, fontweight='bold')
            plt.xlabel('t-SNE Component 1', fontsize=12)
            plt.ylabel('t-SNE Component 2', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(CF_RESULTS_DIR / 'cf_study_buddy_user_clustering.png', dpi=300, bbox_inches='tight')
            plt.close()
    except ImportError:
        print("  Skipping t-SNE visualization (sklearn.manifold not available)")
    
    # 9. Interaction Statistics Summary
    print("  Creating interaction statistics summary...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Friend Finder stats
    ff_stats = {
        'Total Interactions': np.sum(ff_interactions) / 2,
        'Avg per Student': np.mean(ff_interaction_counts),
        'Max per Student': np.max(ff_interaction_counts),
        'Min per Student': np.min(ff_interaction_counts),
        'Density': ff_density
    }
    
    axes[0].barh(list(ff_stats.keys()), list(ff_stats.values()), alpha=0.7, color='orange')
    axes[0].set_title('Friend Finder - Interaction Statistics', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Value', fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Study Buddy stats
    sb_stats = {
        'Total Interactions': np.sum(sb_interactions) / 2,
        'Avg per Student': np.mean(sb_interaction_counts),
        'Max per Student': np.max(sb_interaction_counts),
        'Min per Student': np.min(sb_interaction_counts),
        'Density': sb_density
    }
    
    axes[1].barh(list(sb_stats.keys()), list(sb_stats.values()), alpha=0.7, color='green')
    axes[1].set_title('Study Buddy - Interaction Statistics', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Value', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(CF_RESULTS_DIR / 'cf_interaction_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  All visualizations saved to: {CF_RESULTS_DIR}")


def main():
    """
    Main function to implement Collaborative Filtering.
    """
    print("="*60)
    print("COLLABORATIVE FILTERING IMPLEMENTATION")
    print("="*60)
    
    # Load data for student IDs
    df_marketing = pd.read_csv(PROCESSED_DATA_DIR / 'marketing_processed.csv')
    df_profiles = pd.read_csv(PROCESSED_DATA_DIR / 'profiles_processed.csv')
    
    # Friend Finder CF
    ff_cf_model, ff_interactions = implement_friend_finder_cf()
    
    # Study Buddy CF
    sb_cf_model, sb_interactions = implement_study_buddy_cf()
    
    if ff_cf_model is not None and sb_cf_model is not None:
        # Create visualizations
        ff_student_ids = df_marketing['student_id'].tolist()
        sb_student_ids = df_profiles['student_id'].tolist()
        create_cf_visualizations(ff_cf_model, sb_cf_model, ff_interactions, sb_interactions,
                                ff_student_ids, sb_student_ids)
        
        print("\n" + "="*60)
        print("COLLABORATIVE FILTERING COMPLETED!")
        print("="*60)
        print(f"Results saved to: {CF_RESULTS_DIR}")
        print("Models ready for hybrid recommendations!")


if __name__ == "__main__":
    main()

