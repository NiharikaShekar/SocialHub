#!/usr/bin/env python3
"""
Friend Finder Baseline Model (Embedding-Based)
This script implements the Friend Finder model using K-Means clustering on embeddings.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'notebooks'))
from utils.create_embeddings import create_marketing_embeddings, load_marketing_embeddings

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'baseline'

print(f" FRIEND FINDER BASELINE MODEL (EMBEDDING-BASED)")
print(f"Processed data directory: {PROCESSED_DATA_DIR}")

class FriendFinderBaseline:
    """
    Friend Finder Baseline Model using K-Means clustering on embeddings.
    Groups students based on semantic embeddings for friend recommendations.
    """
    
    def __init__(self, n_clusters=8, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.embeddings = None
        self.student_ids = None
        self.labels = None
        self.df_students = None
    
    def load_or_create_embeddings(self, df_students, force_recreate=False):
        """
        Load existing embeddings or create new ones.
        """
        if not force_recreate:
            # Try to load existing embeddings
            embeddings = load_marketing_embeddings()
            if embeddings is not None and len(embeddings) == len(df_students):
                print("    Loaded existing embeddings")
                return embeddings
        
        # Create new embeddings
        print("   • Creating new embeddings...")
        embeddings = create_marketing_embeddings(df_students, save=True)
        return embeddings
    
    def train(self, df_students, force_recreate_embeddings=False):
        """
        Train the K-Means clustering model on embeddings.
        """
        print(f"\n Training Friend Finder model (Embedding-Based)...")
        
        self.df_students = df_students.copy()
        self.student_ids = df_students['student_id']
        
        # Load or create embeddings
        print(" Loading or creating embeddings...")
        self.embeddings = self.load_or_create_embeddings(df_students, force_recreate_embeddings)
        
        print(f"   • Embedding shape: {self.embeddings.shape}")
        print(f"   • Each student represented by {self.embeddings.shape[1]}-dimensional embedding")
        
        # Train K-Means on embeddings
        print(f"   • Training K-Means with {self.n_clusters} clusters on embeddings...")
        print(f"   • Using ALL {self.embeddings.shape[1]} embedding dimensions for clustering")
        print(f"   • Each dimension contributes to similarity calculation")
        
        # K-Means uses ALL dimensions:
        # - Calculates distance in ALL 386 dimensions
        # - Example: Student A = [0.4, -0.2, ..., 0.3] (386 dims)
        # - Example: Student B = [0.5, -0.1, ..., 0.2] (386 dims)
        # - Distance = √[(0.4-0.5)² + (-0.2-(-0.1))² + ... + (0.3-0.2)²]
        # - ALL dimensions contribute to this distance!
        
        self.kmeans.fit(self.embeddings)
        self.labels = self.kmeans.labels_
        
        # Add cluster labels to dataframe
        self.df_students['cluster'] = self.labels
        
        # Analyze clusters
        self.analyze_clusters()
        
        print(f" Friend Finder model trained successfully!")
        print(f"   • {len(self.student_ids)} students clustered into {self.n_clusters} groups")
        print(f"   • Using semantic embeddings for better recommendations")
        
        # Save the trained model
        self.save_model()
    
    def analyze_clusters(self):
        """
        Analyze the clusters to understand what each group represents.
        """
        print(f"\n Analyzing clusters...")
        
        # Analyze each cluster
        for cluster_id in range(self.n_clusters):
            cluster_data = self.df_students[self.df_students['cluster'] == cluster_id]
            
            print(f"\n   Cluster {cluster_id} ({len(cluster_data)} students):")
            print(f"   • Average age: {cluster_data['age'].mean():.1f}")
            print(f"   • Average friends: {cluster_data['NumberOffriends'].mean():.1f}")
            
            # Find top interests for this cluster (from original data)
            interest_cols = [col for col in cluster_data.columns 
                           if col not in ['gradyear', 'gender', 'age', 'NumberOffriends', 'student_id', 'cluster']]
            
            if interest_cols:
                top_interests = cluster_data[interest_cols].mean().sort_values(ascending=False).head(5)
                print(f"   • Top interests: {', '.join([f'{k}({v:.2f})' for k, v in top_interests.items()])}")
    
    def recommend_friends(self, student_id, top_k=5):
        """
        Recommend friends for a given student using embedding similarity.
        """
        if student_id not in self.student_ids.values:
            print(f" Student ID {student_id} not found.")
            return []
        
        # Find student's cluster
        student_idx = self.student_ids[self.student_ids == student_id].index[0]
        student_cluster = self.labels[student_idx]
        student_embedding = self.embeddings[student_idx].reshape(1, -1)
        
        # Find other students in the same cluster
        same_cluster_mask = self.labels == student_cluster
        same_cluster_indices = np.where(same_cluster_mask)[0]
        same_cluster_indices = [idx for idx in same_cluster_indices if idx != student_idx]
        
        if len(same_cluster_indices) == 0:
            print(f" No other students found in cluster {student_cluster}")
            return []
        
        # Calculate cosine similarity within cluster using embeddings
        cluster_embeddings = self.embeddings[same_cluster_indices]
        similarities = cosine_similarity(student_embedding, cluster_embeddings)[0]
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:top_k]
        recommended_indices = [same_cluster_indices[i] for i in top_indices]
        recommended_ids = self.student_ids.iloc[recommended_indices].tolist()
        
        return recommended_ids
    
    def save_model(self):
        """
        Save the trained model to disk for later use.
        """
        friend_finder_dir = RESULTS_DIR / 'friend_finder_results'
        friend_finder_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_data = {
            'kmeans': self.kmeans,
            'embeddings': self.embeddings,
            'student_ids': self.student_ids,
            'labels': self.labels,
            'df_students': self.df_students,
            'n_clusters': self.n_clusters
        }
        
        model_path = friend_finder_dir / 'friend_finder_model_embedding.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f" Model saved to: {model_path}")
    
    @classmethod
    def load_model(cls, model_path):
        """
        Load a trained model from disk.
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        instance = cls(n_clusters=model_data['n_clusters'])
        
        # Restore model state
        instance.kmeans = model_data['kmeans']
        instance.embeddings = model_data['embeddings']
        instance.student_ids = model_data['student_ids']
        instance.labels = model_data['labels']
        instance.df_students = model_data['df_students']
        
        return instance
    
    def get_recommendation_details(self, student_id, top_k=5):
        """
        Get detailed friend recommendations with explanations.
        """
        recommendations = self.recommend_friends(student_id, top_k)
        
        if not recommendations:
            return None
        
        # Get student info
        student_info = self.df_students[self.df_students['student_id'] == student_id].iloc[0]
        student_cluster = student_info['cluster']
        student_idx = self.student_ids[self.student_ids == student_id].index[0]
        student_embedding = self.embeddings[student_idx]
        
        print(f"\n Student {student_id} (Cluster {student_cluster}):")
        print(f"   • Age: {student_info['age']:.1f}")
        print(f"   • Friends: {student_info['NumberOffriends']}")
        print(f"   • Embedding-based recommendation")
        
        # Show top interests
        interest_cols = [col for col in student_info.index 
                        if col not in ['gradyear', 'gender', 'age', 'NumberOffriends', 'student_id', 'cluster']]
        top_interests = student_info[interest_cols].sort_values(ascending=False).head(5)
        active_interests = [f'{k}({v})' for k, v in top_interests.items() if v > 0]
        if active_interests:
            print(f"   • Top interests: {', '.join(active_interests[:3])}")
        
        print(f"\n Recommended Friends (Embedding Similarity):")
        
        # Get recommendation details
        same_cluster_mask = self.labels == student_cluster
        same_cluster_indices = np.where(same_cluster_mask)[0]
        same_cluster_indices = [idx for idx in same_cluster_indices if idx != student_idx]
        
        cluster_embeddings = self.embeddings[same_cluster_indices]
        similarities = cosine_similarity([student_embedding], cluster_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        for i, rec_idx in enumerate(top_indices, 1):
            actual_idx = same_cluster_indices[rec_idx]
            rec_id = self.student_ids.iloc[actual_idx]
            rec_info = self.df_students[self.df_students['student_id'] == rec_id].iloc[0]
            similarity = similarities[rec_idx]
            
            print(f"   {i}. Student {rec_id}: Age {rec_info['age']:.1f}, Friends {rec_info['NumberOffriends']}, Similarity: {similarity:.3f}")
            
            # Show common interests
            rec_interests = rec_info[interest_cols]
            common_interests = []
            for interest in interest_cols:
                if student_info[interest] > 0 and rec_info[interest] > 0:
                    common_interests.append(interest)
            
            if common_interests:
                print(f"      Common interests: {', '.join(common_interests[:3])}")
        
        return recommendations

def create_cluster_visualization(model):
    """
    Create visualization of the clusters.
    """
    print(f"\n Creating cluster visualization...")
    
    # Use PCA for visualization
    from sklearn.decomposition import PCA
    
    # Reduce embeddings to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(model.embeddings)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Friend Finder - Embedding-Based Cluster Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cluster visualization (2D PCA)
    scatter = axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=model.df_students['cluster'], cmap='tab10', alpha=0.6, s=20)
    axes[0, 0].set_title('Student Embeddings (PCA 2D) Colored by Cluster')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cluster distribution
    cluster_counts = model.df_students['cluster'].value_counts().sort_index()
    axes[0, 1].bar(cluster_counts.index, cluster_counts.values, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Students per Cluster')
    axes[0, 1].set_xlabel('Cluster ID')
    axes[0, 1].set_ylabel('Number of Students')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Average age per cluster
    cluster_ages = model.df_students.groupby('cluster')['age'].mean()
    axes[1, 0].bar(cluster_ages.index, cluster_ages.values, alpha=0.7, edgecolor='black', color='skyblue')
    axes[1, 0].set_title('Average Age per Cluster')
    axes[1, 0].set_xlabel('Cluster ID')
    axes[1, 0].set_ylabel('Average Age')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Average friends per cluster
    cluster_friends = model.df_students.groupby('cluster')['NumberOffriends'].mean()
    axes[1, 1].bar(cluster_friends.index, cluster_friends.values, alpha=0.7, edgecolor='black', color='lightgreen')
    axes[1, 1].set_title('Average Friends per Cluster')
    axes[1, 1].set_xlabel('Cluster ID')
    axes[1, 1].set_ylabel('Average Number of Friends')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    friend_finder_dir = RESULTS_DIR / 'friend_finder_results'
    friend_finder_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(friend_finder_dir / 'friend_finder_clusters_embedding.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Visualization saved to: {friend_finder_dir / 'friend_finder_clusters_embedding.png'}")

def load_optimal_k(default_k=8):
    """
    Load optimal k value from saved file, or use default.
    
    Args:
        default_k: Default number of clusters if optimal k not found
    
    Returns:
        Optimal k value
    """
    optimal_k_file = RESULTS_DIR / 'optimal_k_friend_finder.pkl'
    
    if optimal_k_file.exists():
        try:
            with open(optimal_k_file, 'rb') as f:
                optimal_k = pickle.load(f)
            print(f"    Loaded optimal k = {optimal_k} from previous analysis")
            return optimal_k
        except:
            pass
    
    print(f"   • Using default k = {default_k} (run 09_find_optimal_clusters.py to find optimal)")
    return default_k

def test_friend_finder():
    """
    Test the Friend Finder model with sample students.
    """
    print(f"\n TESTING FRIEND FINDER MODEL (EMBEDDING-BASED)")
    print("="*50)
    
    # Load data
    print(" Loading marketing dataset...")
    df_marketing = pd.read_csv(PROCESSED_DATA_DIR / 'marketing_processed.csv')
    print(f"   • Loaded {len(df_marketing)} students")
    
    # Load optimal k or use default
    optimal_k = load_optimal_k(default_k=8)
    
    # Initialize and train model
    friend_finder = FriendFinderBaseline(n_clusters=optimal_k)
    friend_finder.train(df_marketing, force_recreate_embeddings=False)
    
    # Test with sample students
    print(f"\n Testing recommendations...")
    
    # Test with different students
    test_students = [1, 100, 500, 1000, 2000]
    
    for student_id in test_students:
        if student_id in df_marketing['student_id'].values:
            print(f"\n{'='*60}")
            friend_finder.get_recommendation_details(student_id, top_k=3)
        else:
            print(f" Student {student_id} not found in dataset")
    
    # Create visualization
    create_cluster_visualization(friend_finder)
    
    print(f"\n Friend Finder testing completed!")

def main():
    """
    Main function to run Friend Finder baseline model.
    """
    print(" STARTING FRIEND FINDER BASELINE MODEL (EMBEDDING-BASED)")
    print("="*60)
    
    # Test the model
    test_friend_finder()
    
    print("\n" + "="*60)
    print(" FRIEND FINDER BASELINE MODEL COMPLETED!")
    print("="*60)
    print(" Model ready for friend recommendations using embeddings!")
    print(" Cluster analysis and visualizations created!")
    print(" Model saved for future use!")

if __name__ == "__main__":
    main()

