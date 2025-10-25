#!/usr/bin/env python3
"""
Friend Finder Baseline Model
This script implements and tests the Friend Finder model using K-Means clustering on the marketing dataset.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

print(f"👥 FRIEND FINDER BASELINE MODEL")
print(f"Processed data directory: {PROCESSED_DATA_DIR}")

class FriendFinderBaseline:
    """
    Friend Finder Baseline Model using K-Means clustering.
    Groups students based on interests and demographics for friend recommendations.
    """
    
    def __init__(self, n_clusters=8, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.student_features = None
        self.student_ids = None
        self.labels = None
        self.df_students = None
        
    def prepare_features(self, df):
        """
        Prepare features for clustering.
        """
        print("🔧 Preparing features for clustering...")
        
        # Select features for clustering
        feature_cols = ['age', 'NumberOffriends']
        
        # Add interest features (exclude demographic columns)
        interest_cols = [col for col in df.columns 
                        if col not in ['gradyear', 'gender', 'age', 'NumberOffriends', 'student_id']]
        
        # Combine all features
        all_features = feature_cols + interest_cols
        self.feature_cols = all_features
        
        print(f"   • Using {len(all_features)} features for clustering")
        print(f"   • Demographics: {feature_cols}")
        print(f"   • Interests: {len(interest_cols)} keywords")
        
        return df[all_features]
    
    def train(self, df_students):
        """
        Train the K-Means clustering model.
        """
        print(f"\n🎯 Training Friend Finder model...")
        
        self.df_students = df_students.copy()
        self.student_ids = df_students['student_id']
        
        # Prepare features
        self.student_features = self.prepare_features(df_students)
        
        # Handle missing values
        self.student_features = self.student_features.fillna(0)
        
        # Scale features
        print("   • Scaling features...")
        scaled_features = self.scaler.fit_transform(self.student_features)
        
        # Train K-Means
        print(f"   • Training K-Means with {self.n_clusters} clusters...")
        self.kmeans.fit(scaled_features)
        self.labels = self.kmeans.labels_
        
        # Analyze clusters
        self.analyze_clusters()
        
        print(f"✅ Friend Finder model trained successfully!")
        print(f"   • {len(self.student_ids)} students clustered into {self.n_clusters} groups")
        
        # Save the trained model
        self.save_model()
    
    def analyze_clusters(self):
        """
        Analyze the clusters to understand what each group represents.
        """
        print(f"\n📊 Analyzing clusters...")
        
        # Add cluster labels to dataframe
        self.df_students['cluster'] = self.labels
        
        # Analyze each cluster
        for cluster_id in range(self.n_clusters):
            cluster_data = self.df_students[self.df_students['cluster'] == cluster_id]
            
            print(f"\n   Cluster {cluster_id} ({len(cluster_data)} students):")
            print(f"   • Average age: {cluster_data['age'].mean():.1f}")
            print(f"   • Average friends: {cluster_data['NumberOffriends'].mean():.1f}")
            
            # Find top interests for this cluster
            interest_cols = [col for col in cluster_data.columns 
                           if col not in ['gradyear', 'gender', 'age', 'NumberOffriends', 'student_id', 'cluster']]
            
            top_interests = cluster_data[interest_cols].mean().sort_values(ascending=False).head(5)
            print(f"   • Top interests: {', '.join([f'{k}({v:.2f})' for k, v in top_interests.items()])}")
    
    def recommend_friends(self, student_id, top_k=5):
        """
        Recommend friends for a given student.
        """
        if student_id not in self.student_ids.values:
            print(f"❌ Student ID {student_id} not found.")
            return []
        
        # Find student's cluster
        student_idx = self.student_ids[self.student_ids == student_id].index[0]
        student_cluster = self.labels[student_idx]
        
        # Find other students in the same cluster
        same_cluster_students = self.df_students[self.df_students['cluster'] == student_cluster]
        same_cluster_students = same_cluster_students[same_cluster_students['student_id'] != student_id]
        
        if len(same_cluster_students) == 0:
            print(f"❌ No other students found in cluster {student_cluster}")
            return []
        
        # Calculate similarity within cluster
        student_features = self.student_features.iloc[student_idx].values.reshape(1, -1)
        cluster_features = self.student_features.iloc[same_cluster_students.index].values
        
        # Calculate cosine similarity
        similarities = cosine_similarity(student_features, cluster_features)[0]
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:top_k]
        recommended_ids = same_cluster_students.iloc[top_indices]['student_id'].tolist()
        
        return recommended_ids
    
    def save_model(self):
        """
        Save the trained model to disk for later use.
        """
        results_dir = PROJECT_ROOT / 'results' / 'baseline'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'student_features': self.student_features,
            'student_ids': self.student_ids,
            'labels': self.labels,
            'df_students': self.df_students,
            'feature_cols': self.feature_cols,
            'n_clusters': self.n_clusters
        }
        
        model_path = results_dir / 'friend_finder_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to: {model_path}")
    
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
        instance.scaler = model_data['scaler']
        instance.student_features = model_data['student_features']
        instance.student_ids = model_data['student_ids']
        instance.labels = model_data['labels']
        instance.df_students = model_data['df_students']
        instance.feature_cols = model_data['feature_cols']
        
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
        
        print(f"\n👤 Student {student_id} (Cluster {student_cluster}):")
        print(f"   • Age: {student_info['age']:.1f}")
        print(f"   • Friends: {student_info['NumberOffriends']}")
        
        # Show top interests
        interest_cols = [col for col in student_info.index 
                        if col not in ['gradyear', 'gender', 'age', 'NumberOffriends', 'student_id', 'cluster']]
        top_interests = student_info[interest_cols].sort_values(ascending=False).head(5)
        print(f"   • Top interests: {', '.join([f'{k}({v})' for k, v in top_interests.items() if v > 0])}")
        
        print(f"\n👥 Recommended Friends:")
        
        # Get recommendation details
        for i, rec_id in enumerate(recommendations, 1):
            rec_info = self.df_students[self.df_students['student_id'] == rec_id].iloc[0]
            print(f"   {i}. Student {rec_id}: Age {rec_info['age']:.1f}, Friends {rec_info['NumberOffriends']}")
            
            # Show common interests
            rec_interests = rec_info[interest_cols]
            common_interests = []
            for interest in interest_cols:
                if student_info[interest] > 0 and rec_info[interest] > 0:
                    common_interests.append(interest)
            
            if common_interests:
                print(f"      Common interests: {', '.join(common_interests[:3])}")
        
        return recommendations

def test_friend_finder():
    """
    Test the Friend Finder model with sample students.
    """
    print(f"\n🧪 TESTING FRIEND FINDER MODEL")
    print("="*50)
    
    # Load data
    print("📁 Loading marketing dataset...")
    df_marketing = pd.read_csv(PROCESSED_DATA_DIR / 'marketing_processed.csv')
    print(f"   • Loaded {len(df_marketing)} students")
    
    # Initialize and train model
    friend_finder = FriendFinderBaseline(n_clusters=8)
    friend_finder.train(df_marketing)
    
    # Test with sample students
    print(f"\n🎯 Testing recommendations...")
    
    # Test with different students
    test_students = [1, 100, 500, 1000, 2000]
    
    for student_id in test_students:
        if student_id in df_marketing['student_id'].values:
            print(f"\n{'='*60}")
            friend_finder.get_recommendation_details(student_id, top_k=3)
        else:
            print(f"❌ Student {student_id} not found in dataset")
    
    # Create visualization
    create_cluster_visualization(friend_finder)
    
    print(f"\n✅ Friend Finder testing completed!")

def create_cluster_visualization(model):
    """
    Create visualization of the clusters.
    """
    print(f"\n📊 Creating cluster visualization...")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Friend Finder - Cluster Analysis', fontsize=16, fontweight='bold')
    
    # 1. Age vs Friends colored by cluster
    scatter = axes[0, 0].scatter(model.df_students['age'], model.df_students['NumberOffriends'], 
                               c=model.df_students['cluster'], cmap='tab10', alpha=0.6)
    axes[0, 0].set_title('Age vs Friends (Colored by Cluster)')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Number of Friends')
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
    # Save to results folder
    results_dir = PROJECT_ROOT / 'results' / 'baseline'
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / 'friend_finder_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved to: {results_dir / 'friend_finder_clusters.png'}")

def main():
    """
    Main function to run Friend Finder baseline model.
    """
    print("🚀 STARTING FRIEND FINDER BASELINE MODEL")
    print("="*60)
    
    # Test the model
    test_friend_finder()
    
    print("\n" + "="*60)
    print("✅ FRIEND FINDER BASELINE MODEL COMPLETED!")
    print("="*60)
    print("🎯 Model ready for friend recommendations!")
    print("📊 Cluster analysis and visualizations created!")

if __name__ == "__main__":
    main()
