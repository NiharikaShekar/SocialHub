#!/usr/bin/env python3
"""
Find Optimal Number of Clusters
This script helps determine the optimal number of clusters using the Elbow Method and Silhouette Analysis.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'notebooks'))
from utils.create_embeddings import load_marketing_embeddings, load_profiles_embeddings

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'baseline'

def find_optimal_clusters(embeddings, max_clusters=20, dataset_name="Dataset"):
    """
    Find optimal number of clusters using Elbow Method and Silhouette Score.
    
    Args:
        embeddings: The embedding matrix (n_samples, n_features)
        max_clusters: Maximum number of clusters to test
        dataset_name: Name of the dataset for display
    
    Returns:
        Optimal number of clusters
    """
    print(f"\n Finding Optimal Clusters for {dataset_name}")
    print("="*60)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Testing clusters from 2 to {max_clusters}...")
    
    # Sample data if too large (for faster computation)
    if len(embeddings) > 5000:
        print(f"   • Sampling {5000} students for faster computation...")
        sample_indices = np.random.choice(len(embeddings), 5000, replace=False)
        embeddings_sample = embeddings[sample_indices]
    else:
        embeddings_sample = embeddings
    
    # Calculate metrics for different numbers of clusters
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    print("\n   Computing metrics for each k...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_sample)
        
        inertia = kmeans.inertia_
        silhouette = silhouette_score(embeddings_sample, labels)
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        
        print(f"   k={k:2d}: Inertia={inertia:12.2f}, Silhouette={silhouette:.3f}")
    
    # Find optimal k using Elbow Method
    # Calculate rate of change (second derivative)
    inertia_changes = np.diff(inertias)
    inertia_changes_2 = np.diff(inertia_changes)
    
    # Elbow is where the rate of change slows down significantly
    # Find the point with maximum second derivative (steepest drop)
    elbow_k = np.argmax(np.abs(inertia_changes_2)) + 3  # +3 because of double diff
    
    # Find optimal k using Silhouette Score (highest score)
    optimal_silhouette_k = k_range[np.argmax(silhouette_scores)]
    
    print(f"\n Results:")
    print(f"   • Elbow Method suggests: k = {elbow_k}")
    print(f"   • Silhouette Score suggests: k = {optimal_silhouette_k}")
    print(f"   • Recommended: k = {optimal_silhouette_k} (highest silhouette score)")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Elbow Method
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow at k={elbow_k}')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    ax1.set_title(f'Elbow Method for {dataset_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Silhouette Score
    ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_silhouette_k, color='r', linestyle='--', 
                label=f'Optimal k={optimal_silhouette_k}')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title(f'Silhouette Analysis for {dataset_name}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    viz_filename = f'optimal_clusters_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(RESULTS_DIR / viz_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"    Visualization saved to: {RESULTS_DIR / viz_filename}")
    
    # Save optimal k value for Friend Finder
    if "Friend Finder" in dataset_name:
        optimal_k_path = RESULTS_DIR / 'optimal_k_friend_finder.pkl'
        with open(optimal_k_path, 'wb') as f:
            pickle.dump(optimal_silhouette_k, f)
        print(f"    Optimal k saved to: {optimal_k_path}")
        print(f"    Friend Finder will use k = {optimal_silhouette_k} in next training")
    
    return optimal_silhouette_k, elbow_k

def explain_clustering():
    """
    Explain how clustering works with embeddings.
    """
    print("\n" + "="*60)
    print(" HOW CLUSTERING WORKS WITH EMBEDDINGS")
    print("="*60)
    
    print("\n1. WHAT ARE EMBEDDINGS?")
    print("   • Each student is represented by a vector of numbers")
    print("   • Example: Student A = [0.4, -0.2, 0.6, ..., 0.3] (386 dimensions)")
    print("   • These numbers capture: interests, demographics, behaviors")
    
    print("\n2. HOW DO WE USE ALL EMBEDDING INFO?")
    print("   • YES! We use ALL dimensions in the embedding")
    print("   • Example embedding shape: (15000 students, 386 dimensions)")
    print("   • Each dimension contributes to similarity calculation")
    
    print("\n3. HOW K-MEANS CLUSTERING WORKS:")
    print("   Step 1: Initialize k cluster centers randomly")
    print("   Step 2: Assign each student to nearest cluster (using ALL dimensions)")
    print("   Step 3: Recalculate cluster centers (average of all students in cluster)")
    print("   Step 4: Repeat until clusters stabilize")
    
    print("\n4. DISTANCE CALCULATION (Uses ALL embedding dimensions):")
    print("   • Euclidean Distance: √[(x₁-y₁)² + (x₂-y₂)² + ... + (x₃₈₆-y₃₈₆)²]")
    print("   • Each dimension (feature) contributes to the distance")
    print("   • Students with similar embeddings (in ALL dimensions) → Same cluster")
    
    print("\n5. EXAMPLE:")
    print("   Student A embedding: [0.4, -0.2, 0.6, 0.1, ...]")
    print("   Student B embedding: [0.5, -0.1, 0.7, 0.2, ...]")
    print("   Student C embedding: [0.1, 0.8, -0.3, 0.9, ...]")
    print("   ")
    print("   Distance A-B: Small (similar in ALL dimensions) → Same cluster")
    print("   Distance A-C: Large (different in ALL dimensions) → Different cluster")

def main():
    """
    Main function to find optimal clusters.
    """
    print(" FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("="*60)
    
    # Explain how clustering works
    explain_clustering()
    
    # Load embeddings
    print("\n Loading embeddings...")
    marketing_embeddings = load_marketing_embeddings()
    profiles_embeddings = load_profiles_embeddings()
    
    if marketing_embeddings is None or profiles_embeddings is None:
        print("\n Embeddings not found. Please create embeddings first:")
        print("   Run: python notebooks/utils/create_embeddings.py")
        return
    
    # Find optimal clusters for Friend Finder
    print("\n" + "="*60)
    ff_optimal, ff_elbow = find_optimal_clusters(
        marketing_embeddings, 
        max_clusters=15,
        dataset_name="Friend Finder (Marketing)"
    )
    
    # Find optimal clusters for Study Buddy (optional, if using clustering)
    print("\n" + "="*60)
    print("\n Note: Study Buddy uses similarity matching, not clustering.")
    print("   If you want to use clustering, optimal k would be:", end=" ")
    sb_optimal, sb_elbow = find_optimal_clusters(
        profiles_embeddings,
        max_clusters=15,
        dataset_name="Study Buddy (Profiles)"
    )
    
    print("\n" + "="*60)
    print(" OPTIMAL CLUSTER ANALYSIS COMPLETED!")
    print("="*60)
    print(f"\n RECOMMENDATIONS:")
    print(f"   • Friend Finder: Use k = {ff_optimal} clusters")
    print(f"   • Study Buddy: Uses similarity matching (no clustering needed)")
    print(f"\n You can update the models with these optimal values!")

if __name__ == "__main__":
    main()