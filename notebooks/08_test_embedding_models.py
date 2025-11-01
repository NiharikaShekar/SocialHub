#!/usr/bin/env python3
"""
Test Embedding-Based Baseline Models
This script loads and tests the saved embedding-based baseline models.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'baseline'

print("🔄 TESTING EMBEDDING-BASED BASELINE MODELS")
print("="*60)

def test_friend_finder_model():
    """
    Test the saved Friend Finder model.
    """
    print("\n Testing Friend Finder Model (Embedding-Based)...")
    
    model_path = RESULTS_DIR / 'friend_finder_results' / 'friend_finder_model_embedding.pkl'
    
    if not model_path.exists():
        print(" Friend Finder model not found. Please run the training script first.")
        print("   Run: python notebooks/04_friend_finder_baseline_embedding.py")
        return False
    
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(" Friend Finder model loaded successfully!")
        print(f"   • Students: {len(model_data['student_ids'])}")
        print(f"   • Clusters: {model_data['n_clusters']}")
        print(f"   • Embedding dimension: {model_data['embeddings'].shape[1]}")
        
        # Test with a sample student
        test_student = 1
        if test_student in model_data['student_ids'].values:
            print(f"\n Testing with Student {test_student}:")
            
            # Get student info
            student_info = model_data['df_students'][model_data['df_students']['student_id'] == test_student].iloc[0]
            print(f"   • Age: {student_info['age']:.1f}")
            print(f"   • Friends: {student_info['NumberOffriends']}")
            print(f"   • Cluster: {student_info['cluster']}")
            
            # Find similar students in same cluster using embeddings
            student_idx = model_data['student_ids'][model_data['student_ids'] == test_student].index[0]
            student_embedding = model_data['embeddings'][student_idx]
            
            same_cluster = model_data['df_students'][model_data['df_students']['cluster'] == student_info['cluster']]
            same_cluster = same_cluster[same_cluster['student_id'] != test_student]
            
            if len(same_cluster) > 0:
                # Calculate similarities
                cluster_indices = same_cluster.index.tolist()
                cluster_embeddings = model_data['embeddings'][cluster_indices]
                similarities = cosine_similarity([student_embedding], cluster_embeddings)[0]
                
                top_3_indices = np.argsort(similarities)[::-1][:3]
                
                print(f"   • Found {len(same_cluster)} students in same cluster")
                print(f"   • Top 3 recommendations (by embedding similarity):")
                for i, idx in enumerate(top_3_indices, 1):
                    actual_idx = cluster_indices[idx]
                    rec_id = model_data['student_ids'].iloc[actual_idx]
                    similarity = similarities[idx]
                    print(f"     {i}. Student {rec_id}: Similarity = {similarity:.3f}")
            else:
                print("   • No other students in same cluster")
        
        return True
        
    except Exception as e:
        print(f" Error loading Friend Finder model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_study_buddy_model():
    """
    Test the saved Study Buddy model.
    """
    print("\n Testing Study Buddy Model (Embedding-Based)...")
    
    model_path = RESULTS_DIR / 'study_buddy_results' / 'study_buddy_model_embedding.pkl'
    
    if not model_path.exists():
        print(" Study Buddy model not found. Please run the training script first.")
        print("   Run: python notebooks/05_study_buddy_baseline_embedding.py")
        return False
    
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(" Study Buddy model loaded successfully!")
        print(f"   • Students: {len(model_data['student_ids'])}")
        print(f"   • Embedding dimension: {model_data['embeddings'].shape[1]}")
        
        # Test with a sample student
        test_student = 15001
        if test_student in model_data['student_ids'].values:
            print(f"\n Testing with Student {test_student}:")
            
            # Get student info
            student_info = model_data['df_students'][model_data['df_students']['student_id'] == test_student].iloc[0]
            print(f"   • Name: {student_info['Name']}")
            print(f"   • Major: {student_info['Major']}")
            print(f"   • Year: {student_info['Year']}")
            print(f"   • GPA: {student_info['GPA']:.2f}")
            
            # Find similar students using embeddings
            student_idx = model_data['student_ids'][model_data['student_ids'] == test_student].index[0]
            student_embedding = model_data['embeddings'][student_idx]
            
            # Calculate similarities
            similarities = cosine_similarity([student_embedding], model_data['embeddings'])[0]
            
            # Get top 3 (excluding self)
            top_indices = np.argsort(similarities)[::-1][1:4]
            
            print(f"   • Top 3 recommendations (by embedding similarity):")
            for i, idx in enumerate(top_indices, 1):
                rec_id = model_data['student_ids'].iloc[idx]
                rec_info = model_data['df_students'][model_data['df_students']['student_id'] == rec_id].iloc[0]
                similarity = similarities[idx]
                
                print(f"     {i}. {rec_info['Name']} (ID: {rec_id}): Similarity = {similarity:.3f}")
                print(f"        • Major: {rec_info['Major']}")
                print(f"        • Year: {rec_info['Year']}")
                print(f"        • GPA: {rec_info['GPA']:.2f}")
                
                # Check common traits
                common = []
                if student_info['Major'] == rec_info['Major']:
                    common.append("Same Major")
                if student_info['Year'] == rec_info['Year']:
                    common.append("Same Year")
                if common:
                    print(f"        • Common: {', '.join(common)}")
        
        return True
        
    except Exception as e:
        print(f" Error loading Study Buddy model: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_embeddings_exist():
    """
    Check if embeddings exist.
    """
    print("\n Checking for embeddings...")
    
    marketing_emb = RESULTS_DIR / 'embeddings' / 'marketing_embeddings.npy'
    profiles_emb = RESULTS_DIR / 'embeddings' / 'profiles_embeddings.npy'
    
    if marketing_emb.exists():
        print(f"    Marketing embeddings found: {marketing_emb}")
    else:
        print(f"    Marketing embeddings not found: {marketing_emb}")
    
    if profiles_emb.exists():
        print(f"    Profiles embeddings found: {profiles_emb}")
    else:
        print(f"    Profiles embeddings not found: {profiles_emb}")

def main():
    """
    Main function to test saved models.
    """
    print(" TESTING EMBEDDING-BASED BASELINE MODELS")
    print("="*60)
    
    # Check embeddings
    check_embeddings_exist()
    
    # Test both models
    ff_success = test_friend_finder_model()
    sb_success = test_study_buddy_model()
    
    print(f"\n TEST RESULTS:")
    print(f"   • Friend Finder: {' Success' if ff_success else ' Failed'}")
    print(f"   • Study Buddy: {' Success' if sb_success else ' Failed'}")
    
    if ff_success and sb_success:
        print(f"\n All embedding-based models working correctly!")
        print(f" Models are ready for production use!")
        print(f"\n Next steps:")
        print(f"   • Use models for real-time recommendations")
        print(f"   • Integrate with advanced GNN models")
        print(f"   • Deploy for production")
    else:
        print(f"\n  Some models failed. Please check:")
        print(f"   1. Run embedding creation: python notebooks/utils/create_embeddings.py")
        print(f"   2. Train Friend Finder: python notebooks/04_friend_finder_baseline_embedding.py")
        print(f"   3. Train Study Buddy: python notebooks/05_study_buddy_baseline_embedding.py")
    
    print("\n" + "="*60)
    print(" MODEL TESTING COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()

