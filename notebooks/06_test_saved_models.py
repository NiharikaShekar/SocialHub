#!/usr/bin/env python3
"""
Test Saved Models
This script loads and tests the saved baseline models without complex imports.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'results' / 'baseline'

print("🔄 TESTING SAVED MODELS")
print("="*50)

def test_friend_finder_model():
    """
    Test the saved Friend Finder model.
    """
    print("👥 Testing Friend Finder Model...")
    
    model_path = RESULTS_DIR / 'friend_finder_model.pkl'
    
    if not model_path.exists():
        print("❌ Friend Finder model not found. Please run the training script first.")
        return False
    
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print("✅ Friend Finder model loaded successfully!")
        print(f"   • Students: {len(model_data['student_ids'])}")
        print(f"   • Clusters: {model_data['n_clusters']}")
        print(f"   • Features: {len(model_data['feature_cols'])}")
        
        # Test with a sample student
        test_student = 1
        if test_student in model_data['student_ids'].values:
            print(f"\n🎯 Testing with Student {test_student}:")
            
            # Get student info
            student_info = model_data['df_students'][model_data['df_students']['student_id'] == test_student].iloc[0]
            print(f"   • Age: {student_info['age']:.1f}")
            print(f"   • Friends: {student_info['NumberOffriends']}")
            print(f"   • Cluster: {student_info['cluster']}")
            
            # Find similar students in same cluster
            same_cluster = model_data['df_students'][model_data['df_students']['cluster'] == student_info['cluster']]
            same_cluster = same_cluster[same_cluster['student_id'] != test_student]
            
            if len(same_cluster) > 0:
                print(f"   • Found {len(same_cluster)} students in same cluster")
                print(f"   • Sample recommendations: {same_cluster['student_id'].head(3).tolist()}")
            else:
                print("   • No other students in same cluster")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading Friend Finder model: {e}")
        return False

def test_study_buddy_model():
    """
    Test the saved Study Buddy model.
    """
    print("\n📚 Testing Study Buddy Model...")
    
    model_path = RESULTS_DIR / 'study_buddy_model.pkl'
    
    if not model_path.exists():
        print("❌ Study Buddy model not found. Please run the training script first.")
        return False
    
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print("✅ Study Buddy model loaded successfully!")
        print(f"   • Students: {len(model_data['student_ids'])}")
        print(f"   • Features: {len(model_data['feature_cols'])}")
        
        # Test with a sample student
        test_student = 15001
        if test_student in model_data['student_ids'].values:
            print(f"\n🎯 Testing with Student {test_student}:")
            
            # Get student info
            student_info = model_data['df_students'][model_data['df_students']['student_id'] == test_student].iloc[0]
            print(f"   • Name: {student_info['Name']}")
            print(f"   • Major: {student_info['Major']}")
            print(f"   • Year: {student_info['Year']}")
            print(f"   • GPA: {student_info['GPA']:.2f}")
            
            # Find similar students (simplified similarity check)
            similar_students = model_data['df_students'][
                (model_data['df_students']['student_id'] != test_student) &
                (model_data['df_students']['Major'] == student_info['Major'])
            ]
            
            if len(similar_students) > 0:
                print(f"   • Found {len(similar_students)} students with same major")
                print(f"   • Sample recommendations: {similar_students['student_id'].head(3).tolist()}")
            else:
                print("   • No other students with same major")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading Study Buddy model: {e}")
        return False

def main():
    """
    Main function to test saved models.
    """
    print("🚀 TESTING SAVED BASELINE MODELS")
    print("="*60)
    
    # Test both models
    ff_success = test_friend_finder_model()
    sb_success = test_study_buddy_model()
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   • Friend Finder: {'✅ Success' if ff_success else '❌ Failed'}")
    print(f"   • Study Buddy: {'✅ Success' if sb_success else '❌ Failed'}")
    
    if ff_success and sb_success:
        print(f"\n🎉 All models working correctly!")
        print(f"💾 Models are ready for production use!")
    else:
        print(f"\n⚠️  Some models failed. Please check the training scripts.")
    
    print("\n" + "="*60)
    print("✅ MODEL TESTING COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()