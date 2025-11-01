#!/usr/bin/env python3
"""
Study Buddy Baseline Model (Embedding-Based)
This script implements the Study Buddy model using embedding-based similarity on the profiles dataset.
"""

import pandas as pd
import numpy as np
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
from utils.create_embeddings import create_profiles_embeddings, load_profiles_embeddings

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'baseline'

print(f" STUDY BUDDY BASELINE MODEL (EMBEDDING-BASED)")
print(f"Processed data directory: {PROCESSED_DATA_DIR}")

class StudyBuddyBaseline:
    """
    Study Buddy Baseline Model using embedding-based similarity.
    Matches students based on semantic embeddings for study partner recommendations.
    """
    
    def __init__(self):
        self.embeddings = None
        self.student_ids = None
        self.df_students = None
    
    def load_or_create_embeddings(self, df_students, force_recreate=False):
        """
        Load existing embeddings or create new ones.
        """
        if not force_recreate:
            # Try to load existing embeddings
            embeddings = load_profiles_embeddings()
            if embeddings is not None and len(embeddings) == len(df_students):
                print("    Loaded existing embeddings")
                return embeddings
        
        # Create new embeddings
        print("   • Creating new embeddings...")
        embeddings = create_profiles_embeddings(df_students, save=True)
        return embeddings
    
    def train(self, df_students, force_recreate_embeddings=False):
        """
        Train the Study Buddy model (prepare embeddings).
        """
        print(f"\n Training Study Buddy model (Embedding-Based)...")
        
        self.df_students = df_students.copy()
        self.student_ids = df_students['student_id']
        
        # Load or create embeddings
        print(" Loading or creating embeddings...")
        self.embeddings = self.load_or_create_embeddings(df_students, force_recreate_embeddings)
        
        print(f"   • Embedding shape: {self.embeddings.shape}")
        print(f"   • Each student represented by {self.embeddings.shape[1]}-dimensional embedding")
        
        print(f" Study Buddy model trained successfully!")
        print(f"   • {len(self.student_ids)} students ready for academic matching")
        print(f"   • Using semantic embeddings + complementary GPA matching")
        print(f"   • Pairs lower GPA students with higher GPA students for mutual benefit")
        
        # Save the trained model
        self.save_model()
    
    def recommend_study_buddies(self, student_id, top_k=5):
        """
        Recommend study buddies using complementary matching.
        Pairs students with lower GPA with students with higher GPA.
        Uses embedding similarity for interests/major, but applies GPA complementarity.
        """
        if student_id not in self.student_ids.values:
            print(f" Student ID {student_id} not found.")
            return []
        
        # Find student's embedding and GPA
        student_idx = self.student_ids[self.student_ids == student_id].index[0]
        student_embedding = self.embeddings[student_idx].reshape(1, -1)
        student_gpa = self.df_students.iloc[student_idx]['GPA']
        
        # Calculate embedding similarity (for interests, major, year compatibility)
        similarities = cosine_similarity(student_embedding, self.embeddings)[0]
        
        # Calculate GPA complementarity scores
        # Higher score = better complementary match (larger GPA difference in beneficial direction)
        all_gpas = self.df_students['GPA'].values
        
        # For students with lower GPA: prefer higher GPA buddies (complementary)
        # For students with higher GPA: prefer lower GPA buddies (can help)
        gpa_complementarity = np.zeros(len(all_gpas))
        
        for i, other_gpa in enumerate(all_gpas):
            if i == student_idx:
                gpa_complementarity[i] = -999  # Skip self
            else:
                gpa_diff = abs(student_gpa - other_gpa)
                # Higher score for larger GPA difference (complementary matching)
                # Normalize to 0-1 range
                gpa_complementarity[i] = gpa_diff / 4.0  # Max GPA diff is ~4.0 (2.0 to 4.0)
        
        # Combine embedding similarity and GPA complementarity
        # Weight: 70% embedding similarity, 30% GPA complementarity
        embedding_weight = 0.7
        gpa_weight = 0.3
        
        # Normalize both scores to 0-1 range
        normalized_similarities = (similarities + 1) / 2  # Cosine similarity [-1,1] -> [0,1]
        normalized_gpa = gpa_complementarity  # Already in [0,1]
        
        # Combined score
        combined_scores = (embedding_weight * normalized_similarities + 
                          gpa_weight * normalized_gpa)
        
        # Get top recommendations (excluding the student themselves)
        top_indices = np.argsort(combined_scores)[::-1][1:top_k+1]  # Skip self
        recommended_ids = self.student_ids.iloc[top_indices].tolist()
        
        return recommended_ids
    
    def save_model(self):
        """
        Save the trained model to disk for later use.
        """
        study_buddy_dir = RESULTS_DIR / 'study_buddy_results'
        study_buddy_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_data = {
            'embeddings': self.embeddings,
            'student_ids': self.student_ids,
            'df_students': self.df_students
        }
        
        model_path = study_buddy_dir / 'study_buddy_model_embedding.pkl'
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
        instance = cls()
        
        # Restore model state
        instance.embeddings = model_data['embeddings']
        instance.student_ids = model_data['student_ids']
        instance.df_students = model_data['df_students']
        
        return instance
    
    def get_recommendation_details(self, student_id, top_k=5):
        """
        Get detailed study buddy recommendations with explanations.
        """
        recommendations = self.recommend_study_buddies(student_id, top_k)
        
        if not recommendations:
            return None
        
        # Get student info
        student_info = self.df_students[self.df_students['student_id'] == student_id].iloc[0]
        student_idx = self.student_ids[self.student_ids == student_id].index[0]
        student_embedding = self.embeddings[student_idx]
        
        print(f"\n Student {student_id}:")
        print(f"   • Name: {student_info['Name']}")
        print(f"   • Major: {student_info['Major']}")
        print(f"   • Year: {student_info['Year']}")
        print(f"   • GPA: {student_info['GPA']:.2f}")
        print(f"   • Age: {student_info['Age']}")
        print(f"   • Complementary matching (lower GPA ↔ higher GPA)")
        
        # Show hobbies
        try:
            hobbies = eval(student_info['Hobbies']) if isinstance(student_info['Hobbies'], str) else student_info['Hobbies']
            if isinstance(hobbies, list):
                print(f"   • Hobbies: {', '.join(hobbies[:3])}")
        except:
            pass
        
        print(f"\n Recommended Study Buddies (Complementary Matching):")
        
        # Calculate similarities and GPA differences for display
        similarities = cosine_similarity([student_embedding], self.embeddings)[0]
        student_gpa = student_info['GPA']
        
        # Get recommendation details
        for i, rec_id in enumerate(recommendations, 1):
            rec_info = self.df_students[self.df_students['student_id'] == rec_id].iloc[0]
            rec_idx = self.student_ids[self.student_ids == rec_id].index[0]
            similarity = similarities[rec_idx]
            rec_gpa = rec_info['GPA']
            gpa_diff = abs(student_gpa - rec_gpa)
            
            # Determine complementary benefit
            if rec_gpa > student_gpa:
                benefit = f"Higher GPA (+{rec_gpa - student_gpa:.2f}) - Can help you"
            elif rec_gpa < student_gpa:
                benefit = f"Lower GPA ({rec_gpa - student_gpa:.2f}) - You can help"
            else:
                benefit = "Similar GPA - Peer support"
            
            print(f"   {i}. {rec_info['Name']} (ID: {rec_id})")
            print(f"      • Major: {rec_info['Major']}")
            print(f"      • Year: {rec_info['Year']}")
            print(f"      • GPA: {rec_info['GPA']:.2f} - {benefit}")
            print(f"      • Age: {rec_info['Age']}")
            print(f"      • Embedding Similarity: {similarity:.3f}, GPA Diff: {gpa_diff:.2f}")
            
            # Show common academic traits
            common_traits = []
            if student_info['Major'] == rec_info['Major']:
                common_traits.append("Same Major")
            if student_info['Year'] == rec_info['Year']:
                common_traits.append("Same Year")
            
            if common_traits:
                print(f"      • Common: {', '.join(common_traits)}")
        
        return recommendations
    
    def analyze_academic_patterns(self):
        """
        Analyze academic patterns in the dataset.
        """
        print(f"\n Analyzing academic patterns...")
        
        # GPA analysis
        print(f"   • GPA Statistics:")
        print(f"     - Mean: {self.df_students['GPA'].mean():.2f}")
        print(f"     - Median: {self.df_students['GPA'].median():.2f}")
        print(f"     - Range: {self.df_students['GPA'].min():.2f} - {self.df_students['GPA'].max():.2f}")
        
        # Major analysis
        print(f"   • Top 5 Majors:")
        top_majors = self.df_students['Major'].value_counts().head(5)
        for major, count in top_majors.items():
            print(f"     - {major}: {count} students")
        
        # Year analysis
        print(f"   • Year Distribution:")
        year_counts = self.df_students['Year'].value_counts()
        for year, count in year_counts.items():
            print(f"     - {year}: {count} students")
        
        # Age analysis
        print(f"   • Age Statistics:")
        print(f"     - Mean: {self.df_students['Age'].mean():.1f}")
        print(f"     - Range: {self.df_students['Age'].min()} - {self.df_students['Age'].max()}")

def create_academic_visualization(model):
    """
    Create visualization of academic patterns.
    """
    print(f"\n Creating academic visualization...")
    
    # Use PCA for visualization
    from sklearn.decomposition import PCA
    
    # Reduce embeddings to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(model.embeddings)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Study Buddy - Embedding-Based Academic Analysis', fontsize=16, fontweight='bold')
    
    # 1. GPA distribution
    axes[0, 0].hist(model.df_students['GPA'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    axes[0, 0].set_title('GPA Distribution')
    axes[0, 0].set_xlabel('GPA')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Age distribution
    axes[0, 1].hist(model.df_students['Age'], bins=30, alpha=0.7, edgecolor='black', color='lightgreen')
    axes[0, 1].set_title('Age Distribution')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Year distribution
    year_counts = model.df_students['Year'].value_counts()
    axes[0, 2].bar(year_counts.index, year_counts.values, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 2].set_title('Year Distribution')
    axes[0, 2].set_xlabel('Year')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Top 10 majors
    major_counts = model.df_students['Major'].value_counts().head(10)
    axes[1, 0].barh(range(len(major_counts)), major_counts.values, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 0].set_yticks(range(len(major_counts)))
    axes[1, 0].set_yticklabels(major_counts.index, fontsize=8)
    axes[1, 0].set_title('Top 10 Majors')
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. GPA vs Age scatter
    sample_data = model.df_students.sample(n=min(2000, len(model.df_students)))
    axes[1, 1].scatter(sample_data['Age'], sample_data['GPA'], alpha=0.6, s=20, color='red')
    axes[1, 1].set_title('Age vs GPA')
    axes[1, 1].set_xlabel('Age')
    axes[1, 1].set_ylabel('GPA')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Embedding space (2D PCA) colored by GPA
    scatter = axes[1, 2].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=model.df_students['GPA'], cmap='viridis', 
                                alpha=0.6, s=20)
    axes[1, 2].set_title('Student Embeddings (PCA 2D) Colored by GPA')
    axes[1, 2].set_xlabel('PC1')
    axes[1, 2].set_ylabel('PC2')
    axes[1, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 2], label='GPA')
    
    plt.tight_layout()
    study_buddy_dir = RESULTS_DIR / 'study_buddy_results'
    study_buddy_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(study_buddy_dir / 'study_buddy_analysis_embedding.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Visualization saved to: {study_buddy_dir / 'study_buddy_analysis_embedding.png'}")

def test_study_buddy():
    """
    Test the Study Buddy model with sample students.
    """
    print(f"\n TESTING STUDY BUDDY MODEL (EMBEDDING-BASED)")
    print("="*50)
    
    # Load data
    print(" Loading profiles dataset...")
    df_profiles = pd.read_csv(PROCESSED_DATA_DIR / 'profiles_processed.csv')
    print(f"   • Loaded {len(df_profiles)} students")
    
    # Initialize and train model
    study_buddy = StudyBuddyBaseline()
    study_buddy.train(df_profiles, force_recreate_embeddings=False)
    
    # Analyze academic patterns
    study_buddy.analyze_academic_patterns()
    
    # Test with sample students
    print(f"\n Testing recommendations...")
    
    # Test with different students
    test_students = [15001, 16000, 20000, 25000, 30000]  # From profiles dataset
    
    for student_id in test_students:
        if student_id in df_profiles['student_id'].values:
            print(f"\n{'='*60}")
            study_buddy.get_recommendation_details(student_id, top_k=3)
        else:
            print(f" Student {student_id} not found in dataset")
    
    # Create visualization
    create_academic_visualization(study_buddy)
    
    print(f"\n Study Buddy testing completed!")

def main():
    """
    Main function to run Study Buddy baseline model.
    """
    print(" STARTING STUDY BUDDY BASELINE MODEL (EMBEDDING-BASED)")
    print("="*60)
    
    # Test the model
    test_study_buddy()
    
    print("\n" + "="*60)
    print(" STUDY BUDDY BASELINE MODEL COMPLETED!")
    print("="*60)
    print(" Model ready for study buddy recommendations using embeddings!")
    print(" Academic analysis and visualizations created!")
    print(" Model saved for future use!")

if __name__ == "__main__":
    main()

