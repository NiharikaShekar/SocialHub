#!/usr/bin/env python3
"""
Study Buddy Baseline Model
This script implements and tests the Study Buddy model using academic profile matching on the profiles dataset.
"""

import pandas as pd
import numpy as np
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

print(f"📚 STUDY BUDDY BASELINE MODEL")
print(f"Processed data directory: {PROCESSED_DATA_DIR}")

class StudyBuddyBaseline:
    """
    Study Buddy Baseline Model using academic profile matching.
    Matches students based on GPA, major, year, and academic interests.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.df_students = None
        self.student_features = None
        self.student_ids = None
        self.feature_cols = None
        
    def prepare_features(self, df):
        """
        Prepare academic features for matching.
        """
        print("🔧 Preparing academic features for matching...")
        
        # Select academic features
        feature_cols = ['Age', 'GPA']
        
        # Add year and major as encoded features
        df_encoded = df.copy()
        
        # Encode year
        year_mapping = {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4, 'Graduate': 5}
        df_encoded['year_encoded'] = df_encoded['Year'].map(year_mapping)
        
        # Encode major (simple approach - could be improved)
        unique_majors = df_encoded['Major'].unique()
        major_mapping = {major: i for i, major in enumerate(unique_majors)}
        df_encoded['major_encoded'] = df_encoded['Major'].map(major_mapping)
        
        # Select features for matching
        self.feature_cols = feature_cols + ['year_encoded', 'major_encoded']
        
        print(f"   • Using {len(self.feature_cols)} academic features")
        print(f"   • Features: {self.feature_cols}")
        
        return df_encoded[self.feature_cols]
    
    def train(self, df_students):
        """
        Train the Study Buddy model.
        """
        print(f"\n🎯 Training Study Buddy model...")
        
        self.df_students = df_students.copy()
        self.student_ids = df_students['student_id']
        
        # Prepare features
        self.student_features = self.prepare_features(df_students)
        
        # Handle missing values
        self.student_features = self.student_features.fillna(0)
        
        # Scale features
        print("   • Scaling academic features...")
        self.student_features_scaled = self.scaler.fit_transform(self.student_features)
        
        print(f"✅ Study Buddy model trained successfully!")
        print(f"   • {len(self.student_ids)} students ready for academic matching")
        
        # Save the trained model
        self.save_model()
    
    def recommend_study_buddies(self, student_id, top_k=5):
        """
        Recommend study buddies for a given student.
        """
        if student_id not in self.student_ids.values:
            print(f"❌ Student ID {student_id} not found.")
            return []
        
        # Find student's features
        student_idx = self.student_ids[self.student_ids == student_id].index[0]
        student_features = self.student_features_scaled[student_idx].reshape(1, -1)
        
        # Calculate similarity with all other students
        similarities = cosine_similarity(student_features, self.student_features_scaled)[0]
        
        # Get top recommendations (excluding the student themselves)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Skip self
        recommended_ids = self.student_ids.iloc[top_indices].tolist()
        
        return recommended_ids
    
    def save_model(self):
        """
        Save the trained model to disk for later use.
        """
        results_dir = PROJECT_ROOT / 'results' / 'baseline'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_data = {
            'scaler': self.scaler,
            'student_features': self.student_features,
            'student_features_scaled': self.student_features_scaled,
            'student_ids': self.student_ids,
            'df_students': self.df_students,
            'feature_cols': self.feature_cols
        }
        
        model_path = results_dir / 'study_buddy_model.pkl'
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
        instance = cls()
        
        # Restore model state
        instance.scaler = model_data['scaler']
        instance.student_features = model_data['student_features']
        instance.student_features_scaled = model_data['student_features_scaled']
        instance.student_ids = model_data['student_ids']
        instance.df_students = model_data['df_students']
        instance.feature_cols = model_data['feature_cols']
        
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
        
        print(f"\n👤 Student {student_id}:")
        print(f"   • Name: {student_info['Name']}")
        print(f"   • Major: {student_info['Major']}")
        print(f"   • Year: {student_info['Year']}")
        print(f"   • GPA: {student_info['GPA']:.2f}")
        print(f"   • Age: {student_info['Age']}")
        
        # Show hobbies
        hobbies = eval(student_info['Hobbies']) if isinstance(student_info['Hobbies'], str) else student_info['Hobbies']
        print(f"   • Hobbies: {', '.join(hobbies[:3])}")
        
        print(f"\n📚 Recommended Study Buddies:")
        
        # Get recommendation details
        for i, rec_id in enumerate(recommendations, 1):
            rec_info = self.df_students[self.df_students['student_id'] == rec_id].iloc[0]
            print(f"   {i}. {rec_info['Name']} (ID: {rec_id})")
            print(f"      • Major: {rec_info['Major']}")
            print(f"      • Year: {rec_info['Year']}")
            print(f"      • GPA: {rec_info['GPA']:.2f}")
            print(f"      • Age: {rec_info['Age']}")
            
            # Show common academic traits
            common_traits = []
            if student_info['Major'] == rec_info['Major']:
                common_traits.append("Same Major")
            if student_info['Year'] == rec_info['Year']:
                common_traits.append("Same Year")
            if abs(student_info['GPA'] - rec_info['GPA']) < 0.5:
                common_traits.append("Similar GPA")
            
            if common_traits:
                print(f"      • Common: {', '.join(common_traits)}")
        
        return recommendations
    
    def analyze_academic_patterns(self):
        """
        Analyze academic patterns in the dataset.
        """
        print(f"\n📊 Analyzing academic patterns...")
        
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

def test_study_buddy():
    """
    Test the Study Buddy model with sample students.
    """
    print(f"\n🧪 TESTING STUDY BUDDY MODEL")
    print("="*50)
    
    # Load data
    print("📁 Loading profiles dataset...")
    df_profiles = pd.read_csv(PROCESSED_DATA_DIR / 'profiles_processed.csv')
    print(f"   • Loaded {len(df_profiles)} students")
    
    # Initialize and train model
    study_buddy = StudyBuddyBaseline()
    study_buddy.train(df_profiles)
    
    # Analyze academic patterns
    study_buddy.analyze_academic_patterns()
    
    # Test with sample students
    print(f"\n🎯 Testing recommendations...")
    
    # Test with different students
    test_students = [15001, 16000, 20000, 25000, 30000]  # From profiles dataset
    
    for student_id in test_students:
        if student_id in df_profiles['student_id'].values:
            print(f"\n{'='*60}")
            study_buddy.get_recommendation_details(student_id, top_k=3)
        else:
            print(f"❌ Student {student_id} not found in dataset")
    
    # Create visualization
    create_academic_visualization(study_buddy)
    
    print(f"\n✅ Study Buddy testing completed!")

def create_academic_visualization(model):
    """
    Create visualization of academic patterns.
    """
    print(f"\n📊 Creating academic visualization...")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Study Buddy - Academic Analysis', fontsize=16, fontweight='bold')
    
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
    
    # 6. GPA by Year
    gpa_by_year = model.df_students.groupby('Year')['GPA'].mean()
    axes[1, 2].bar(gpa_by_year.index, gpa_by_year.values, alpha=0.7, edgecolor='black', color='gold')
    axes[1, 2].set_title('Average GPA by Year')
    axes[1, 2].set_xlabel('Year')
    axes[1, 2].set_ylabel('Average GPA')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save to results folder
    results_dir = PROJECT_ROOT / 'results' / 'baseline'
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / 'study_buddy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved to: {results_dir / 'study_buddy_analysis.png'}")

def main():
    """
    Main function to run Study Buddy baseline model.
    """
    print("🚀 STARTING STUDY BUDDY BASELINE MODEL")
    print("="*60)
    
    # Test the model
    test_study_buddy()
    
    print("\n" + "="*60)
    print("✅ STUDY BUDDY BASELINE MODEL COMPLETED!")
    print("="*60)
    print("🎯 Model ready for study buddy recommendations!")
    print("📊 Academic analysis and visualizations created!")

if __name__ == "__main__":
    main()
