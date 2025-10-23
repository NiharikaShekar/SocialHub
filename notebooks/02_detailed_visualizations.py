#!/usr/bin/env python3
"""
Detailed Visualizations for Processed Datasets
This script creates comprehensive charts to understand our processed data better.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

print(f"📊 CREATING DETAILED VISUALIZATIONS")
print(f"Processed data directory: {PROCESSED_DATA_DIR}")

# Load processed datasets
print("\n📁 Loading processed datasets...")
marketing_df = pd.read_csv(PROCESSED_DATA_DIR / 'marketing_processed.csv')
profiles_df = pd.read_csv(PROCESSED_DATA_DIR / 'profiles_processed.csv')

print(f"Marketing dataset: {marketing_df.shape}")
print(f"Profiles dataset: {profiles_df.shape}")

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def create_marketing_visualizations():
    """Create detailed visualizations for marketing dataset."""
    print("\n🎯 Creating Marketing Dataset Visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Marketing Dataset - Detailed Analysis', fontsize=16, fontweight='bold')
    
    # 1. Age distribution
    axes[0, 0].hist(marketing_df['age'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    axes[0, 0].set_title('Age Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Number of friends distribution
    axes[0, 1].hist(marketing_df['NumberOffriends'], bins=30, alpha=0.7, edgecolor='black', color='lightgreen')
    axes[0, 1].set_title('Number of Friends Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Number of Friends')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Gender distribution
    gender_counts = marketing_df['gender'].value_counts()
    axes[0, 2].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 2].set_title('Gender Distribution', fontweight='bold')
    
    # 4. Top 10 most popular interests
    interest_cols = [col for col in marketing_df.columns if col not in ['gradyear', 'gender', 'age', 'NumberOffriends', 'student_id']]
    interest_sums = marketing_df[interest_cols].sum().sort_values(ascending=False).head(10)
    
    axes[1, 0].barh(range(len(interest_sums)), interest_sums.values, alpha=0.7, edgecolor='black')
    axes[1, 0].set_yticks(range(len(interest_sums)))
    axes[1, 0].set_yticklabels(interest_sums.index)
    axes[1, 0].set_title('Top 10 Most Popular Interests', fontweight='bold')
    axes[1, 0].set_xlabel('Total Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Graduation year distribution
    year_counts = marketing_df['gradyear'].value_counts().sort_index()
    axes[1, 1].bar(year_counts.index, year_counts.values, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 1].set_title('Graduation Year Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Graduation Year')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Age vs Number of Friends scatter
    sample_data = marketing_df.sample(n=min(1000, len(marketing_df)))  # Sample for performance
    axes[1, 2].scatter(sample_data['age'], sample_data['NumberOffriends'], alpha=0.6, s=20)
    axes[1, 2].set_title('Age vs Number of Friends', fontweight='bold')
    axes[1, 2].set_xlabel('Age')
    axes[1, 2].set_ylabel('Number of Friends')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'marketing_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Marketing visualizations saved to: {PROCESSED_DATA_DIR / 'marketing_detailed_analysis.png'}")

def create_profiles_visualizations():
    """Create detailed visualizations for profiles dataset."""
    print("\n🎓 Creating Profiles Dataset Visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Student Profiles Dataset - Detailed Analysis', fontsize=16, fontweight='bold')
    
    # 1. GPA distribution
    axes[0, 0].hist(profiles_df['GPA'], bins=30, alpha=0.7, edgecolor='black', color='lightcoral')
    axes[0, 0].set_title('GPA Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('GPA')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Age distribution
    axes[0, 1].hist(profiles_df['Age'], bins=30, alpha=0.7, edgecolor='black', color='lightblue')
    axes[0, 1].set_title('Age Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Year distribution
    year_counts = profiles_df['Year'].value_counts()
    axes[0, 2].bar(year_counts.index, year_counts.values, alpha=0.7, edgecolor='black', color='lightgreen')
    axes[0, 2].set_title('Year Distribution', fontweight='bold')
    axes[0, 2].set_xlabel('Year')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Top 10 majors
    major_counts = profiles_df['Major'].value_counts().head(10)
    axes[1, 0].barh(range(len(major_counts)), major_counts.values, alpha=0.7, edgecolor='black', color='gold')
    axes[1, 0].set_yticks(range(len(major_counts)))
    axes[1, 0].set_yticklabels(major_counts.index, fontsize=8)
    axes[1, 0].set_title('Top 10 Majors', fontweight='bold')
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Sex distribution
    sex_counts = profiles_df['Sex'].value_counts()
    axes[1, 1].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Gender Distribution', fontweight='bold')
    
    # 6. GPA vs Age scatter
    sample_data = profiles_df.sample(n=min(2000, len(profiles_df)))  # Sample for performance
    axes[1, 2].scatter(sample_data['Age'], sample_data['GPA'], alpha=0.6, s=20, color='purple')
    axes[1, 2].set_title('Age vs GPA', fontweight='bold')
    axes[1, 2].set_xlabel('Age')
    axes[1, 2].set_ylabel('GPA')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'profiles_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Profiles visualizations saved to: {PROCESSED_DATA_DIR / 'profiles_detailed_analysis.png'}")

def create_combined_analysis():
    """Create combined analysis of both datasets."""
    print("\n🔗 Creating Combined Analysis...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Combined Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Age comparison between datasets
    axes[0, 0].hist(marketing_df['age'].dropna(), bins=20, alpha=0.7, label='Marketing', color='skyblue')
    axes[0, 0].hist(profiles_df['Age'], bins=20, alpha=0.7, label='Profiles', color='lightcoral')
    axes[0, 0].set_title('Age Distribution Comparison', fontweight='bold')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Dataset sizes comparison
    dataset_sizes = [len(marketing_df), len(profiles_df)]
    dataset_labels = ['Marketing\n(Interests)', 'Profiles\n(Academic)']
    axes[0, 1].bar(dataset_labels, dataset_sizes, alpha=0.7, edgecolor='black', color=['skyblue', 'lightcoral'])
    axes[0, 1].set_title('Dataset Sizes', fontweight='bold')
    axes[0, 1].set_ylabel('Number of Students')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature comparison
    marketing_features = len(marketing_df.columns)
    profiles_features = len(profiles_df.columns)
    feature_counts = [marketing_features, profiles_features]
    feature_labels = ['Marketing\nFeatures', 'Profiles\nFeatures']
    axes[1, 0].bar(feature_labels, feature_counts, alpha=0.7, edgecolor='black', color=['lightgreen', 'gold'])
    axes[1, 0].set_title('Number of Features per Dataset', fontweight='bold')
    axes[1, 0].set_ylabel('Number of Features')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Data quality summary
    marketing_missing = marketing_df.isnull().sum().sum()
    profiles_missing = profiles_df.isnull().sum().sum()
    missing_data = [marketing_missing, profiles_missing]
    missing_labels = ['Marketing\nMissing Values', 'Profiles\nMissing Values']
    axes[1, 1].bar(missing_labels, missing_data, alpha=0.7, edgecolor='black', color=['orange', 'purple'])
    axes[1, 1].set_title('Data Quality - Missing Values', fontweight='bold')
    axes[1, 1].set_ylabel('Total Missing Values')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'combined_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Combined analysis saved to: {PROCESSED_DATA_DIR / 'combined_analysis.png'}")

def print_data_summary():
    """Print comprehensive data summary."""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE DATA SUMMARY")
    print("="*60)
    
    print(f"\n🎯 MARKETING DATASET:")
    print(f"   • Students: {len(marketing_df):,}")
    print(f"   • Features: {len(marketing_df.columns)}")
    print(f"   • Age range: {marketing_df['age'].min():.1f} - {marketing_df['age'].max():.1f}")
    print(f"   • Friends range: {marketing_df['NumberOffriends'].min()} - {marketing_df['NumberOffriends'].max()}")
    print(f"   • Missing values: {marketing_df.isnull().sum().sum()}")
    
    print(f"\n🎓 PROFILES DATASET:")
    print(f"   • Students: {len(profiles_df):,}")
    print(f"   • Features: {len(profiles_df.columns)}")
    print(f"   • Age range: {profiles_df['Age'].min()} - {profiles_df['Age'].max()}")
    print(f"   • GPA range: {profiles_df['GPA'].min():.2f} - {profiles_df['GPA'].max():.2f}")
    print(f"   • Missing values: {profiles_df.isnull().sum().sum()}")
    
    print(f"\n🔗 COMBINED SUMMARY:")
    print(f"   • Total students: {len(marketing_df) + len(profiles_df):,}")
    print(f"   • Marketing → Friend Finder (interest-based)")
    print(f"   • Profiles → Study Buddy (academic-based)")
    print(f"   • Ready for recommendation models!")

def main():
    """Main function to create all visualizations."""
    print("🚀 STARTING DETAILED VISUALIZATION ANALYSIS")
    print("="*60)
    
    # Create visualizations
    create_marketing_visualizations()
    create_profiles_visualizations()
    create_combined_analysis()
    
    # Print summary
    print_data_summary()
    
    print("\n" + "="*60)
    print("✅ DETAILED VISUALIZATION ANALYSIS COMPLETED!")
    print("="*60)
    print(f"📁 All charts saved to: {PROCESSED_DATA_DIR}")
    print("🎯 Ready for baseline model testing!")

if __name__ == "__main__":
    main()
