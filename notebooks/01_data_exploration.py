#!/usr/bin/env python3
"""
Data Exploration and Preprocessing for Social Hub Project
This script explores the downloaded datasets and prepares them for our models.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

print(f"Project root: {PROJECT_ROOT}")
print(f"Raw data directory: {RAW_DATA_DIR}")
print(f"Processed data directory: {PROCESSED_DATA_DIR}")

# Create processed directory if it doesn't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_marketing_dataset():
    """Load and explore the marketing/clustering dataset."""
    print("\n" + "="*50)
    print("📊 LOADING MARKETING/CLUSTERING DATASET")
    print("="*50)
    
    file_path = RAW_DATA_DIR / '03_Clustering_Marketing.csv'
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    return df

def load_student_profiles():
    """Load and explore the student profiles dataset."""
    print("\n" + "="*50)
    print("📊 LOADING STUDENT PROFILES DATASET")
    print("="*50)
    
    file_path = RAW_DATA_DIR / 'student_profiles.jsonl'
    
    # Load JSONL file
    profiles = []
    with open(file_path, 'r') as f:
        for line in f:
            profiles.append(json.loads(line.strip()))
    
    df = pd.DataFrame(profiles)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    return df

def analyze_interest_features(marketing_df):
    """Analyze the interest features in the marketing dataset."""
    print("\n" + "="*50)
    print("🎯 ANALYZING INTEREST FEATURES")
    print("="*50)
    
    # Get interest columns (exclude demographic columns)
    interest_cols = [col for col in marketing_df.columns 
                    if col not in ['gradyear', 'gender', 'age', 'NumberOffriends']]
    
    print(f"Number of interest features: {len(interest_cols)}")
    print(f"Interest features: {interest_cols}")
    
    # Analyze interest patterns
    interest_stats = marketing_df[interest_cols].describe()
    print(f"\nInterest statistics:")
    print(interest_stats)
    
    # Find most popular interests
    interest_sums = marketing_df[interest_cols].sum().sort_values(ascending=False)
    print(f"\nTop 10 most popular interests:")
    print(interest_sums.head(10))
    
    return interest_cols

def analyze_student_profiles(profiles_df):
    """Analyze the student profiles dataset."""
    print("\n" + "="*50)
    print("🎓 ANALYZING STUDENT PROFILES")
    print("="*50)
    
    # Analyze majors
    print(f"Unique majors: {profiles_df['Major'].nunique()}")
    print(f"Top 10 majors:")
    print(profiles_df['Major'].value_counts().head(10))
    
    # Analyze years
    print(f"\nYear distribution:")
    print(profiles_df['Year'].value_counts())
    
    # Analyze GPA
    print(f"\nGPA statistics:")
    print(profiles_df['GPA'].describe())
    
    # Analyze hobbies
    all_hobbies = []
    for hobbies_list in profiles_df['Hobbies']:
        if isinstance(hobbies_list, list):
            all_hobbies.extend(hobbies_list)
    
    from collections import Counter
    hobby_counts = Counter(all_hobbies)
    print(f"\nTop 10 most common hobbies:")
    for hobby, count in hobby_counts.most_common(10):
        print(f"{hobby}: {count}")
    
    return hobby_counts

def create_processed_datasets(marketing_df, profiles_df, interest_cols):
    """Create processed datasets for our models."""
    print("\n" + "="*50)
    print("🔧 CREATING PROCESSED DATASETS")
    print("="*50)
    
    # Process marketing dataset
    print("Processing marketing dataset...")
    marketing_processed = marketing_df.copy()
    
    # Handle missing values
    # Convert age to numeric first, then fill missing values
    marketing_processed['age'] = pd.to_numeric(marketing_processed['age'], errors='coerce')
    marketing_processed['age'] = marketing_processed['age'].fillna(marketing_processed['age'].median())
    marketing_processed['NumberOffriends'] = marketing_processed['NumberOffriends'].fillna(0)
    
    # Create student IDs
    marketing_processed['student_id'] = range(1, len(marketing_processed) + 1)
    
    # Process profiles dataset
    print("Processing student profiles dataset...")
    profiles_processed = profiles_df.copy()
    
    # Create student IDs
    profiles_processed['student_id'] = range(len(marketing_processed) + 1, 
                                           len(marketing_processed) + len(profiles_processed) + 1)
    
    # Save processed datasets
    marketing_processed.to_csv(PROCESSED_DATA_DIR / 'marketing_processed.csv', index=False)
    profiles_processed.to_csv(PROCESSED_DATA_DIR / 'profiles_processed.csv', index=False)
    
    print(f"Saved processed datasets to {PROCESSED_DATA_DIR}")
    
    return marketing_processed, profiles_processed

def create_visualizations(marketing_df, profiles_df):
    """Create visualizations for data exploration."""
    print("\n" + "="*50)
    print("📈 CREATING VISUALIZATIONS")
    print("="*50)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Social Hub Dataset Exploration', fontsize=16)
    
    # 1. Age distribution
    axes[0, 0].hist(marketing_df['age'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Age Distribution (Marketing Dataset)')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Number of friends distribution
    axes[0, 1].hist(marketing_df['NumberOffriends'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Number of Friends Distribution')
    axes[0, 1].set_xlabel('Number of Friends')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. GPA distribution from profiles
    axes[1, 0].hist(profiles_df['GPA'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('GPA Distribution (Profiles Dataset)')
    axes[1, 0].set_xlabel('GPA')
    axes[1, 0].set_ylabel('Frequency')
    
    # 4. Year distribution
    year_counts = profiles_df['Year'].value_counts()
    axes[1, 1].bar(year_counts.index, year_counts.values, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Year Distribution (Profiles Dataset)')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'data_exploration_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved visualization to {PROCESSED_DATA_DIR / 'data_exploration_plots.png'}")

def main():
    """Main function to run the data exploration."""
    print("🚀 STARTING DATA EXPLORATION AND PREPROCESSING")
    print("="*60)
    
    # Load datasets
    marketing_df = load_marketing_dataset()
    profiles_df = load_student_profiles()
    
    # Analyze features
    interest_cols = analyze_interest_features(marketing_df)
    hobby_counts = analyze_student_profiles(profiles_df)
    
    # Create processed datasets
    marketing_processed, profiles_processed = create_processed_datasets(
        marketing_df, profiles_df, interest_cols
    )
    
    # Create visualizations
    create_visualizations(marketing_df, profiles_df)
    
    print("\n" + "="*60)
    print("✅ DATA EXPLORATION COMPLETED!")
    print("="*60)
    print(f"Processed datasets saved to: {PROCESSED_DATA_DIR}")
    print(f"Ready for next step: Model development!")

if __name__ == "__main__":
    main()
