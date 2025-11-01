#!/usr/bin/env python3
"""
Embedding Creation Utility
This script creates embeddings for student profiles that can be used by both baseline and advanced models.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'baseline'

def create_marketing_embeddings(df_marketing, save=True):
    """
    Create embeddings for marketing dataset (Friend Finder).
    
    Args:
        df_marketing: DataFrame with marketing student data
        save: Whether to save embeddings to file
    
    Returns:
        Unified embeddings matrix
    """
    print("Creating embeddings for Marketing dataset (Friend Finder)...")
    
    # 1. Create text embeddings from interests
    print("   • Step 1: Generating text embeddings from interests...")
    interest_cols = [col for col in df_marketing.columns 
                    if col not in ['gradyear', 'gender', 'age', 'NumberOffriends', 'student_id']]
    
    # Create text representation from interests
    def create_interest_text(row):
        active_interests = [col for col in interest_cols if row[col] > 0]
        return ' '.join(active_interests) if active_interests else 'no interests'
    
    interest_texts = df_marketing.apply(create_interest_text, axis=1)
    
    # Load sentence transformer model
    print("   • Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate text embeddings
    print("   • Encoding interest texts (this may take a moment)...")
    text_embeddings = model.encode(interest_texts.tolist(), show_progress_bar=True)
    print(f"   • Text embeddings shape: {text_embeddings.shape}")
    
    # 2. Process numerical features
    print("   • Step 2: Processing numerical features...")
    numerical_features = ['age', 'NumberOffriends']
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(df_marketing[numerical_features].fillna(0))
    
    # 3. Combine features
    print("   • Step 3: Combining all features...")
    # Normalize each component
    text_norm = normalize(text_embeddings, norm='l2', axis=1)
    num_norm = normalize(numerical_scaled, norm='l2', axis=1)
    
    # Combine
    unified_embeddings = np.hstack([text_norm, num_norm])
    
    # Final normalization
    unified_embeddings = normalize(unified_embeddings, norm='l2', axis=1)
    
    print(f"   • Final unified embeddings shape: {unified_embeddings.shape}")
    
    # 4. Save if requested
    if save:
        embeddings_dir = RESULTS_DIR / 'embeddings'
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = embeddings_dir / 'marketing_embeddings.npy'
        np.save(embedding_path, unified_embeddings)
        
        # Save scaler and metadata
        metadata = {
            'scaler': scaler,
            'interest_cols': interest_cols,
            'numerical_features': numerical_features,
            'embedding_model': 'all-MiniLM-L6-v2'
        }
        metadata_path = embeddings_dir / 'marketing_embeddings_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"    Embeddings saved to: {embedding_path}")
        print(f"    Metadata saved to: {metadata_path}")
    
    return unified_embeddings

def create_profiles_embeddings(df_profiles, save=True):
    """
    Create embeddings for profiles dataset (Study Buddy).
    
    Args:
        df_profiles: DataFrame with student profile data
        save: Whether to save embeddings to file
    
    Returns:
        Unified embeddings matrix
    """
    print(" Creating embeddings for Profiles dataset (Study Buddy)...")
    
    # 1. Create text embeddings from profile text
    print("   • Step 1: Generating text embeddings from profile text...")
    
    # Combine text fields
    text_columns = ['Hobbies', 'Unique Quality', 'Story']
    for col in text_columns:
        if col in df_profiles.columns:
            df_profiles[col] = df_profiles[col].fillna('')
    
    # Create combined profile text
    df_profiles['profile_text'] = df_profiles[text_columns].apply(
        lambda row: ' '.join(row.values.astype(str)), axis=1
    )
    
    # Load sentence transformer model
    print("   • Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate text embeddings
    print("   • Encoding profile texts (this may take a moment)...")
    text_embeddings = model.encode(df_profiles['profile_text'].tolist(), show_progress_bar=True)
    print(f"   • Text embeddings shape: {text_embeddings.shape}")
    
    # 2. Process numerical and categorical features
    print("   • Step 2: Processing numerical and categorical features...")
    
    # Numerical features
    numerical_features = ['Age', 'GPA']
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(df_profiles[numerical_features])
    
    # Categorical features (one-hot encoding)
    categorical_features = ['Sex', 'Major', 'Year']
    categorical_df = pd.get_dummies(df_profiles[categorical_features], 
                                   prefix_sep='_', drop_first=False)
    
    # 3. Combine all features
    print("   • Step 3: Combining all features...")
    
    # Normalize each component
    text_norm = normalize(text_embeddings, norm='l2', axis=1)
    num_norm = normalize(numerical_scaled, norm='l2', axis=1)
    cat_norm = normalize(categorical_df.values, norm='l2', axis=1)
    
    # Combine
    unified_embeddings = np.hstack([text_norm, num_norm, cat_norm])
    
    # Final normalization
    unified_embeddings = normalize(unified_embeddings, norm='l2', axis=1)
    
    print(f"   • Final unified embeddings shape: {unified_embeddings.shape}")
    
    # 4. Save if requested
    if save:
        embeddings_dir = RESULTS_DIR / 'embeddings'
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = embeddings_dir / 'profiles_embeddings.npy'
        np.save(embedding_path, unified_embeddings)
        
        # Save scaler, categorical mapping, and metadata
        metadata = {
            'scaler': scaler,
            'categorical_features': categorical_features,
            'categorical_columns': categorical_df.columns.tolist(),
            'numerical_features': numerical_features,
            'embedding_model': 'all-MiniLM-L6-v2'
        }
        metadata_path = embeddings_dir / 'profiles_embeddings_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"    Embeddings saved to: {embedding_path}")
        print(f"    Metadata saved to: {metadata_path}")
    
    return unified_embeddings

def load_marketing_embeddings():
    """Load pre-computed marketing embeddings."""
    embedding_path = RESULTS_DIR / 'embeddings' / 'marketing_embeddings.npy'
    if embedding_path.exists():
        return np.load(embedding_path)
    else:
        print(" Marketing embeddings not found. Run create_marketing_embeddings first.")
        return None

def load_profiles_embeddings():
    """Load pre-computed profiles embeddings."""
    embedding_path = RESULTS_DIR / 'embeddings' / 'profiles_embeddings.npy'
    if embedding_path.exists():
        return np.load(embedding_path)
    else:
        print(" Profiles embeddings not found. Run create_profiles_embeddings first.")
        return None

if __name__ == "__main__":
    """
    Main function to create embeddings for both datasets.
    """
    print(" CREATING EMBEDDINGS FOR BASELINE MODELS")
    print("="*60)
    
    # Load datasets
    print("\n Loading datasets...")
    df_marketing = pd.read_csv(PROCESSED_DATA_DIR / 'marketing_processed.csv')
    df_profiles = pd.read_csv(PROCESSED_DATA_DIR / 'profiles_processed.csv')
    
    print(f"   • Marketing dataset: {len(df_marketing)} students")
    print(f"   • Profiles dataset: {len(df_profiles)} students")
    
    # Create embeddings
    print("\n" + "="*60)
    marketing_embeddings = create_marketing_embeddings(df_marketing)
    
    print("\n" + "="*60)
    profiles_embeddings = create_profiles_embeddings(df_profiles)
    
    print("\n" + "="*60)
    print(" ALL EMBEDDINGS CREATED SUCCESSFULLY!")
    print("="*60)
    print(" Embeddings saved to:", RESULTS_DIR)
    print(" Ready for baseline model training!")

