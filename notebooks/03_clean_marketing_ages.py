#!/usr/bin/env python3
"""
Clean Marketing Dataset Ages
This script filters out unrealistic ages (>60) and replaces them with median age.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

print(f"🧹 CLEANING MARKETING DATASET AGES")
print(f"Processed data directory: {PROCESSED_DATA_DIR}")

# Load the marketing dataset
print("\n📁 Loading marketing dataset...")
marketing_df = pd.read_csv(PROCESSED_DATA_DIR / 'marketing_processed.csv')

print(f"Original dataset shape: {marketing_df.shape}")
print(f"Age range before cleaning: {marketing_df['age'].min():.1f} - {marketing_df['age'].max():.1f}")

# Check for unrealistic ages
print(f"\n🔍 Analyzing age distribution...")
print(f"Students with age > 60: {(marketing_df['age'] > 60).sum()}")
print(f"Students with age < 15: {(marketing_df['age'] < 15).sum()}")
print(f"Students with age between 15-60: {((marketing_df['age'] >= 15) & (marketing_df['age'] <= 60)).sum()}")

# Show some examples of unrealistic ages
unrealistic_ages = marketing_df[marketing_df['age'] > 60]
if len(unrealistic_ages) > 0:
    print(f"\n📊 Examples of unrealistic ages:")
    print(unrealistic_ages[['student_id', 'age', 'gender', 'NumberOffriends']].head(10))

# Clean the data
print(f"\n🧹 Cleaning ages...")

# Calculate median age from realistic ages (15-60)
realistic_ages = marketing_df[(marketing_df['age'] >= 15) & (marketing_df['age'] <= 60)]['age']
median_age = realistic_ages.median()

print(f"Median age from realistic data: {median_age:.1f}")

# Replace unrealistic ages with median
marketing_cleaned = marketing_df.copy()
marketing_cleaned.loc[marketing_cleaned['age'] > 60, 'age'] = median_age
marketing_cleaned.loc[marketing_cleaned['age'] < 15, 'age'] = median_age

print(f"Age range after cleaning: {marketing_cleaned['age'].min():.1f} - {marketing_cleaned['age'].max():.1f}")

# Show the impact
print(f"\n📈 Cleaning impact:")
print(f"Students with age > 60: {(marketing_cleaned['age'] > 60).sum()}")
print(f"Students with age < 15: {(marketing_cleaned['age'] < 15).sum()}")
print(f"Students with age between 15-60: {((marketing_cleaned['age'] >= 15) & (marketing_cleaned['age'] <= 60)).sum()}")

# Save the cleaned dataset
output_file = PROCESSED_DATA_DIR / 'marketing_cleaned.csv'
marketing_cleaned.to_csv(output_file, index=False)

print(f"\n✅ Cleaned dataset saved to: {output_file}")
print(f"Final dataset shape: {marketing_cleaned.shape}")

# Create a quick visualization of the cleaning
import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Age Cleaning Impact', fontsize=16, fontweight='bold')

# Before cleaning
ax1.hist(marketing_df['age'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='red')
ax1.set_title('Before Cleaning', fontweight='bold')
ax1.set_xlabel('Age')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=60, color='red', linestyle='--', label='Age > 60')
ax1.legend()

# After cleaning
ax2.hist(marketing_cleaned['age'], bins=30, alpha=0.7, edgecolor='black', color='green')
ax2.set_title('After Cleaning', fontweight='bold')
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=60, color='red', linestyle='--', label='Age > 60')
ax2.legend()

plt.tight_layout()
plt.savefig(PROCESSED_DATA_DIR / 'age_cleaning_impact.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"📊 Visualization saved to: {PROCESSED_DATA_DIR / 'age_cleaning_impact.png'}")

print("\n" + "="*60)
print("✅ AGE CLEANING COMPLETED!")
print("="*60)
print(f"📁 Cleaned dataset: {output_file}")
print(f"📊 Visualization: {PROCESSED_DATA_DIR / 'age_cleaning_impact.png'}")
print("🎯 Ready for baseline model testing!")
