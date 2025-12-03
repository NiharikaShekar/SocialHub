#!/usr/bin/env python3
"""
Simple Event Recommendations Demo
Takes 100 student profiles and recommends events for them, then visualizes results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'notebooks'))

# Import the event recommendation engine
import importlib.util
spec = importlib.util.spec_from_file_location(
    "event_recommendation",
    Path(__file__).parent / "15_event_recommendation.py"
)
event_recommendation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(event_recommendation_module)
EventRecommendationEngine = event_recommendation_module.EventRecommendationEngine

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'event_recommendations'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_student_profiles(n_profiles=100):
    """Load n student profiles from processed data."""
    print(f"Loading {n_profiles} student profiles...")
    
    profiles_path = PROCESSED_DATA_DIR / 'profiles_processed.csv'
    if not profiles_path.exists():
        print(f"Error: Profiles file not found at {profiles_path}")
        return None
    
    df = pd.read_csv(profiles_path)
    
    # Take first n profiles
    df_sample = df.head(n_profiles).copy()
    
    print(f"✓ Loaded {len(df_sample)} student profiles")
    return df_sample

def get_recommendations_for_students(engine, students_df, top_k=3):
    """Get event recommendations for all students."""
    print(f"\nGetting event recommendations (top {top_k} per student)...")
    
    results = []
    
    for idx, student in students_df.iterrows():
        # Convert student row to dictionary
        student_dict = student.to_dict()
        
        # Get recommendations
        recommendations = engine.recommend_events(
            student_profile=student_dict,
            top_k=top_k,
            date_range_days=60  # Look at events in next 60 days
        )
        
        # Store results
        for rec in recommendations:
            results.append({
                'student_name': student_dict.get('Name', f'Student_{idx}'),
                'student_major': student_dict.get('Major', 'Unknown'),
                'student_hobbies': student_dict.get('Hobbies', ''),
                'event_title': rec.get('title', ''),
                'event_category': rec.get('category', ''),
                'event_date': rec.get('date', ''),
                'event_location': rec.get('location', ''),
                'similarity_score': rec.get('similarity_score', 0.0),
                'rank': rec.get('rank', 0)
            })
        
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(students_df)} students...")
    
    print(f"✓ Generated {len(results)} recommendations")
    return pd.DataFrame(results)

def visualize_recommendations(recommendations_df):
    """Create visualizations for event recommendations - saves 6 separate figures."""
    print("\nCreating 6 separate visualizations...")
    
    # 1. Top recommended event categories
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    category_counts = recommendations_df['event_category'].value_counts().head(10)
    ax1.barh(range(len(category_counts)), category_counts.values, color='steelblue')
    ax1.set_yticks(range(len(category_counts)))
    ax1.set_yticklabels(category_counts.index, fontsize=10)
    ax1.set_xlabel('Number of Recommendations', fontsize=11)
    ax1.set_title('Top Recommended Event Categories', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    output_path1 = RESULTS_DIR / '1_top_event_categories.png'
    plt.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path1.name}")
    plt.close()
    
    # 2. Similarity score distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(recommendations_df['similarity_score'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Similarity Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Similarity Scores', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    output_path2 = RESULTS_DIR / '2_similarity_score_distribution.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path2.name}")
    plt.close()
    
    # 3. Events by location type
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    locations = recommendations_df['event_location'].fillna('Unknown')
    is_online = locations.str.contains('Online|online|virtual', case=False, na=False)
    location_type = ['Online' if online else 'In-Person' for online in is_online]
    location_counts = pd.Series(location_type).value_counts()
    ax3.pie(location_counts.values, labels=location_counts.index, autopct='%1.1f%%',
            colors=['lightblue', 'lightcoral'], startangle=90, textprops={'fontsize': 12})
    ax3.set_title('Online vs In-Person Events', fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_path3 = RESULTS_DIR / '3_online_vs_inperson.png'
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path3.name}")
    plt.close()
    
    # 4. Top majors getting recommendations
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    major_counts = recommendations_df['student_major'].value_counts().head(10)
    ax4.barh(range(len(major_counts)), major_counts.values, color='mediumseagreen')
    ax4.set_yticks(range(len(major_counts)))
    ax4.set_yticklabels(major_counts.index, fontsize=10)
    ax4.set_xlabel('Number of Recommendations', fontsize=11)
    ax4.set_title('Top Majors Receiving Recommendations', fontsize=13, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    output_path4 = RESULTS_DIR / '4_top_majors_recommendations.png'
    plt.savefig(output_path4, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path4.name}")
    plt.close()
    
    # 5. Average similarity score by category
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    category_scores = recommendations_df.groupby('event_category')['similarity_score'].mean().sort_values(ascending=False).head(8)
    ax5.barh(range(len(category_scores)), category_scores.values, color='gold')
    ax5.set_yticks(range(len(category_scores)))
    ax5.set_yticklabels(category_scores.index, fontsize=10)
    ax5.set_xlabel('Average Similarity Score', fontsize=11)
    ax5.set_title('Average Similarity Score by Category', fontsize=13, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    output_path5 = RESULTS_DIR / '5_avg_similarity_by_category.png'
    plt.savefig(output_path5, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path5.name}")
    plt.close()
    
    # 6. Recommendations per student
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    recs_per_student = recommendations_df.groupby('student_name').size()
    ax6.hist(recs_per_student, bins=15, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Number of Recommendations', fontsize=11)
    ax6.set_ylabel('Number of Students', fontsize=11)
    ax6.set_title('Recommendations per Student', fontsize=13, fontweight='bold')
    ax6.grid(alpha=0.3)
    plt.tight_layout()
    output_path6 = RESULTS_DIR / '6_recommendations_per_student.png'
    plt.savefig(output_path6, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path6.name}")
    plt.close()
    
    print(f"\n✓ All 6 visualizations saved to {RESULTS_DIR}")

def create_summary_table(recommendations_df):
    """Create a summary table of top recommendations."""
    print("\nCreating summary table...")
    
    # Get top recommendations by similarity score
    top_recommendations = recommendations_df.nlargest(20, 'similarity_score')[
        ['student_name', 'student_major', 'event_title', 'event_category', 
         'similarity_score', 'event_date']
    ]
    
    # Save to CSV
    summary_path = RESULTS_DIR / 'top_event_recommendations.csv'
    top_recommendations.to_csv(summary_path, index=False)
    print(f"✓ Saved summary table to {summary_path}")
    
    # Print top 10
    print("\n" + "="*80)
    print(" TOP 10 EVENT RECOMMENDATIONS")
    print("="*80)
    print(top_recommendations.head(10).to_string(index=False))
    
    return top_recommendations

def main():
    """Main function."""
    print("="*80)
    print(" UIC EVENT RECOMMENDATIONS DEMO")
    print("="*80)
    
    # Step 1: Initialize recommendation engine
    print("\n[Step 1] Initializing Event Recommendation Engine...")
    engine = EventRecommendationEngine()
    engine.load_data()
    engine.create_event_embeddings()
    print("✓ Engine ready!")
    
    # Step 2: Load student profiles
    print("\n[Step 2] Loading Student Profiles...")
    students_df = load_student_profiles(n_profiles=100)
    if students_df is None:
        return
    
    # Step 3: Get recommendations
    print("\n[Step 3] Generating Recommendations...")
    recommendations_df = get_recommendations_for_students(engine, students_df, top_k=3)
    
    if len(recommendations_df) == 0:
        print("No recommendations generated. Check if events data is available.")
        return
    
    # Step 4: Visualize
    print("\n[Step 4] Creating Visualizations...")
    visualize_recommendations(recommendations_df)
    
    # Step 5: Create summary
    print("\n[Step 5] Creating Summary...")
    summary = create_summary_table(recommendations_df)
    
    # Final statistics
    print("\n" + "="*80)
    print(" SUMMARY STATISTICS")
    print("="*80)
    print(f"Total Students: {students_df.shape[0]}")
    print(f"Total Recommendations: {len(recommendations_df)}")
    print(f"Average Recommendations per Student: {len(recommendations_df) / len(students_df):.1f}")
    print(f"Average Similarity Score: {recommendations_df['similarity_score'].mean():.3f}")
    print(f"Unique Events Recommended: {recommendations_df['event_title'].nunique()}")
    print(f"Unique Categories: {recommendations_df['event_category'].nunique()}")
    print("="*80)
    
    print("\n✓ Demo completed successfully!")
    print(f"Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()

