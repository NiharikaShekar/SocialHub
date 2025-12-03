#!/usr/bin/env python3
"""
Event Recommendation System for UIC Students
Uses content-based filtering and semantic embeddings to recommend events to students.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
import warnings
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'notebooks'))

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'event_recommendations'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class EventRecommendationEngine:
    """
    Content-based event recommendation engine using semantic embeddings.
    Matches student profiles to UIC events based on interests, major, and preferences.
    """
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the recommendation engine."""
        print("Initializing Event Recommendation Engine...")
        print(f"  Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.events_df = None
        self.students_df = None
        self.event_embeddings = None
        self.student_embeddings = None
        self.event_text_features = None
        
    def load_data(self, events_path: Optional[str] = None, 
                  students_path: Optional[str] = None):
        """Load events and student data."""
        print("\nLoading data...")
        
        # Load events
        if events_path is None:
            events_path = PROCESSED_DATA_DIR / 'uic_events_processed.csv'
        
        if Path(events_path).exists():
            self.events_df = pd.read_csv(events_path)
            # Parse date column
            if 'date' in self.events_df.columns:
                self.events_df['date'] = pd.to_datetime(self.events_df['date'], errors='coerce')
            print(f"  ✓ Loaded {len(self.events_df)} events")
        else:
            print(f"  ✗ Events file not found: {events_path}")
            print("  Creating sample events...")
            self._create_sample_events()
        
        # Load student profiles
        if students_path is None:
            students_path = PROCESSED_DATA_DIR / 'profiles_processed.csv'
        
        if Path(students_path).exists():
            self.students_df = pd.read_csv(students_path)
            print(f"  ✓ Loaded {len(self.students_df)} student profiles")
        else:
            print(f"  ✗ Student profiles not found: {students_path}")
            self.students_df = None
        
        # Filter to future events only
        if self.events_df is not None and 'date' in self.events_df.columns:
            future_mask = self.events_df['date'] >= datetime.now()
            self.events_df = self.events_df[future_mask].copy()
            print(f"  ✓ Filtered to {len(self.events_df)} future events")
    
    def _create_sample_events(self):
        """Create sample events if scraping failed."""
        base_date = datetime.now()
        events_data = []
        
        sample_templates = [
            {'title': 'Wellness Wonderland', 'category': 'Health & Medicine', 
             'location': 'Sport and Fitness Center', 'keywords': ['wellness', 'fitness']},
            {'title': 'Hot Chocolate with the Chancellor', 'category': 'Special event',
             'location': 'Student Center East', 'keywords': ['social', 'networking']},
            {'title': 'Study with Snacks at the Library', 'category': 'Workshop',
             'location': 'Richard J. Daley Library', 'keywords': ['study', 'academic']},
            {'title': 'Digital Accessibility Training', 'category': 'Workshop',
             'location': 'Online', 'keywords': ['technology', 'training']},
            {'title': 'Free Webinar: Cybersecurity', 'category': 'Workshop',
             'location': 'Online', 'keywords': ['cybersecurity', 'technology']},
        ]
        
        for i, template in enumerate(sample_templates):
            events_data.append({
                'event_id': i + 1,
                'title': template['title'],
                'date': base_date + timedelta(days=i*3),
                'time': '12:00 pm - 1:00 pm',
                'location': template['location'],
                'category': template['category'],
                'description': f"{template['title']} - {template['category']} event",
                'keywords': template['keywords'],
                'is_online': 'Online' in template['location'],
                'source': 'sample'
            })
        
        self.events_df = pd.DataFrame(events_data)
        if 'date' in self.events_df.columns:
            self.events_df['date'] = pd.to_datetime(self.events_df['date'])
    
    def create_event_embeddings(self):
        """Create semantic embeddings for events."""
        if self.events_df is None or len(self.events_df) == 0:
            print("No events data available!")
            return
        
        print("\nCreating event embeddings...")
        
        # Create rich text representation for each event
        def create_event_text(row):
            parts = []
            
            # Title
            if pd.notna(row.get('title')):
                parts.append(str(row['title']))
            
            # Category
            if pd.notna(row.get('category')):
                parts.append(f"Category: {row['category']}")
            
            # Description
            if pd.notna(row.get('description')):
                parts.append(str(row['description']))
            
            # Keywords
            if pd.notna(row.get('keywords')):
                keywords = row['keywords']
                if isinstance(keywords, str):
                    # Try to parse if it's a string representation of list
                    try:
                        keywords = eval(keywords) if keywords.startswith('[') else keywords.split(',')
                    except:
                        keywords = keywords.split(',')
                if isinstance(keywords, list):
                    parts.extend([f"Interest: {kw}" for kw in keywords if kw])
            
            # Location context
            if pd.notna(row.get('location')):
                location = str(row['location'])
                if 'library' in location.lower():
                    parts.append("academic study")
                elif 'center' in location.lower():
                    parts.append("campus activity")
                elif 'online' in location.lower():
                    parts.append("virtual event")
            
            return ' '.join(parts)
        
        event_texts = self.events_df.apply(create_event_text, axis=1).tolist()
        
        # Generate embeddings
        print("  Encoding event texts...")
        self.event_embeddings = self.embedding_model.encode(
            event_texts, 
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        print(f"  ✓ Created embeddings: {self.event_embeddings.shape}")
        
        # Store event text features for reference
        self.event_text_features = event_texts
    
    def create_student_profile_embedding(self, student_profile: Dict) -> np.ndarray:
        """Create embedding for a single student profile."""
        parts = []
        
        # Major and academic info
        if 'Major' in student_profile or 'major' in student_profile:
            major = student_profile.get('Major') or student_profile.get('major', '')
            if major:
                parts.append(f"Major: {major}")
        
        if 'Year' in student_profile or 'year' in student_profile:
            year = student_profile.get('Year') or student_profile.get('year', '')
            if year:
                parts.append(f"Year: {year}")
        
        # Hobbies and interests
        if 'Hobbies' in student_profile or 'hobbies' in student_profile:
            hobbies = student_profile.get('Hobbies') or student_profile.get('hobbies', [])
            if hobbies:
                if isinstance(hobbies, str):
                    try:
                        hobbies = eval(hobbies) if hobbies.startswith('[') else [hobbies]
                    except:
                        hobbies = hobbies.split(',')
                if isinstance(hobbies, list):
                    parts.extend([f"Hobby: {hobby}" for hobby in hobbies if hobby])
        
        # Story/description
        if 'Story' in student_profile or 'story' in student_profile:
            story = student_profile.get('Story') or student_profile.get('story', '')
            if story:
                parts.append(str(story))
        
        # Unique quality
        if 'Unique Quality' in student_profile or 'unique_quality' in student_profile:
            quality = student_profile.get('Unique Quality') or student_profile.get('unique_quality', '')
            if quality:
                parts.append(str(quality))
        
        # Courses (relevant for academic events)
        if 'courses' in student_profile:
            courses = student_profile.get('courses', [])
            if courses and isinstance(courses, list):
                parts.extend([f"Course: {course}" for course in courses if course])
        
        # Combine all parts
        student_text = ' '.join(parts)
        
        # Generate embedding
        student_embedding = self.embedding_model.encode(
            [student_text],
            normalize_embeddings=True
        )[0]
        
        return student_embedding
    
    def recommend_events(self, 
                        student_profile: Dict,
                        top_k: int = 5,
                        category_filter: Optional[str] = None,
                        date_range_days: int = 30) -> List[Dict]:
        """
        Recommend events for a student profile.
        
        Args:
            student_profile: Dictionary with student information
            top_k: Number of recommendations
            category_filter: Optional category to filter by
            date_range_days: Only recommend events within N days
        
        Returns:
            List of recommended event dictionaries with scores
        """
        if self.event_embeddings is None:
            print("Event embeddings not created. Creating now...")
            self.create_event_embeddings()
        
        # Create student embedding
        student_embedding = self.create_student_profile_embedding(student_profile)
        student_embedding = student_embedding.reshape(1, -1)
        
        # Create boolean mask for filtering (keeps original row positions)
        mask = pd.Series(True, index=self.events_df.index)
        
        # Filter by date range
        if 'date' in self.events_df.columns:
            max_date = datetime.now() + timedelta(days=date_range_days)
            date_mask = self.events_df['date'] <= max_date
            mask = mask & date_mask
        
        # Filter by category
        if category_filter:
            category_mask = self.events_df['category'].str.contains(category_filter, case=False, na=False)
            mask = mask & category_mask
        
        if mask.sum() == 0:
            return []
        
        # Get filtered events DataFrame
        events_filtered = self.events_df[mask].copy()
        
        # Get embeddings for filtered events using boolean mask
        # embeddings[i] corresponds to events_df.iloc[i], so use mask for iloc positions
        filtered_embeddings = self.event_embeddings[mask.values]
        
        # Reset index for easier iteration
        events_filtered_df = events_filtered.reset_index(drop=True)
        
        # Calculate similarity
        similarities = cosine_similarity(student_embedding, filtered_embeddings)[0]
        
        # Add boost factors
        scores = similarities.copy()
        
        # Boost for matching categories with student interests
        if 'Hobbies' in student_profile or 'hobbies' in student_profile:
            hobbies = student_profile.get('Hobbies') or student_profile.get('hobbies', [])
            if isinstance(hobbies, str):
                try:
                    hobbies = eval(hobbies) if hobbies.startswith('[') else [hobbies]
                except:
                    hobbies = hobbies.split(',')
            
            if isinstance(hobbies, list):
                hobby_text = ' '.join([str(h) for h in hobbies]).lower()
                
                for idx, (_, event) in enumerate(events_filtered_df.iterrows()):
                    event_text = str(event.get('title', '') + ' ' + 
                                   event.get('description', '') + ' ' + 
                                   event.get('category', '')).lower()
                    
                    # Boost if hobbies match event content
                    hobby_matches = sum(1 for h in hobbies if h.lower() in event_text)
                    if hobby_matches > 0:
                        scores[idx] += 0.1 * hobby_matches
        
        # Boost for academic events if student is academically focused
        if 'Major' in student_profile or 'major' in student_profile:
            major = (student_profile.get('Major') or 
                    student_profile.get('major', '')).lower()
            
            academic_categories = ['workshop', 'lecture', 'conference']
            for idx, (_, event) in enumerate(events_filtered_df.iterrows()):
                event_category = str(event.get('category', '')).lower()
                if any(ac in event_category for ac in academic_categories):
                    # Boost academic events slightly
                    scores[idx] += 0.05
        
        # Get top-k recommendations
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        recommendations = []
        # Get the original indices that passed the filter
        original_filtered_indices = events_filtered.index.tolist()
        
        for idx in top_indices:
            if idx < len(events_filtered_df):
                # Get the event from filtered DataFrame
                event = events_filtered_df.iloc[idx].to_dict()
                event['similarity_score'] = float(scores[idx])
                event['rank'] = len(recommendations) + 1
                recommendations.append(event)
        
        return recommendations
    
    def recommend_for_multiple_students(self, 
                                       student_profiles: List[Dict],
                                       top_k: int = 5) -> Dict[str, List[Dict]]:
        """Recommend events for multiple students."""
        results = {}
        
        for student in student_profiles:
            name = student.get('Name') or student.get('name', 'Unknown')
            recommendations = self.recommend_events(student, top_k=top_k)
            results[name] = recommendations
        
        return results
    
    def save_model(self, filepath: Optional[str] = None):
        """Save the trained model and embeddings."""
        if filepath is None:
            filepath = RESULTS_DIR / 'event_recommendation_model.pkl'
        
        model_data = {
            'event_embeddings': self.event_embeddings,
            'events_df': self.events_df,
            'embedding_model_name': 'all-MiniLM-L6-v2'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Saved model to {filepath}")
        
        # Also save events DataFrame
        events_path = RESULTS_DIR / 'events_for_recommendation.csv'
        self.events_df.to_csv(events_path, index=False)
        print(f"✓ Saved events to {events_path}")
    
    def visualize_recommendations(self, 
                                 student_name: str,
                                 recommendations: List[Dict],
                                 save_path: Optional[str] = None):
        """Visualize event recommendations."""
        if not recommendations:
            print("No recommendations to visualize")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Similarity scores
        scores = [r['similarity_score'] for r in recommendations]
        titles = [r['title'][:30] + '...' if len(r['title']) > 30 else r['title'] 
                 for r in recommendations]
        
        axes[0].barh(range(len(scores)), scores, color='steelblue')
        axes[0].set_yticks(range(len(titles)))
        axes[0].set_yticklabels(titles)
        axes[0].set_xlabel('Similarity Score')
        axes[0].set_title(f'Event Recommendations for {student_name}')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Plot 2: Category distribution
        categories = [r.get('category', 'Unknown') for r in recommendations]
        category_counts = pd.Series(categories).value_counts()
        
        axes[1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[1].set_title('Recommended Event Categories')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = RESULTS_DIR / f'event_recommendations_{student_name.replace(" ", "_")}.png'
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
        plt.close()


def main():
    """Main function to demonstrate event recommendations."""
    print("="*60)
    print(" UIC EVENT RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Initialize engine
    engine = EventRecommendationEngine()
    
    # Load data
    engine.load_data()
    
    # Create event embeddings
    engine.create_event_embeddings()
    
    # Load sample student profiles
    students_path = PROCESSED_DATA_DIR / 'profiles_processed.csv'
    if students_path.exists():
        students_df = pd.read_csv(students_path)
        print(f"\nLoaded {len(students_df)} student profiles for testing")
        
        # Test with a few students
        sample_students = students_df.head(3).to_dict('records')
        
        print("\n" + "="*60)
        print(" GENERATING RECOMMENDATIONS")
        print("="*60)
        
        for student in sample_students:
            name = student.get('Name', 'Unknown')
            print(f"\n{'='*60}")
            print(f"Recommendations for: {name}")
            print(f"Major: {student.get('Major', 'N/A')}")
            print(f"Hobbies: {student.get('Hobbies', 'N/A')}")
            print(f"{'='*60}")
            
            recommendations = engine.recommend_events(student, top_k=5)
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{i}. {rec['title']}")
                    print(f"   Category: {rec.get('category', 'N/A')}")
                    print(f"   Date: {rec.get('date', 'N/A')}")
                    print(f"   Location: {rec.get('location', 'N/A')}")
                    print(f"   Similarity Score: {rec['similarity_score']:.3f}")
                
                # Visualize
                engine.visualize_recommendations(name, recommendations)
            else:
                print("  No recommendations found")
    
    # Save model
    print("\n" + "="*60)
    engine.save_model()
    
    print("\n" + "="*60)
    print(" EVENT RECOMMENDATION SYSTEM READY!")
    print("="*60)


if __name__ == "__main__":
    main()

