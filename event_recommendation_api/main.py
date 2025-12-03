from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path
import pandas as pd
import importlib.util

# Import the recommendation engine from the notebook
PROJECT_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "event_recommendation",
    PROJECT_ROOT / "notebooks" / "15_event_recommendation.py"
)
event_recommendation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(event_recommendation_module)
EventRecommendationEngine = event_recommendation_module.EventRecommendationEngine

app = FastAPI(title="UIC Event Recommendation API")

# Initialize engine (lazy loading)
engine = None

def get_engine():
    """Get or initialize the recommendation engine."""
    global engine
    if engine is None:
        print("Initializing Event Recommendation Engine...")
        engine = EventRecommendationEngine()
        engine.load_data()
        engine.create_event_embeddings()
        print("Engine ready!")
    return engine

# Define Input Data Model
class StudentProfile(BaseModel):
    name: str
    age: Optional[int] = None
    sex: Optional[str] = None
    major: Optional[str] = None
    year: Optional[str] = None
    gpa: Optional[float] = None
    unique_quality: Optional[str] = None
    story: Optional[str] = None
    hobbies: Optional[List[str]] = []
    courses: Optional[List[str]] = []

class EventRecommendationRequest(BaseModel):
    student: StudentProfile
    top_k: Optional[int] = 5
    category_filter: Optional[str] = None
    date_range_days: Optional[int] = 30

@app.get("/")
def home():
    return {
        "status": "active",
        "message": "UIC Event Recommendation API",
        "endpoints": {
            "recommend": "/recommend",
            "events": "/events/list"
        }
    }

@app.post("/recommend")
def recommend_events(request: EventRecommendationRequest) -> Dict[str, Any]:
    """
    Get event recommendations for a student.
    
    Example request:
    {
        "student": {
            "name": "John Doe",
            "major": "Computer Science",
            "hobbies": ["photography", "coding"],
            "year": "Junior"
        },
        "top_k": 5,
        "category_filter": null,
        "date_range_days": 30
    }
    """
    try:
        # Get engine
        rec_engine = get_engine()
        
        # Convert student profile to dict
        student_dict = request.student.dict(exclude_none=True)
        
        # Get recommendations
        recommendations = rec_engine.recommend_events(
            student_profile=student_dict,
            top_k=request.top_k,
            category_filter=request.category_filter,
            date_range_days=request.date_range_days
        )
        
        # Format response
        formatted_recommendations = []
        for rec in recommendations:
            formatted_rec = {
                "title": rec.get('title', ''),
                "category": rec.get('category', ''),
                "date": str(rec.get('date', '')) if rec.get('date') else None,
                "time": rec.get('time', ''),
                "location": rec.get('location', ''),
                "description": rec.get('description', ''),
                "similarity_score": rec.get('similarity_score', 0.0),
                "rank": rec.get('rank', 0)
            }
            formatted_recommendations.append(formatted_rec)
        
        return {
            "student_name": request.student.name,
            "recommendations_count": len(formatted_recommendations),
            "recommendations": formatted_recommendations
        }
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/events/list")
def list_events(
    category: Optional[str] = None,
    limit: Optional[int] = 20
) -> Dict[str, Any]:
    """List available events (future events only)."""
    try:
        rec_engine = get_engine()
        
        if rec_engine.events_df is None:
            raise HTTPException(status_code=500, detail="Events data not loaded")
        
        events_df = rec_engine.events_df.copy()
        
        # Filter by category if provided
        if category:
            events_df = events_df[
                events_df['category'].str.contains(category, case=False, na=False)
            ]
        
        # Limit results
        events_df = events_df.head(limit)
        
        # Format events
        events_list = []
        for _, row in events_df.iterrows():
            event = {
                "title": row.get('title', ''),
                "category": row.get('category', ''),
                "date": str(row.get('date', '')) if pd.notna(row.get('date')) else None,
                "time": row.get('time', ''),
                "location": row.get('location', ''),
                "description": row.get('description', '')
            }
            events_list.append(event)
        
        return {
            "total_events": len(events_list),
            "events": events_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --port 8002 --reload
