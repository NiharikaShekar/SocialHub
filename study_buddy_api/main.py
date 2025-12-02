from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from engine import engine

app = FastAPI(title="UIC Study Buddy API")

# Define Input Data Model
class StudentProfile(BaseModel):
    name: str
    age: int
    sex: str
    major: str
    year: str
    gpa: float
    unique_quality: str
    story: str
    courses: List[str]     # e.g. ["CS 411", "MATH 180"]
    free_slots: List[str]  # e.g. ["Mon_14:00", "Fri_10:30"]

@app.get("/")
def home():
    return {"status": "active", "message": "Send POST request to /recommend/study_buddy"}

@app.post("/recommend/study_buddy")
def get_study_buddy(profile: StudentProfile):
    """
    Finds study partners for a NEW student based on Course Overlap > Schedule > Vibe.
    """
    try:
        # 1. Convert to dictionary
        user_data = profile.dict()
        
        # 2. Run the Engine
        results = engine.recommend_for_new_student(user_data)
        
        # 3. Return JSON
        return {
            "target_student": profile.name,
            "recommendations": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload