from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from engine import FriendFinderEngine

app = FastAPI(title="UIC Friend Finder API")
# Initialize Engine
engine = FriendFinderEngine()

class FriendRequest(BaseModel):
    name: str
    age: int
    sex: str
    major: str
    year: str
    gpa: float
    unique_quality: str
    story: str
    hobbies: List[str] # Essential for this API
    
    # Optional fields (just in case frontend sends them)
    courses: List[str] = []
    free_slots: List[str] = []

@app.get("/")
def home():
    return {"status": "Friend Finder Online"}

@app.post("/find_friends")
def find_friends(profile: FriendRequest):
    try:
        user_data = profile.dict()
        results = engine.find_friends(user_data)
        
        return {
            "target_student": profile.name,
            "type": "Social Match",
            "recommendations": results
        }
    except Exception as e:
        # Print error to console for debugging
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run on PORT 8001
# Command: uvicorn main:app --port 8001 --reload