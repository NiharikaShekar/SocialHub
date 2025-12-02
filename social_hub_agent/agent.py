from google.adk.agents.llm_agent import Agent

from typing import List, Dict, Any
import requests


STUDY_BUDDY_API_URL = "http://127.0.0.1:8000/recommend/study_buddy"


def find_study_buddy(
    name: str,
    age: int,
    sex: str,
    major: str,
    year: str,
    gpa: float,
    unique_quality: str,
    story: str,
    courses: List[str],
    free_slots: List[str],
) -> Dict[str, Any]:
    """
    Call the Social Hub FastAPI backend to get study buddy recommendations.

    REQUIRED FIELDS (the agent MUST collect these from the user via conversation):
    - name: full name of the student (string)
    - age: age in years (integer)
    - sex: 'Male', 'Female', 'Non-binary', etc. (string)
    - major: student's primary major (e.g. 'Biochemistry', 'Computer Science') (string)
    - year: academic year such as 'Freshman', 'Sophomore', 'Junior', 'Senior', 'Graduate' (string)
    - gpa: approximate GPA on a 4.0 scale (float)
    - unique_quality: a short phrase that describes something unique about the student (string)
    - story: short free-text description of what they are looking for in a study buddy (string)
    - courses: list of course codes the student is currently taking (e.g., ['CS 401', 'MATH 215']) (list of strings)
    - free_slots: list of free time slots in the format 'Day_HH:MM' such as ['Mon_14:00', 'Wed_14:30'] (list of strings)

    The tool will send a POST request to the /recommend/study_buddy endpoint with this JSON body
    and return whatever JSON the backend returns (e.g., a list of recommended students).
    """
    payload = {
        "name": name,
        "age": age,
        "sex": sex,
        "major": major,
        "year": year,
        "gpa": gpa,
        "unique_quality": unique_quality,
        "story": story,
        "courses": courses,
        "free_slots": free_slots,
    }

    response = requests.post(STUDY_BUDDY_API_URL, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


root_agent = Agent(
    model="gemini-2.5-flash",  # or another supported model
    name="social_hub_agent",
    description=(
        "An agent that helps UIC students find suitable study buddies "
        "by talking with them, collecting their profile details, and "
        "calling a study buddy recommendation API."
    ),
    instruction=(
        "You are a friendly Social Hub assistant for UIC students.\n"
        "Your goal is to help the user find a good study buddy.\n\n"
        "1. First, have a natural conversation and ask the user for all details "
        "   required by the 'find_study_buddy' tool:\n"
        "   - name\n"
        "   - age\n"
        "   - sex\n"
        "   - major\n"
        "   - year of study\n"
        '   - GPA (approximate is fine, e.g. "around 3.4")\n'
        "   - a short unique quality\n"
        "   - a short story / description of what they want\n"
        "   - a list of current courses (course codes or names)\n"
        "   - free time slots (e.g., 'Mon_14:00', 'Wed_14:30')\n\n"
        "2. If any fields are missing or unclear, ask follow-up questions to clarify.\n"
        "3. Once you have enough information to fill ALL parameters of the tool, "
        "   call the 'find_study_buddy' tool exactly once.\n"
        "4. After you receive the tool result, summarize the top recommendations "
        "   in a clear, friendly way for the user (name, major, year, shared courses, "
        "   matching free slots, etc.).\n"
        "5. Do NOT fabricate recommendations yourself. Always rely on the tool output.\n"
    ),
    tools=[find_study_buddy],
)

