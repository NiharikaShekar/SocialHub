from google.adk.agents.llm_agent import Agent

from typing import List, Dict, Any
import requests

# ---------------- API URLs ----------------
STUDY_BUDDY_API_URL = "http://127.0.0.1:8000/recommend/study_buddy"
FRIEND_API_URL      = "http://127.0.0.1:8001/find_friends"


# ---------------- TOOL 1: Study Buddy ----------------
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
    Call the Social Hub FastAPI backend to get STUDY BUDDY recommendations.

    Use this tool when the user is specifically looking for a *study buddy*
    (someone to study with / work on courses together).

    REQUIRED FIELDS (collect these via conversation):
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

    The tool sends a POST request to /recommend/study_buddy and returns the JSON response.
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


# ---------------- TOOL 2: Friend Finder ----------------
def find_friend(
    name: str,
    age: int,
    sex: str,
    major: str,
    year: str,
    gpa: float,
    unique_quality: str,
    story: str,
    hobbies: List[str],
) -> Dict[str, Any]:
    """
    Call the Social Hub FastAPI backend to get FRIEND recommendations.

    Use this tool when the user is looking for *friends / social connections*
    (people with similar hobbies, interests, vibes) rather than a strict study partner.

    REQUIRED FIELDS (collect these via conversation):
    - name: full name of the student (string)
    - age: age in years (integer)
    - sex: 'Male', 'Female', 'Non-binary', etc. (string)
    - major: student's primary major (string)
    - year: academic year such as 'Freshman', 'Sophomore', 'Junior', 'Senior', 'Graduate' (string)
    - gpa: approximate GPA on a 4.0 scale (float)
    - unique_quality: a short phrase that describes something unique about the student (string)
    - story: short free-text description of what they want in a friend (string)
    - hobbies: list of hobbies/interests (e.g., ['photography', 'rock climbing']) (list of strings)

    The tool sends a POST request to /find_friends on port 8001 and returns the JSON response.
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
        "hobbies": hobbies,
    }

    response = requests.post(FRIEND_API_URL, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


# ---------------- ROOT AGENT ----------------
root_agent = Agent(
    model="gemini-2.5-flash",
    name="social_hub_agent",
    description=(
        "An agent that helps UIC students find suitable study buddies or friends "
        "by talking with them, collecting their profile details, and calling "
        "the appropriate recommendation API."
    ),
    instruction=(
        "You are a friendly Social Hub assistant for UIC students.\n"
        "Your goal is to help the user either:\n"
        "  (a) find a good study buddy, or\n"
        "  (b) find new friends / social connections.\n\n"
        "1. First, ask the user what they are looking for:\n"
        "   - If they mention studying, homework, projects, exams, or courses, "
        "     prefer the 'find_study_buddy' tool.\n"
        "   - If they mention hanging out, making friends, hobbies, or socializing, "
        "     prefer the 'find_friend' tool.\n\n"
        "2. Once you know which tool to use, have a natural conversation and collect "
        "   ALL required fields for that specific tool:\n"
        "   COMMON FIELDS (both tools): name, age, sex, major, year, GPA, "
        "   unique quality, short story/description.\n"
        "   - For 'find_study_buddy': also collect current courses and free time slots.\n"
        "   - For 'find_friend': also collect hobbies/interests.\n\n"
        "3. If any fields are missing or unclear, ask follow-up questions to clarify.\n"
        "4. Once you have enough information to fill all parameters of the selected tool, "
        "   call that tool exactly once.\n"
        "5. After you receive the tool result, summarize the top recommendations "
        "   in a clear, friendly way for the user (name, major, year, relevant overlaps "
        "   in courses or hobbies, etc.).\n"
        "6. Do NOT invent or fabricate recommendations yourself. Always rely on the tool output.\n"
    ),
    tools=[find_study_buddy, find_friend],
)
