import requests
import json

url = "http://127.0.0.1:8000/recommend/study_buddy"

new_student = {
    "name": "Alice Beckerr",
    "age": 20,
    "sex": "Female",
    "major": "Biochemistry",
    "year": "Junior",
    "gpa": 3.5,
    "unique_quality": "Loves lab work",
    "story": "Looking for study partners.",
    "courses": ["CHEM 232: Organic Chemistry I", "BIOS 110: Biology of Cells"], 
    "free_slots": ["Mon_14:00", "Mon_14:30", "Wed_14:00"]
}

response = requests.post(url, json=new_student)

# --- DEBUGGING BLOCK ---
print(f"Status Code: {response.status_code}")
print("Raw Response Text:")
print(response.text)  # <--- This will show you the ACTUAL error from the server
# -----------------------

if response.status_code == 200:
    print(json.dumps(response.json(), indent=2))
else:
    print("Request failed.")