import requests
import json

# Note the PORT is 8001
url = "http://127.0.0.1:8001/find_friends"

# A user looking for social connections (Hobbies match)
new_profile =  {
    "name": "Sarah Social",
    "age": 22,
    "sex": "Female",
    "major": "Psychology",
    "year": "Senior",
    "gpa": 3.4,
    "unique_quality": "Loves live music and road trips",
    "story": "I am looking for people to go to concerts with on weekends.",
    "hobbies": ['photography', 'rock climbing'],
}

try:
    print(f"📡 Sending request to {url}...")
    response = requests.post(url, json=new_profile)

    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("\n✅ API RESPONSE:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("\n❌ ERROR RESPONSE:")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\n❌ CONNECTION ERROR: Is the server running?")
    print("Run: uvicorn friend_main:app --port 8001 --reload")