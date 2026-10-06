import os
import requests
from dotenv import load_dotenv

load_dotenv()

r = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
    timeout=30,
)
print("STATUS:", r.status_code)
for m in sorted(r.json().get("data", []), key=lambda x: x["id"]):
    print(m["id"])