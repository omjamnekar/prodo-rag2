import requests
import json
import os

# Test data
files = [
    {"filename": "main.py", "content": "def foo():\n    return 42"},
    {"filename": "vercel.json", "content": "def bar(x):\n    return x * 2"}
]
prompt = "How can I improve the code quality?"
metadata = {"provider": "github", "remoteId": "1025382709", "title": "numpy"}

# API endpoint (use environment variable or default to localhost)
url = os.getenv("API_URL", "http://127.0.0.1:5000/rag/index")

try:
    response = requests.post(url, json={
        "repoId": "test-repo-123",
        "files": files,
        "prompt": prompt,
        "metadata": metadata
    })
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    print("Status:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=4))
except requests.exceptions.RequestException as e:
    print("Error:", e)
