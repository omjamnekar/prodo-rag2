from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key) # type: ignore

models = genai.list_models()  # type: ignore # List all models
for i in models:
    print(i.name)

