import os
import json
import google.generativeai as genai


api_key = os.getenv('GEMINI_API_KEY')
MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')

def generate_from_gemini(prompt: str) -> dict:
    """
    Generate content using Gemini API.
    Only uses the required packages for this functionality.
    """
    genai.configure(api_key=api_key) # type: ignore
    model = genai.GenerativeModel(MODEL) # type: ignore
    resp = model.generate_content(prompt)
    raw = ''
    try:
        if hasattr(resp, 'candidates'):
            raw = '\n'.join([str(c.text) if hasattr(c, 'text') else str(c) for c in resp.candidates])
        elif hasattr(resp, 'text'):
            raw = str(resp.text)
        else:
            raw = str(resp)
    except Exception:
        raw = str(resp)

    parsed = None
    try:
        start = raw.index('{')
        candidate = raw[start:]
        parsed = json.loads(candidate)
    except Exception:
        parsed = None

    return {'raw': raw, 'json': parsed}