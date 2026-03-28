import os
import json
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GENAI_API_KEY)

# Define expected keys and default values
EXPECTED_KEYS = {
    "keyinsights": None,
    "strengths": None,
    "weaknesses": None,
    "recommendation": None,
    "rating": 0,
    "grade": 0
}


def analyze_student_solution(exam_text: str, student_text: str) -> dict:
    """
    Compare student's solution with the exam solution using Gemini.
    Returns a validated JSON object with all expected fields.
    """
    prompt = f"""
You are an expert examiner. You will be given a correct exam solution and a student's solution.
Both texts may have OCR errors, missing spaces, or typos.

Your tasks:
1. Understand both texts.
2. Compare the student's solution against the exam solution.
3. Identify what the student did well and what is missing.
4. Give a clear structured evaluation.

EXAM SOLUTION:
{exam_text}

STUDENT SOLUTION:
{student_text}

Return only JSON in this exact format:

{{
  "keyinsights": "",
  "strengths": "",
  "weaknesses": "",
  "recommendation": "",
  "rating": 0,
  "grade": 0
}}
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    # Parse the response safely
    try:
        data = json.loads(response.text)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    # Ensure all keys exist
    for key, default in EXPECTED_KEYS.items():
        if key not in data:
            data[key] = default

    return data
