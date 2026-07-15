"""
Handles answer generation using the Google Gemini model.
"""

import os
import google.generativeai as genai 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key=os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found.")
genai.configure(api_key=api_key)


# Initialize LLM
model=genai.GenerativeModel(
    "gemini-2.5-flash"
)


def generate_answer(prompt):
    """
    Generate an answer from the LLM.

    Args:
        prompt (str):
            Fully constructed prompt.

    Returns:
        str:
            Generated answer.
    """
    try:
        response=model.generate_content(
            prompt,
            generation_config={
                "temperature":0.0,
            }
        )

        return response.text

    except  Exception as e:
        raise RuntimeError(f"Gemini Generation failed:{e}")

