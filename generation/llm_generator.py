"""
Handles answer generation using the Google Gemini model.
"""

import os
import google.generativeai as genai 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

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

    response=model.generate_content(prompt)

    return response.text

if __name__=="__main__":
        prompt="""
        Context:
        LoRA is a parameter efficient fine tuning technique.

        Question:
        What is LoRA?

        Answer:
        """



        answer=generate_answer(prompt)

        print(answer)