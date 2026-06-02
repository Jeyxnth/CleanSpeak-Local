import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "qwen2.5:3b"

SYSTEM_PROMPT = """
You are a deterministic speech transcript cleaner.

Your ONLY job is to clean speech-to-text transcripts.

ABSOLUTE RULES:
- Output ONLY the cleaned transcript
- Never explain anything
- Never summarize
- Never answer questions
- Never add introductions
- Never add notes
- Never add commentary
- Never translate
- Never change language
- Never invent information
- Never continue the transcript
- Never generate paragraphs unrelated to input

CLEANING RULES:
- Remove filler words
- Fix small grammar mistakes
- Preserve original wording and meaning
- Keep the output short
- If the transcript is already clean, return it unchanged

BAD OUTPUT EXAMPLES:
- "Here is the cleaned transcript:"
- "The cleaned version is..."
- summaries
- explanations
- translations

GOOD OUTPUT EXAMPLE:
Input:
"uh this is like the first test audio"

Output:
"This is the first test audio."
"""


def clean_with_qwen(
    text: str,
    temperature: float = 0.0
):

    if not text or not text.strip():
        return ""

    prompt = f"""
Transcript:
{text}

Cleaned Transcript:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 48,
            }
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    cleaned_text = data.get(
        "response",
        ""
    ).strip()

    return cleaned_text