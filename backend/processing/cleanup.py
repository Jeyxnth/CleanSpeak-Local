import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "sarvam-1"


SYSTEM_PROMPT = """
You are a transcript cleanup assistant.

Your tasks:
- remove filler words
- remove repeated words/stutters
- improve punctuation
- fix grammar lightly
- preserve original meaning
- do NOT summarize
- do NOT change intent
- do NOT add new information
- return only cleaned transcript
"""


def clean_with_sarvam(
    text: str,
    temperature: float = 0.2
):
    """
    Clean transcript using local Sarvam model via Ollama.
    """

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
                "num_predict": 512
            }
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    cleaned_text = data.get("response", "").strip()

    return cleaned_text