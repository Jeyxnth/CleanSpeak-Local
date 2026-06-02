import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "qwen2.5:3b"

SYSTEM_PROMPT = """
You clean speech-to-text transcripts.

Rules:
- Return ONLY the cleaned transcript
- Do not explain anything
- Do not summarize
- Do not add extra text
- Do not translate
- Preserve original meaning
- Remove filler words
- Fix minor grammar mistakes
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
                "num_predict": 64,
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