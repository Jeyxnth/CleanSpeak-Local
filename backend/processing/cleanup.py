import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "qwen2.5:3b"

SYSTEM_PROMPT = """
You are a speech-to-text transcript cleaner.

Your task is ONLY to minimally clean transcripts.

STRICT RULES:
- Preserve ALL information from the original transcript
- Do NOT shorten the transcript
- Do NOT summarize
- Do NOT compress content
- Do NOT remove sentences
- Do NOT omit details
- Do NOT invent new text
- Do NOT translate
- Do NOT explain anything
- Return ONLY the cleaned transcript

ALLOWED CHANGES:
- Remove filler words like:
  um, uh, like, you know
- Fix tiny grammar mistakes
- Fix punctuation
- Fix capitalization

IMPORTANT:
The cleaned transcript should contain nearly the SAME number of words as the original transcript.

If the transcript is already clean:
RETURN IT UNCHANGED.

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
                "num_predict": 512,
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