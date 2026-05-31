
import os

# Force model/cache storage to D drive
os.environ["HF_HOME"] = r"D:\AI\huggingface"
os.environ["NEMO_CACHE_DIR"] = r"D:\AI\models"

import nemo.collections.asr as nemo_asr
import torch


print("Loading NVIDIA Parakeet model...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

if DEVICE == "cuda":
    model = model.cuda()

print(f"Parakeet loaded successfully on {DEVICE}")


def preload_transcriber(
    model_name: str | None = None,
    language: str = "en"
) -> str:
    """
    Preload model during FastAPI startup.
    """
    return "parakeet-tdt-0.6b-v2"


def transcribe(
    audio_path: str,
    model_name: str | None = None,
    language: str = "en"
) -> str:
    """
    Transcribe audio using NVIDIA Parakeet.
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    result = model.transcribe([audio_path])

    if isinstance(result, list):
        transcript = result[0]
    else:
        transcript = str(result)

    transcript = " ".join(
        transcript.strip().split()
    )

    return transcript

