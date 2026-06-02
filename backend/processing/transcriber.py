
import os
import traceback

print("STEP 1: Setting cache paths")

os.environ["HF_HOME"] = r"D:\AI\huggingface"
os.environ["NEMO_CACHE_DIR"] = r"D:\AI\models"

print("STEP 2: Importing torch")

import torch

print("CUDA AVAILABLE:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("STEP 3: Importing NeMo")

import nemo.collections.asr as nemo_asr

print("STEP 4: Loading Parakeet model")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:

    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )

    print("MODEL DOWNLOADED")

    if DEVICE == "cuda":
        model = model.cuda()

    print(f"MODEL MOVED TO {DEVICE}")

except Exception as e:

    print("MODEL LOAD FAILED")
    traceback.print_exc()

    raise e

print("STEP 5: Parakeet loaded successfully")


def preload_transcriber(
    model_name=None,
    language="en"
):
    return "parakeet"


def transcribe(
    audio_path,
    model_name=None,
    language="en"
):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    result = model.transcribe([audio_path])

    # Extract transcript safely
    if isinstance(result, list):

        first = result[0]

        # NeMo Hypothesis object
        if hasattr(first, "text"):
            transcript = first.text

        else:
            transcript = str(first)

    else:

        if hasattr(result, "text"):
            transcript = result.text
        else:
            transcript = str(result)

    transcript = " ".join(
        transcript.strip().split()
    )

    return transcript


