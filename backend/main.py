import os
import uuid
import shutil
import logging

import librosa
import soundfile as sf

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from processing.transcriber import (
    transcribe,
    preload_transcriber
)

from processing.preprocess import preprocess_audio

from processing.cleanup import clean_with_qwen

from processing.word_filter import (
    filter_words,
    get_filler_words,
    get_flagged_words,
    add_flagged_word
)


UPLOAD_DIR = "backend/temp/uploads"
ORIGINAL_DIR = "backend/temp/original"
CLEANED_DIR = "backend/temp/cleaned"
LEGACY_OUTPUT_DIR = "backend/temp/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(LEGACY_OUTPUT_DIR, exist_ok=True)


app = FastAPI()

logger = logging.getLogger("clean_speech.pipeline")

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s"
        )
    )
    logger.addHandler(handler)

logger.setLevel(logging.INFO)
logger.propagate = False


@app.on_event("startup")
async def preload_transcriber_model() -> None:
    try:
        model = preload_transcriber(language="en")

        logger.info(
            "transcriber.preloaded model=%s",
            model
        )

    except Exception:
        logger.exception(
            "transcriber.preload_failed"
        )


ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://localhost:8081",
    os.getenv(
        "FRONTEND_URL",
        "https://clean-speak-tau.vercel.app/"
    ),
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


AUDIO_EXTENSIONS = {
    "mp3",
    "wav",
    "ogg",
    "m4a",
    "webm"
}


def _normalize_mode(mode: str) -> str:
    normalized = (mode or "normal").strip().lower()

    if normalized not in {"normal", "clear"}:
        raise HTTPException(
            status_code=400,
            detail="mode must be either 'normal' or 'clear'."
        )

    return normalized


def _load_as_wav(
    input_path: str,
    output_path: str,
    target_sr: int = 16000
) -> None:

    audio, sr = librosa.load(
        input_path,
        sr=target_sr,
        mono=True
    )

    if audio.size == 0:
        raise ValueError(
            "Uploaded audio has no samples"
        )

    sf.write(
        output_path,
        audio,
        sr,
        format="WAV",
        subtype="PCM_16"
    )


def _safe_audio_filename(filename: str) -> str:
    safe_name = os.path.basename(filename)

    if (
        safe_name != filename or
        not safe_name.lower().endswith(".wav")
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid audio filename"
        )

    return safe_name


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    mode: str = Form("normal"),
    apply_noise_reduction: bool = Form(True),
    language: str = Form("en"),
):

    ext = file.filename.split(".")[-1].lower()

    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format."
        )

    selected_mode = _normalize_mode(mode)

    job_id = str(uuid.uuid4())

    upload_path = os.path.join(
        UPLOAD_DIR,
        f"{job_id}.{ext}"
    )

    original_filename = f"{job_id}.wav"

    original_path = os.path.join(
        ORIGINAL_DIR,
        original_filename
    )

    processed_path = os.path.join(
        CLEANED_DIR,
        f"{job_id}_processed.wav"
    )

    # Save uploaded file
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Convert to WAV
        _load_as_wav(
            upload_path,
            original_path,
            target_sr=16000
        )

        # Audio preprocessing
        preprocess_audio(
            original_path,
            processed_path,
            apply_noise_reduction=apply_noise_reduction
        )

    except Exception as e:

        if os.path.exists(upload_path):
            os.remove(upload_path)

        logger.exception(
            "audio.preprocessing_failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Audio preprocessing failed: {e}"
        )

    logger.info(
        "transcription.start job_id=%s language=%s",
        job_id,
        language
    )

    # STT transcription
    try:

        raw_transcript = transcribe(
            processed_path,
            language=language
        )

    except Exception as e:

        logger.exception(
            "transcription.failed job_id=%s",
            job_id
        )

        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {e}"
        )

    # LLM cleanup
    try:

        if selected_mode == "clear":

            qwen_cleaned = clean_with_qwen(
                raw_transcript
            )

            cleaned = filter_words(
                qwen_cleaned
            )

        else:

            cleaned = filter_words(
                raw_transcript
            )

    except Exception as e:

        logger.exception(
            "cleanup.failed job_id=%s",
            job_id
        )

        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {e}"
        )

    # Remove upload temp
    if os.path.exists(upload_path):
        os.remove(upload_path)

    cleaned_transcript = cleaned["cleaned_transcript"]

    logger.info(
        "transcription.complete job_id=%s raw_words=%s cleaned_words=%s",
        job_id,
        len(raw_transcript.split()),
        len(cleaned_transcript.split())
    )

    original_audio_url = (
        f"/audio/original/{original_filename}"
    )

    response = {
        "job_id": job_id,

        "mode": selected_mode,

        "language": language,

        "noise_applied": apply_noise_reduction,

        "model_used": "parakeet-tdt-0.6b-v2",

        "transcript": raw_transcript,

        "raw_transcript": raw_transcript,

        "cleaned_transcript": cleaned_transcript,

        "removed_words": cleaned["removed_words"],

        "fillers_removed": cleaned["filler_count"],

        "words_censored": cleaned["flagged_count"],

        "original_word_count": len(
            raw_transcript.split()
        ),

        "cleaned_word_count": len(
            cleaned_transcript.split()
        ),

        "original_audio_url": original_audio_url,

        "processed_audio_path": processed_path
    }

    return response


@app.get("/download/{job_id}")
def download_audio(job_id: str):

    processed_path = os.path.join(
        CLEANED_DIR,
        f"{job_id}_processed.wav"
    )

    if os.path.exists(processed_path):

        return FileResponse(
            processed_path,
            media_type="audio/wav",
            filename=f"processed_{job_id}.wav"
        )

    raise HTTPException(
        status_code=404,
        detail="Processed audio not found."
    )


@app.get("/audio/original/{filename}")
def get_original_audio(filename: str):

    safe_name = _safe_audio_filename(
        filename
    )

    original_path = os.path.join(
        ORIGINAL_DIR,
        safe_name
    )

    if not os.path.exists(original_path):

        raise HTTPException(
            status_code=404,
            detail="Original audio not found."
        )

    return FileResponse(
        original_path,
        media_type="audio/wav",
        filename=safe_name
    )


@app.get("/audio/processed/{filename}")
def get_processed_audio(filename: str):

    safe_name = _safe_audio_filename(
        filename
    )

    processed_path = os.path.join(
        CLEANED_DIR,
        safe_name
    )

    if not os.path.exists(processed_path):

        raise HTTPException(
            status_code=404,
            detail="Processed audio not found."
        )

    return FileResponse(
        processed_path,
        media_type="audio/wav",
        filename=safe_name
    )


@app.get("/words/filler")
def list_filler_words():

    return {
        "filler_words": get_filler_words()
    }


@app.get("/words/flagged")
def list_flagged_words():

    return {
        "flagged_words": get_flagged_words()
    }


@app.post("/words/flagged/add")
def add_flagged(word: str):

    if not word or not word.strip():

        raise HTTPException(
            status_code=400,
            detail="Word must not be empty."
        )

    add_flagged_word(
        word.strip()
    )

    return {
        "flagged_words": get_flagged_words()
    }

