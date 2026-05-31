
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln


TARGET_SR = 16000
TARGET_LOUDNESS = -14.0


def preprocess_audio(
    input_path: str,
    output_path: str,
    apply_noise_reduction: bool = True,
):
    """
    Full preprocessing pipeline:
    - mono conversion
    - resampling to 16kHz
    - noise reduction
    - loudness normalization
    - clipping prevention
    """

    # Load audio
    audio, sr = librosa.load(
        input_path,
        sr=TARGET_SR,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("Audio contains no samples")

    # Noise reduction
    if apply_noise_reduction:
        audio = nr.reduce_noise(
            y=audio,
            sr=sr
        )

    # Loudness normalization
    meter = pyln.Meter(sr)

    loudness = meter.integrated_loudness(audio)

    normalized_audio = pyln.normalize.loudness(
        audio,
        loudness,
        TARGET_LOUDNESS
    )

    # Prevent clipping
    normalized_audio = np.clip(
        normalized_audio,
        -1.0,
        1.0
    )

    # Save processed audio
    sf.write(
        output_path,
        normalized_audio,
        sr,
        format="WAV",
        subtype="PCM_16"
    )

    return output_path

