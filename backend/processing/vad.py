
import webrtcvad
import collections
import numpy as np


# Aggressiveness:
# 0 = least aggressive
# 3 = most aggressive
vad = webrtcvad.Vad(2)


def is_speech(frame: bytes, sample_rate: int = 16000) -> bool:
    """
    Check whether a frame contains speech.
    """
    return vad.is_speech(frame, sample_rate)


def frame_generator(
    audio: bytes,
    frame_duration_ms: int,
    sample_rate: int
):
    """
    Split PCM audio into frames.
    """

    bytes_per_sample = 2  # 16-bit PCM
    frame_size = int(
        sample_rate *
        (frame_duration_ms / 1000.0) *
        bytes_per_sample
    )

    offset = 0

    while offset + frame_size <= len(audio):
        yield audio[offset:offset + frame_size]
        offset += frame_size


def collect_speech_frames(
    audio: bytes,
    sample_rate: int = 16000,
    frame_duration_ms: int = 30
):
    """
    Keep only speech frames.
    """

    speech_frames = []

    for frame in frame_generator(
        audio,
        frame_duration_ms,
        sample_rate
    ):
        if is_speech(frame, sample_rate):
            speech_frames.append(frame)

    return b"".join(speech_frames)


def pcm16_bytes_to_numpy(audio_bytes: bytes):
    """
    Convert PCM16 bytes to numpy float32 audio.
    """

    audio_np = np.frombuffer(
        audio_bytes,
        dtype=np.int16
    ).astype(np.float32)

    audio_np /= 32768.0

    return audio_np

