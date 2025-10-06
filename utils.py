import os
import tempfile
from typing import Optional

import speech_recognition as sr
from pydub import AudioSegment

# Keep a recognizer instance for Google backend
_recognizer = sr.Recognizer()


def load_whisper_model(model_name: str = "small"):
    """Load and return a whisper model or an Exception on failure."""
    try:
        import whisper
        model = whisper.load_model(model_name)
        return model
    except Exception as e:
        return e


def transcribe_audiosegment(segment: AudioSegment, backend: str = "google", whisper_model: Optional[object] = None, max_chunk_s: int = 50):
    """Transcribe a pydub.AudioSegment using the chosen backend.

    Args:
        segment: AudioSegment to transcribe.
        backend: 'google' or 'whisper'
        whisper_model: loaded whisper model if backend is 'whisper'
        max_chunk_s: chunk size in seconds.

    Returns:
        Concatenated transcription string.
    """
    if len(segment) == 0:
        return ""

    ms_per_chunk = max_chunk_s * 1000
    texts = []
    with tempfile.TemporaryDirectory() as td:
        for i, start in enumerate(range(0, len(segment), ms_per_chunk)):
            chunk = segment[start:start + ms_per_chunk]
            path = os.path.join(td, f"segment_{i}.wav")
            chunk.export(path, format="wav")

            if backend == "whisper":
                if whisper_model is None or isinstance(whisper_model, Exception):
                    texts.append("[Whisper model not available]")
                    continue
                try:
                    result = whisper_model.transcribe(path)
                    texts.append(result.get('text', '').strip())
                except Exception:
                    texts.append("[Unrecognized Speech]")
            else:
                with sr.AudioFile(path) as source:
                    audio_data = _recognizer.record(source)
                    try:
                        texts.append(_recognizer.recognize_google(audio_data))
                    except Exception:
                        texts.append("[Unrecognized Speech]")

    return " ".join([t for t in texts if t])
