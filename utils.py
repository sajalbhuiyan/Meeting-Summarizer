import os
import tempfile
from typing import Optional

"""Utility helpers for transcription and a simple diarization fallback.

Optional heavy dependencies (pydub, speech_recognition, whisper) are imported
inside functions to avoid import errors at module import time when they are
not installed in the environment.
"""

_LOCAL_RECOGNIZER = None


def load_whisper_model(model_name: str = "small"):
    """Load and return a whisper model or an Exception on failure."""
    try:
        import whisper
        model = whisper.load_model(model_name)
        return model
    except Exception as e:
        return e


def transcribe_audiosegment(segment, backend: str = "google", whisper_model: Optional[object] = None, max_chunk_s: int = 50):
    """Transcribe a pydub.AudioSegment using the chosen backend.

    Args:
        segment: AudioSegment to transcribe.
        backend: 'google' or 'whisper'
        whisper_model: loaded whisper model if backend is 'whisper'
        max_chunk_s: chunk size in seconds.

    Returns:
        Concatenated transcription string.
    """
    # require a segment-like object with __len__ and export capabilities (pydub.AudioSegment)
    if not segment or len(segment) == 0:
        return ""

    ms_per_chunk = max_chunk_s * 1000
    texts = []

    # Lazy-import speech_recognition when needed
    if backend != "whisper":
        try:
            import speech_recognition as sr
        except Exception as e:
            raise RuntimeError("speech_recognition is required for the 'google' backend. Install it with: pip install SpeechRecognition") from e

    with tempfile.TemporaryDirectory() as td:
        for i, start in enumerate(range(0, len(segment), ms_per_chunk)):
            chunk = segment[start:start + ms_per_chunk]
            path = os.path.join(td, f"segment_{i}.wav")
            # export expects pydub.AudioSegment; if not present, this will raise
            try:
                chunk.export(path, format="wav")
            except Exception as e:
                raise RuntimeError("Failed to export audio chunk. Ensure pydub and its dependencies (ffmpeg) are installed.") from e

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
                # default: Google Web Speech API via speech_recognition
                try:
                    import speech_recognition as sr
                except Exception as e:
                    raise RuntimeError("speech_recognition is required for the 'google' backend. Install it with: pip install SpeechRecognition") from e
                recognizer = sr.Recognizer()
                with sr.AudioFile(path) as source:
                    audio_data = recognizer.record(source)
                    try:
                        texts.append(recognizer.recognize_google(audio_data))
                    except Exception:
                        texts.append("[Unrecognized Speech]")

    return " ".join([t for t in texts if t])


def simple_diarize_file(file_path: str, min_silence_len: int = 700, silence_thresh: int = -40, max_speakers: int = 4):
    """Very simple silence-based diarization fallback.

    This is approximate: it detects non-silent chunks and assigns speaker labels in a round-robin fashion.
    Returns a list of segments: {"speaker": str, "start": float, "end": float} (seconds).
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
    except Exception as e:
        raise RuntimeError("pydub is required for simple diarization fallback. Install it with: pip install pydub and ensure ffmpeg is available") from e

    audio = AudioSegment.from_file(file_path)
    # detect_nonsilent returns list of [start_ms, end_ms]
    nonsilent = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    segments = []
    for i, (start_ms, end_ms) in enumerate(nonsilent):
        speaker = f"SPEAKER_{i % max_speakers:02d}"
        segments.append({"speaker": speaker, "start": start_ms / 1000.0, "end": end_ms / 1000.0})
    return segments
