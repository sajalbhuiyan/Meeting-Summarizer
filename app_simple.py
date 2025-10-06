import streamlit as st
from utils import simple_diarize_file, transcribe_audiosegment
import tempfile
import os
import shutil
import subprocess
import pandas as pd

st.set_page_config(page_title="Meeting Summarizer (Simple)", layout="wide")
st.title("🎙️ Meeting Summarizer — Simple")

st.markdown(
    "This lightweight version uses a simple silence-based diarization and the Google Web Speech API (via SpeechRecognition).\n"
    "It avoids heavy dependencies and gated models. For best results upload WAV/MP3 files. For MP4, ensure `ffmpeg` is installed on the host."
)

uploaded_file = st.file_uploader("Upload audio/video (wav/mp3/mp4)", type=["wav", "mp3", "mp4"])

if not uploaded_file:
    st.info("Upload a file to get started.")
    st.stop()

with tempfile.TemporaryDirectory() as td:
    input_path = os.path.join(td, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # If mp4, try to extract audio via ffmpeg if available
    if uploaded_file.name.lower().endswith(".mp4"):
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            st.error("ffmpeg is required to extract audio from MP4. Install ffmpeg or upload WAV/MP3.")
            st.stop()
        audio_path = os.path.join(td, "extracted.wav")
        cmd = [ffmpeg_path, "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            input_path = audio_path
        except subprocess.CalledProcessError:
            st.error("ffmpeg failed to extract audio from the video. Try converting locally and re-uploading.")
            st.stop()

    # load audio using pydub (lazy import with helpful error)
    try:
        from pydub import AudioSegment
    except Exception:
        st.error("pydub is required. Install with: pip install pydub and ensure ffmpeg is available.")
        st.stop()

    try:
        audio = AudioSegment.from_file(input_path)
    except Exception as e:
        st.error(f"Failed to read audio: {e}")
        st.stop()

    st.audio(input_path)

    st.header("🔊 Speaker Segmentation")
    min_silence_len = st.number_input("Min silence length (ms)", value=700, min_value=100, max_value=5000)
    silence_thresh = st.slider("Silence threshold (dBFS)", -80, -10, -40)
    max_speakers = st.number_input("Max speakers (fallback)", min_value=1, max_value=10, value=4)

    with st.spinner("Detecting speaker segments..."):
        segments = None
        # Try public pyannote if available (no token required); fall back to simple diarization if not
        try:
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
            diarization = pipeline(input_path)
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({"speaker": speaker, "start": turn.start, "end": turn.end})
            st.info("Using pyannote diarization (public)")
        except Exception:
            # silently fallback to simple diarization for simplicity
            segments = simple_diarize_file(input_path, min_silence_len=min_silence_len, silence_thresh=silence_thresh, max_speakers=max_speakers)

    if not segments:
        st.warning("No segments detected. Try loosening silence threshold or uploading clearer audio.")
        st.stop()

    st.write(pd.DataFrame(segments))

    st.header("📝 Transcription")
    results = []
    total = len(segments)
    progress = st.progress(0)

    for i, seg in enumerate(segments):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        clip = audio[start_ms:end_ms]
        try:
            text = transcribe_audiosegment(clip, backend="google")
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        results.append({"speaker": seg["speaker"], "start": seg["start"], "end": seg["end"], "text": text})
        progress.progress(int((i + 1) / total * 100))

    df = pd.DataFrame(results)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="transcript.csv", mime="text/csv")
