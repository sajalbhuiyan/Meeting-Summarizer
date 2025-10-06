# ===========================================
# AI-Powered Meeting Summarizer & Analytics (Improved)
# Multi-Speaker Diarization + Sentiment + Dashboard + PDF Export
# ===========================================

# --------------------
# 1) Imports
# --------------------
import streamlit as st
from pyannote.audio import Pipeline
from pydub import AudioSegment
from transformers import pipeline as hf_pipeline
from utils import load_whisper_model, transcribe_audiosegment
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from fpdf import FPDF
import os
from io import BytesIO
import tempfile
try:
    import moviepy.editor as mp
    _MOVIEPY_AVAILABLE = True
except Exception:
    mp = None
    _MOVIEPY_AVAILABLE = False

# --------------------
# 2) Tiny contract / assumptions
# --------------------
# - Input: uploaded audio/video file (wav/mp3/mp4)
# - Output: speaker-segment table, per-speaker transcripts, sentiment, simple analytics and download options
# - Assumes: internet access for model downloads and Google speech API (speech_recognition)
# - Edge cases handled: long segments, model load errors, empty/unrecognized speech

# --------------------
# 3) App Title
# --------------------
st.title("🎙️ AI-Powered Meeting Summarizer & Analytics (Improved)")

# --------------------
# Helpers and cached resources
# --------------------
@st.cache_resource
def load_diarization_pipeline():
    try:
        return Pipeline.from_pretrained("pyannote/speaker-diarization")
    except Exception as e:
        # Return None and surface message where used
        return e

@st.cache_resource
def load_sentiment_pipeline():
    try:
        return hf_pipeline("sentiment-analysis")
    except Exception as e:
        return e


@st.cache_resource
def get_whisper_model(model_name: str = "small"):
    # use the helper from utils but cache at Streamlit layer
    return load_whisper_model(model_name)


# --------------------
# 4) Upload Audio/Video
# --------------------
uploaded_file = st.file_uploader("Upload Meeting Audio/Video (mp3/wav/mp4)", type=["mp3", "wav", "mp4"]) 

# If moviepy isn't available on the host or the user prefers local conversion,
# show a short ffmpeg command to extract audio locally before upload.
with st.expander("Need to convert mp4 to wav locally? \n(Click to expand)"):
    st.markdown("If you can't upload mp4 or don't want to install `moviepy` on the server, extract audio locally with ffmpeg:")
    st.code("ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 output.wav", language='bash')
    st.markdown("Then upload `output.wav` instead of the mp4. This avoids server-side video processing and is faster for large files.")
    st.markdown("---")
    st.markdown(
        "Or convert the mp4 to wav directly in your browser (no server deps). This uses ffmpeg.wasm and runs entirely in your browser; the converted file is downloadable locally and you can then upload it to the app."
    )

    ffmpeg_html = r'''
<div>
  <input id="u" type="file" accept="video/*" />
  <button id="c">Convert to WAV</button>
  <div id="status"></div>
  <a id="dl" style="display:none">Download WAV</a>
</div>
<script src="https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.11.9/dist/ffmpeg.min.js"></script>
<script>
const { createFFmpeg, fetchFile } = FFmpeg;
const ffmpeg = createFFmpeg({ log: true });
const input = document.getElementById('u');
const btn = document.getElementById('c');
const status = document.getElementById('status');
const dl = document.getElementById('dl');

btn.onclick = async () => {
  if (!input.files.length) { alert('Select a video file first'); return; }
  const file = input.files[0];
  status.innerText = 'Loading ffmpeg (this may take a few seconds)...';
  if (!ffmpeg.isLoaded()) await ffmpeg.load();
  status.innerText = 'Converting...';
  const name = 'input.' + file.name.split('.').pop();
  ffmpeg.FS('writeFile', name, await fetchFile(file));
  try {
    await ffmpeg.run('-i', name, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', 'output.wav');
    const data = ffmpeg.FS('readFile', 'output.wav');
    const blob = new Blob([data.buffer], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    dl.href = url;
    dl.download = file.name.replace(/\.\w+$/, '') + '.wav';
    dl.style.display = 'inline-block';
    dl.innerText = 'Download converted WAV';
    status.innerText = 'Conversion complete.';
  } catch (err) {
    status.innerText = 'Conversion failed: ' + err.message;
  }
};
</script>
'''
    components.html(ffmpeg_html, height=300)
if uploaded_file:
    # Use TemporaryDirectory so files are cleaned up automatically
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # If video, extract audio
        if uploaded_file.name.lower().endswith(".mp4"):
            # Prefer moviepy if available
            if _MOVIEPY_AVAILABLE:
                st.info("Extracting audio from video (moviepy)...")
                try:
                    clip = mp.VideoFileClip(temp_file_path)
                    audio_path = os.path.join(temp_dir, "extracted_audio.wav")
                    clip.audio.write_audiofile(audio_path, logger=None)
                    temp_file_path = audio_path
                except Exception as e:
                    st.error(f"Failed to extract audio from video using moviepy: {e}")
                    st.stop()
            else:
                # Try ffmpeg CLI if available on PATH
                import shutil, subprocess
                ffmpeg_path = shutil.which('ffmpeg')
                if ffmpeg_path:
                    st.info("Extracting audio from video using ffmpeg on the server...")
                    audio_path = os.path.join(temp_dir, "extracted_audio.wav")
                    cmd = [ffmpeg_path, '-y', '-i', temp_file_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path]
                    try:
                        with st.spinner('Running ffmpeg...'):
                            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        temp_file_path = audio_path
                    except subprocess.CalledProcessError as e:
                        st.error(f"ffmpeg failed to extract audio: {e}")
                        st.stop()
                else:
                    st.error("Video processing (mp4 -> audio) requires either the 'moviepy' package or the 'ffmpeg' binary on PATH. Neither is available in this environment.")
                    st.info("Please upload an audio file (wav/mp3) instead or install 'moviepy' or 'ffmpeg' in your deployment environment.")
                    st.stop()

        st.audio(temp_file_path, format='audio/wav')
        st.success("File ready for processing!")

        # --------------------
        # 5) Multi-Speaker Diarization
        # --------------------
        st.subheader("🔊 Speaker Segmentation")
        with st.spinner("Performing speaker diarization..."):
            model = load_diarization_pipeline()
            if model is None:
                st.error("Failed to load diarization model")
                st.info("If you're using a private model, ensure HF_TOKEN is set as an environment variable and you have access to 'pyannote/speaker-diarization'.")
                st.stop()
            elif isinstance(model, Exception):
                st.error(f"Error loading diarization model: {model}")
                st.stop()
            try:
                diarization = model(temp_file_path)
            except Exception as e:
                st.error(f"Diarization failed: {e}")
                st.stop()

        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({"speaker": speaker, "start": turn.start, "end": turn.end})

        if not speaker_segments:
            st.warning("No speaker segments found.")
            st.stop()

        df_speakers = pd.DataFrame(speaker_segments)
        st.write("Speaker segments:")
        st.dataframe(df_speakers)

        # --------------------
        # 6) Transcription
        # --------------------
        st.subheader("📝 Transcription per Speaker")
        # Transcription backend selection
        backend = st.selectbox("Transcription backend:", options=["google", "whisper"], index=0)
        whisper_model = None
        if backend == "whisper":
            model_name = st.selectbox("Whisper model size:", options=["tiny", "base", "small", "medium", "large"], index=2)
            with st.spinner(f"Loading Whisper model ({model_name})..."):
                whisper_model = load_whisper_model(model_name)
                if isinstance(whisper_model, Exception):
                    st.error(f"Failed to load Whisper model: {whisper_model}")
                    st.info("Falling back to Google backend")
                    backend = "google"

        try:
            audio = AudioSegment.from_file(temp_file_path)
        except Exception as e:
            st.error(f"Failed to load audio file: {e}")
            st.stop()
        speaker_transcripts = []

        total_segments = len(df_speakers)
        progress = st.progress(0)
        for idx, row in df_speakers.iterrows():
            start_ms = int(row['start'] * 1000)
            end_ms = int(row['end'] * 1000)
            # clip boundaries safety
            start_ms = max(0, start_ms)
            end_ms = min(len(audio), end_ms)
            segment = audio[start_ms:end_ms]
            text = transcribe_audiosegment(segment, backend=backend, whisper_model=whisper_model)
            if not text:
                text = "[Unrecognized Speech]"
            speaker_transcripts.append({"speaker": row['speaker'], "text": text})
            progress.progress(int((idx + 1) / total_segments * 100))

        df_transcript = pd.DataFrame(speaker_transcripts)

        # allow renaming/merging speakers
        st.subheader("🧑‍💼 Rename / Merge Speakers")
        unique_speakers = df_transcript['speaker'].unique().tolist()
        new_names = {}
        for sp in unique_speakers:
            new = st.text_input(f"Label for {sp}", value=sp)
            new_names[sp] = new.strip() if new.strip() else sp
        df_transcript['speaker'] = df_transcript['speaker'].map(new_names)

        st.dataframe(df_transcript)

        # --------------------
        # 7) Sentiment Analysis
        # --------------------
        st.subheader("😊 Sentiment Analysis per Speaker")
        sentiment_model = load_sentiment_pipeline()
        if isinstance(sentiment_model, Exception):
            st.error(f"Could not load sentiment model: {sentiment_model}")
            st.info("Sentiment will be skipped.")
            df_transcript['sentiment'] = "UNKNOWN"
        else:
            def safe_sentiment(s):
                if not s or s.strip() == "[Unrecognized Speech]":
                    return "NEUTRAL"
                try:
                    return sentiment_model(s)[0]['label']
                except Exception:
                    return "ERROR"
            df_transcript['sentiment'] = df_transcript['text'].apply(safe_sentiment)
        st.dataframe(df_transcript)

        # Plot sentiment distribution
        if not df_transcript.empty:
            fig_sent = px.histogram(df_transcript, x="speaker", color="sentiment", barmode="group",
                                    title="Sentiment Distribution per Speaker")
            st.plotly_chart(fig_sent)

        # --------------------
        # 8) Speaking Time Analysis
        # --------------------
        st.subheader("⏱️ Speaking Time Distribution")
        df_speakers['duration'] = df_speakers['end'] - df_speakers['start']
        speaking_time = df_speakers.groupby('speaker')['duration'].sum().reset_index()
        fig_time = px.pie(speaking_time, names='speaker', values='duration', title='Speaking Time Distribution')
        st.plotly_chart(fig_time)

        # --------------------
        # 9) Keyword Extraction & WordCloud
        # --------------------
        st.subheader("🔑 Top Keywords per Speaker")
        top_keywords = {}
        for speaker, group in df_transcript.groupby('speaker'):
            words = " ".join(group['text'].tolist()).lower().split()
            # remove common stopwords + tiny words (clean token first)
            cleaned = [w.strip('.,!?()[]"\'').lower() for w in words]
            filtered = [w for w in cleaned if len(w) > 3 and w not in STOPWORDS]
            top_keywords[speaker] = dict(Counter(filtered).most_common(10))
            # display wordcloud
            if top_keywords[speaker]:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_keywords[speaker])
                plt.figure(figsize=(8,4))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
                plt.close()
            else:
                st.write(f"No keywords found for {speaker}")

        # --------------------
        # 10) Download Transcript CSV
        # --------------------
        st.subheader("💾 Download Results")
        csv_bytes = df_transcript.to_csv(index=False).encode('utf-8')
        st.download_button("Download Transcript CSV", data=csv_bytes, file_name="speaker_transcript.csv", mime='text/csv')

        # --------------------
        # 11) Export PDF Report
        # --------------------
        st.subheader("📄 Download PDF Report")
        if st.button("Generate PDF Report"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Meeting Summary Report", ln=True, align='C')
            pdf.ln(10)

            # Speaker Summary
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, "Speaker Segments & Transcription:", ln=True)
            for i, row in df_transcript.iterrows():
                pdf.multi_cell(0, 8, f"{row['speaker']}: {row['text']} (Sentiment: {row.get('sentiment','N/A')})")

            # Speaking time
            pdf.ln(5)
            pdf.cell(0, 10, "Speaking Time (seconds):", ln=True)
            for i, row in speaking_time.iterrows():
                pdf.cell(0, 8, f"{row['speaker']}: {row['duration']:.2f}", ln=True)

            # Keywords
            pdf.ln(5)
            pdf.cell(0, 10, "Top Keywords per Speaker:", ln=True)
            for speaker, kw in top_keywords.items():
                pdf.multi_cell(0, 8, f"{speaker}: {', '.join(kw.keys())}")

            pdf_bytes = pdf.output(dest='S').encode('latin-1') if hasattr(pdf, 'output') else None
            if pdf_bytes:
                st.download_button("Download PDF Report", data=pdf_bytes, file_name="meeting_summary.pdf", mime='application/pdf')
            else:
                # Fallback: write to file then read
                pdf_file = os.path.join(temp_dir, "meeting_summary.pdf")
                pdf.output(pdf_file)
                with open(pdf_file, "rb") as f:
                    st.download_button("Download PDF Report", data=f.read(), file_name="meeting_summary.pdf", mime='application/pdf')

else:
    st.info("Upload an audio or video file to start meeting analysis.")