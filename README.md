# AI-Powered Meeting Summarizer

A Streamlit application that performs multi-speaker diarization, per-speaker transcription, sentiment analysis, simple analytics (speaking time, keywords/wordcloud), and exports (CSV/PDF).

This repository contains a single Streamlit app at `app.py`.

## Quick setup (Windows / PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install dependencies:

```powershell
python -m pip install -U pip
pip install -r requirements.txt

**Note (Whisper / PyTorch on Windows):** Installing `openai-whisper` pulls in PyTorch. On Windows you may want to install a PyTorch wheel optimized for your CUDA setup (or CPU-only). Visit https://pytorch.org/get-started/locally/ and follow the instructions. Example CPU-only pip install:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
```

3. Ensure `ffmpeg` is installed and on your PATH (required by `moviepy` for video/audio processing). You can install it via Chocolatey, scoop, or download from ffmpeg.org. Example with Chocolatey:

```powershell
choco install ffmpeg
```

4. (Optional) If you need access to private Hugging Face models, set your HF token as an environment variable:

```powershell
$env:HF_API_TOKEN = 'your_hf_token_here'
```

5. Run the Streamlit app:

```powershell
streamlit run "d:\AI-Powered Meeting Summarizer\app.py"
```

## Notes & limitations

- The app currently uses `pyannote.audio` for speaker diarization which may require model downloads and (in some cases) an HF token.
- Transcription uses the `speech_recognition` Google recognizer by default (online). For production or privacy, consider switching to a local model such as OpenAI Whisper (local) or Vosk.
- Large audio files may take a long time to process. The app implements chunking for long segments but still runs synchronously.
- If the sentiment model or diarization model fails to load, the app will show an error and skip that step gracefully.

## Next steps (possible improvements)

- Replace `speech_recognition` with Whisper (local or API) for better accuracy and privacy.
- Add speaker label editing/merge UI so users can map anonymous speaker IDs to names.
- Add background processing for long files and a job queue.
- Add unit tests and a simple CI workflow.

If you want, I can implement Whisper-based transcription next, or add a small UI to rename/merge speakers. Which should I do next?