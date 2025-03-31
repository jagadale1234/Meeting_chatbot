# Meeting_chatbot

# Transcript QA Chatbot with Speaker Diarization

This project allows users to upload a `.wav` meeting audio file and interactively ask questions about its content using a chatbot interface. It performs:

-  Audio Transcription using Whisper
-  Speaker Diarization using PyAnnote
-  Q&A over the transcript using LangChain and OpenAI's GPT-4o
-  Conversational memory to enable follow-up questions

## Features

- Upload `.wav` meeting recordings
- Automatic speaker-labeled transcription
- Interactive QA chat with GPT-4o (via OpenAI API)
- Option to use custom Hugging Face and OpenAI tokens
- Transcript is previewed and saved locally for reference

---

## Tech Stack

- [Streamlit](https://streamlit.io/) – Frontend UI
- [OpenAI API (GPT-4o)](https://platform.openai.com/)
- [HuggingFace Transformers + PyAnnote](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [Whisper](https://github.com/openai/whisper)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss) – vector store

---

## Installation

```bash
# Clone this repo
git clone https://github.com/yourusername/transcript-qa-chat.git
cd transcript-qa-chat

# (Recommended) Create and activate virtual environment
conda create -n transcript-qa python=3.10 -y
conda activate transcript-qa

# Install dependencies
pip install -r requirements.txt
```

---
## API Keys:
- Requires OPEN AI Api key
- Hugging Face Api key
You can upload both of them directly on the UI and ask it to save so you won't need to do it again.

---

## Usage

```bash
streamlit run lang.py
```
