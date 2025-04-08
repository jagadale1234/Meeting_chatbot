# Meeting Chatbot

## Transcript QA Chatbot with Speaker Diarization

This project provides two implementations for processing meeting recordings:
1. **Full Version** - Uses Whisper and PyAnnote (requires GPU)
2. **API-Only Version** - Uses Deepgram API (no GPU required)

## Features

### Full Version
- Audio Transcription using Whisper
- Speaker Diarization using PyAnnote  
- Q&A over transcript using LangChain and OpenAI's GPT-4o
- Conversational memory for follow-up questions

### API-Only Version
- Audio Transcription using Deepgram API
- Built-in speaker diarization
- Same Q&A capabilities without local GPU requirements
- More reliable for long recordings

## Tech Stack

### Core Components
- [Streamlit](https://streamlit.io/) – Frontend UI
- [OpenAI API](https://platform.openai.com/) – GPT-4 for Q&A
- [LangChain](https://www.langchain.com/) – Conversation management

### Full Version Additions
- [Whisper](https://github.com/openai/whisper) – Local transcription (GPU)
- [PyAnnote](https://huggingface.co/pyannote/speaker-diarization-3.1) – Speaker diarization  
- [FAISS](https://github.com/facebookresearch/faiss) – Vector store

### API Version Additions
- [Deepgram API](https://deepgram.com/) – Cloud transcription & diarization

---

## Installation

```bash
# Clone repo
git clone https://github.com/jagadale1234/Meeting_chatbot.git
cd Meeting_chatbot

# Create environment (recommended)
conda create -n meetingbot python=3.10 -y
conda activate meetingbot

# Install base requirements
pip install -r requirements.txt
```
## For GPU version only:
```bash
pip install torchaudio pyannote.audio
```
## Configuration
API Keys Required:
OpenAI API key (required for both versions)

Hugging Face token (Full version only)

Deepgram API key (API version only)

Keys can be entered directly in the UI and saved to avoid re-entry.

## Usage
Full Version (GPU)
```bash
streamlit run lang.py
```
API-Only Version (No GPU)
```bash
streamlit run meets_api.py
```
## PyPI Package(For CLI usage):
```bash
pip install meetscribe
```
Package available at: https://pypi.org/project/meetscribe/
