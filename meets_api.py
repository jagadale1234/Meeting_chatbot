import os
import wave
import time
import tempfile
import torch
import whisper
import torchaudio
import streamlit as st
from tempfile import NamedTemporaryFile
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pyannote.audio import Pipeline as PyAnnotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from deepgram import DeepgramClient, PrerecordedOptions, DeepgramApiError
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document as LangDoc
from llama_index.core.node_parser import SentenceSplitter

# === App Configuration ===
st.set_page_config(page_title="MeetScribe", layout="wide")


WHISPER_MODEL = "large"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE_MINUTES = 10  # 10-minute chunks
MAX_RETRIES = 3


with st.sidebar:
    st.title("API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password")
    hf_token = st.text_input("HuggingFace Token", type="password")
    deepgram_key = st.text_input("Deepgram API Key", type="password")
    
    method = st.radio("Transcription Model", ["Deepgram", "OpenAI Whisper"])
    embedding_choice = st.radio("Embedding Model", ["OpenAI", "HuggingFace"])

# === Helper Functions ===
def chunk_audio_file(file_path, chunk_size_mins=CHUNK_SIZE_MINUTES):
    """Split audio file into chunks"""
    with wave.open(file_path, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        duration = n_frames / float(frame_rate)
    
    chunk_size_seconds = chunk_size_mins * 60
    return [(start, min(start + chunk_size_seconds, duration)) 
            for start in range(0, int(duration), chunk_size_seconds)], duration

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(DeepgramApiError)
)
def transcribe_chunk_with_retry(dg_client, audio_buffer, options):
    """retry logic"""
    source = {"buffer": audio_buffer, "mimetype": "audio/wav"}
    return dg_client.listen.prerecorded.v("1").transcribe_file(source, options)

def transcribe_with_deepgram(audio_path, api_key):
    """long audio transcription with chunking"""
    try:
        dg_client = DeepgramClient(api_key)
        chunks, duration = chunk_audio_file(audio_path)
        speaker_transcripts = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (start, end) in enumerate(chunks):
            status_text.text(f"Processing chunk {i+1}/{len(chunks)} ({start//60}m:{start%60}s - {end//60}m:{end%60}s)")
            progress_bar.progress((i + 1) / len(chunks))
            
            # Extract audio segment
            with wave.open(audio_path, 'rb') as wav_file:
                frame_rate = wav_file.getframerate()
                wav_file.setpos(int(start * frame_rate))
                frames = wav_file.readframes(int((end - start) * frame_rate))
            
            # Create temporary WAV file in memory
            with NamedTemporaryFile(suffix='.wav') as temp_file:
                with wave.open(temp_file.name, 'wb') as chunk_file:
                    chunk_file.setnchannels(wav_file.getnchannels())
                    chunk_file.setsampwidth(wav_file.getsampwidth())
                    chunk_file.setframerate(frame_rate)
                    chunk_file.writeframes(frames)
                
                with open(temp_file.name, 'rb') as audio_file:
                    audio_buffer = audio_file.read()
            
            # Transcribe chunk
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                utterances=True,
                punctuate=True,
                diarize=True,
                language="en"
            )
            
            try:
                response = transcribe_chunk_with_retry(dg_client, audio_buffer, options)
                if hasattr(response.results, 'utterances'):
                    speaker_transcripts.extend(
                        f"Speaker {u.speaker}: {u.transcript}" 
                        for u in response.results.utterances
                    )
                time.sleep(1)  # Brief pause between chunks
            except Exception as e:
                st.warning(f"Chunk {i+1} failed, skipping: {str(e)}")
                continue
        
        return "\n".join(speaker_transcripts)
        
    except Exception as e:
        st.error(f"Deepgram processing failed: {str(e)}")
        raise



def transcribe_with_whisper(audio_path, diarization_pipeline, hf_token):
    """Transcribe audio using Whisper and add speaker diarization"""
    try:
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(audio_path)
        
        # diarization results
        waveform, sample_rate = torchaudio.load(audio_path)
        with ProgressHook():
            diarization = diarization_pipeline({
                "waveform": waveform,
                "sample_rate": sample_rate
            })
        
        # Combine Whisper segments with speaker labels
        labeled_segments = []
        for seg in result.get("segments", []):
            speaker = next(
                (spk for turn, _, spk in diarization.itertracks(yield_label=True)
                 if turn.start <= seg["start"] <= turn.end),
                "UNKNOWN"
            )
            labeled_segments.append(f"{speaker}: {seg['text'].strip()}")
        
        return "\n".join(labeled_segments), result.get("segments", [])
        
    except Exception as e:
        st.error(f"Whisper transcription failed: {str(e)}")
        raise

# === Main Processing ===
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if st.button("Transcribe and Process") and uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(uploaded_file.read())
        wav_path = f.name

    try:
        st.info("Starting transcription process...")
        
        if method == "Deepgram":
            if not deepgram_key:
                st.error("Deepgram API key is required")
                st.stop()
                
            transcript_text = transcribe_with_deepgram(wav_path, deepgram_key)
            st.success("✅ Transcription complete (Deepgram with enhanced diarization)")
        else:
            if not hf_token:
                st.error("HuggingFace token is required for Whisper+Pyannote")
                st.stop()
                
            diarizer = PyAnnotePipeline.from_pretrained(
                DIARIZATION_MODEL,
                use_auth_token=hf_token
            )
            transcript_text, _ = transcribe_with_whisper(wav_path, diarizer, hf_token)
            st.success("✅ Transcription complete (Whisper + Pyannote)")

        # Display transcript
        st.subheader("Transcript")
        st.text_area("Full Transcript", transcript_text, height=300)

        # === RAG Setup ===
        st.info("Setting up Q&A system...")
        
        # Document processing
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        chunks = splitter.split_text(transcript_text)
        docs = [LangDoc(page_content=c, metadata={"chunk": i}) for i, c in enumerate(chunks)]
        
        # Model selection
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            embeddings = OpenAIEmbeddings()
            llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        else:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"use_auth_token": hf_token}
            )
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        # Create conversation chain
        vector_store = FAISS.from_documents(docs, embeddings)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            return_source_documents=False
        )
        
        # Store in session state
        st.session_state.update({
            "qa_chain": qa_chain,
            "chat_history": [],
            "transcript": transcript_text
        })

    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

# === Chat Interface ===
if "qa_chain" in st.session_state:
    st.subheader("Ask Questions About the Meeting")
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Generating response..."):
            try:
                response = st.session_state.qa_chain.run(query)
                st.session_state.chat_history.append((query, response))
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    if st.session_state.chat_history:
        st.subheader("Conversation History")
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")
            st.markdown("---")