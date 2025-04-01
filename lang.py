import os
import torch
import streamlit as st
import whisper
import torchaudio
import faiss
import threading
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core.node_parser import SentenceSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document as LangDoc

st.set_page_config(page_title="Transcript QA Chat", layout="centered")

transcript_dir = "/scratch/mir58sab/large_files/tmp"
os.makedirs(transcript_dir, exist_ok=True)

api_key_path = os.path.expanduser("~/.openai_api_key.txt")
hf_token_path = os.path.expanduser("~/.hf_token.txt")

def get_token(path):
    return open(path).read().strip() if os.path.exists(path) else None

st.sidebar.title("🔐 API Keys")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password", value=get_token(api_key_path) or "")
hf_token = st.sidebar.text_input("HuggingFace Token", type="password", value=get_token(hf_token_path) or "")
remember = st.sidebar.checkbox("Remember tokens")

embedding_choice = st.sidebar.radio("Choose embedding model:", ["HuggingFace", "OpenAI"])

if openai_key and hf_token:
    os.environ["OPENAI_API_KEY"] = openai_key
    if remember:
        open(api_key_path, "w").write(openai_key)
        open(hf_token_path, "w").write(hf_token)
else:
    st.warning("Please enter both API keys.")
    st.stop()

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file and "qa_chain" not in st.session_state:
    wav_path = os.path.join("/tmp", uploaded_file.name)
    with open(wav_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("⚙️ Processing Audio File")

    whisper_result = {}
    diarization_result = {}

    def transcribe():
        model = whisper.load_model("large")
        whisper_result["data"] = model.transcribe(wav_path, verbose=False)
        del model
        torch.cuda.empty_cache()

    def diarize():
        pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        pipeline.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        waveform, sample_rate = torchaudio.load(wav_path)
        with ProgressHook():
            diarization_result["data"] = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        del pipeline
        torch.cuda.empty_cache()

    with st.spinner("🔍 Running Whisper + PyAnnote..."):
        t1 = threading.Thread(target=transcribe)
        t2 = threading.Thread(target=diarize)
        t1.start(); t2.start()
        t1.join(); t2.join()

    whisper_segments = whisper_result["data"].get("segments", [])
    diarization = diarization_result["data"]
    transcript = []
    for seg in whisper_segments:
        speaker = "UNKNOWN"
        for turn, _, spk in diarization.itertracks(yield_label=True):
            if turn.start <= seg["start"] <= turn.end:
                speaker = spk
                break
        transcript.append(f"{speaker}: {seg['text'].strip()}")

    transcript_name = os.path.splitext(uploaded_file.name)[0] + ".txt"
    transcript_path = os.path.join(transcript_dir, transcript_name)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript))

    with open(transcript_path, "r", encoding="utf-8") as f:
        preview_text = f.read()

    if preview_text.strip() == "":
        st.error("❌ Transcript is empty.")
        st.stop()

    st.success("✅ Transcript Ready")
    st.text_area("📝 Transcript Preview", value=preview_text, height=300, key="transcript_preview")

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = splitter.split_text(preview_text)
    metadata = [{"file_name": transcript_name, "chunk_id": i} for i in range(len(chunks))]
    lang_docs = [LangDoc(page_content=c, metadata=m) for c, m in zip(chunks, metadata)]

    if embedding_choice == "HuggingFace":
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=os.path.expanduser("~/.cache/huggingface"),
            model_kwargs={"use_auth_token": hf_token}
        )
    else:
        embeddings = OpenAIEmbeddings()

    faiss_store = FAISS.from_documents(lang_docs, embeddings)

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=faiss_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=False
    )
    st.session_state.chat_history = []
    st.session_state.preview_text = preview_text

# === QA Interface ===
if "qa_chain" in st.session_state:
    st.text_area("📝 Transcript Preview", value=st.session_state.preview_text, height=300, key="transcript_preview_again")
    st.subheader("💬 Ask a question")
    user_input = st.text_input("Type your message:")
    if user_input:
        response = st.session_state.qa_chain.run(user_input)
        st.session_state.chat_history.append((user_input, response))

    if st.session_state.chat_history:
        st.subheader("📜 Chat History")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            st.markdown("---")
