import os
import torch
import streamlit as st
import whisper
import torchaudio
import faiss
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core.node_parser import SentenceSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document as LangDoc

st.set_page_config(page_title="Transcript QA Chat", layout="centered")

transcript_dir = "/scratch/mir58sab/large_files/tmp"
os.makedirs(transcript_dir, exist_ok=True)

api_key_path = os.path.expanduser("~/.openai_api_key.txt")

def get_api_key():
    if os.path.exists(api_key_path):
        with open(api_key_path, "r") as f:
            return f.read().strip()
    return None

st.subheader("\U0001F510 OpenAI API Key")
stored_key = get_api_key()
user_api_key = st.text_input("Enter your OpenAI API Key", type="password", value=stored_key or "")
remember_key = st.checkbox("Remember this key for future use")


hf_token_path = os.path.expanduser("~/.hf_token.txt")

def get_hf_token():
    if os.path.exists(hf_token_path):
        with open(hf_token_path, "r") as f:
            return f.read().strip()
    return None

st.subheader("🔐 Hugging Face Access Token")
stored_hf_token = get_hf_token()
user_hf_token = st.text_input("Enter your Hugging Face Token", type="password", value=stored_hf_token or "")
remember_hf = st.checkbox("Remember this token for future use")

if user_hf_token:
    if remember_hf and not os.path.exists(hf_token_path):
        with open(hf_token_path, "w") as f:
            f.write(user_hf_token)
    os.environ["HF_TOKEN"] = user_hf_token
else:
    st.warning("Please enter your Hugging Face token to continue.")
    st.stop()


if user_api_key:
    if remember_key and not os.path.exists(api_key_path):
        with open(api_key_path, "w") as f:
            f.write(user_api_key)
    os.environ["OPENAI_API_KEY"] = user_api_key
else:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file:
    wav_path = os.path.join("/tmp", uploaded_file.name)
    with open(wav_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("\U0001F50A Running speaker diarization and transcription...")

    whisper_model = whisper.load_model("large")
    diarization_pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ["HF_TOKEN"]
    )

    diarization_pipeline.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    waveform, sample_rate = torchaudio.load(wav_path)
    with ProgressHook():
        diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})

    whisper_result = whisper_model.transcribe(wav_path, verbose=False)
    segments = whisper_result.get("segments", [])
    combined_output = []

    for segment in segments:
        seg_start, seg_end = segment["start"], segment["end"]
        seg_text = segment["text"].strip()
        assigned_speaker = "UNKNOWN"
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= seg_start <= turn.end:
                assigned_speaker = speaker
                break
        combined_output.append(f"{assigned_speaker}: {seg_text}")

    del whisper_model, diarization_pipeline
    torch.cuda.empty_cache()

    transcript_name = os.path.splitext(uploaded_file.name)[0] + ".txt"
    transcript_path = os.path.join(transcript_dir, transcript_name)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_output))

    st.success("Transcript ready. Now building QA system...")

    with open(transcript_path, "r", encoding="utf-8") as f:
        preview_text = f.read()

    if preview_text.strip() == "":
        st.error("Transcript is empty. Please try a different audio file.")
        st.stop()
    else:
        st.text_area("Transcript Output", value=preview_text, height=300)

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = splitter.split_text(preview_text)
    metadata = [{"file_name": transcript_name, "chunk_id": i} for i in range(len(chunks))]
    lang_docs = [LangDoc(page_content=c, metadata=m) for c, m in zip(chunks, metadata)]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_store = FAISS.from_documents(lang_docs, embeddings)

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=faiss_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=False
    )

    st.subheader("Ask a question about the transcript")
    user_input = st.text_input("Type your message:")

    if user_input:
        response = qa_chain.run(user_input)
        st.markdown("**Answer:**")
        st.write(response)
