import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from faster_whisper import WhisperModel
from groq import Groq
import yt_dlp
import tempfile
import os


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Chat with YouTube FREE", layout="wide")
st.title("📺 Chat with YouTube (100% FREE)")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    groq_key = st.text_input("Enter GROQ API Key (Free)", type="password")


# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# ---------------- CACHE EMBEDDING MODEL ----------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()


def embed_text(texts):
    return embed_model.encode(texts).tolist()


# ---------------- LANGCHAIN EMBEDDING CLASS ----------------
class FreeEmbedding(Embeddings):

    def embed_documents(self, texts):
        return embed_text(texts)

    def embed_query(self, text):
        return embed_text([text])[0]


# ---------------- VIDEO ID EXTRACTION ----------------
def extract_video_id(url):
    parsed = urlparse(url)

    if "youtu.be" in parsed.netloc:
        return parsed.path[1:]

    return parse_qs(parsed.query).get("v", [None])[0]


# ---------------- TRANSCRIPT FETCH ----------------
def get_transcript(video_id):

    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.list(video_id)

    try:
        transcript = transcript_list.find_transcript(['en'])
    except:
        transcript = transcript_list.find_generated_transcript(['en'])

    data = transcript.fetch()

    return " ".join([i.text for i in data])


# ---------------- WHISPER FALLBACK ----------------
@st.cache_resource
def load_whisper():
    return WhisperModel("base", compute_type="int8")

whisper_model = load_whisper()


def whisper_fallback(video_url):

    with tempfile.TemporaryDirectory() as tmpdir:

        audio_file = os.path.join(tmpdir, "audio.mp3")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_file,
            'quiet': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        segments, _ = whisper_model.transcribe(audio_file)

        text = ""
        for seg in segments:
            text += seg.text + " "

        return text


# ---------------- CACHE TRANSCRIPT ----------------
@st.cache_data
def cached_transcript(video_id, url):

    try:
        return get_transcript(video_id)
    except:
        return whisper_fallback(url)


# ---------------- CREATE VECTOR DB ----------------
def create_db(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    return FAISS.from_documents(docs, FreeEmbedding())


# ---------------- INPUT ----------------
url = st.text_input("Enter YouTube URL")

if st.button("Analyze"):

    vid = extract_video_id(url)

    if not vid:
        st.error("Invalid URL")
    else:
        with st.spinner("Processing Video..."):

            text = cached_transcript(vid, url)
            st.session_state.vector_db = create_db(text)

            st.success("Video Ready! Ask questions below.")


# ---------------- DISPLAY CHAT ----------------
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])


# ---------------- CHAT ----------------
if prompt := st.chat_input("Ask about the video"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if st.session_state.vector_db and groq_key:

        client = Groq(api_key=groq_key)

        docs = st.session_state.vector_db.similarity_search(prompt, k=3)
        context = "\n".join([d.page_content for d in docs])

        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": f"Answer using this context:\n{context}"},
                {"role": "user", "content": prompt}
            ]
        )

        reply = completion.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)

    else:
        st.warning("Analyze video + enter Groq key")
