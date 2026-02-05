import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from groq import Groq
import PyPDF2


# ---------------- PAGE ----------------
st.set_page_config(page_title="Study AI Bot", layout="wide")
st.title("🎓 Study AI Bot")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    groq_key = st.text_input("Enter GROQ API Key", type="password")


# ---------------- SESSION ----------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- EMBEDDING MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()


def embed_text(texts):
    return embed_model.encode(texts).tolist()


class FreeEmbedding(Embeddings):

    def embed_documents(self, texts):
        return embed_text(texts)

    def embed_query(self, text):
        return embed_text([text])[0]


# ---------------- UTIL FUNCTIONS ----------------
def extract_video_id(url):
    parsed = urlparse(url)

    if "youtu.be" in parsed.netloc:
        return parsed.path[1:]

    return parse_qs(parsed.query).get("v", [None])[0]


def get_transcript(video_id):

    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.list(video_id)

    try:
        transcript = transcript_list.find_transcript(['en'])
    except:
        transcript = transcript_list.find_generated_transcript(['en'])

    data = transcript.fetch()

    return " ".join([i.text for i in data])


def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    return text


# ---------------- VECTOR DB ----------------
def create_db(text):

    # Token protection
    if len(text) > 150000:
        text = text[:150000]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.create_documents([text])

    return FAISS.from_documents(docs, FreeEmbedding())


# ---------------- INPUT SECTION ----------------
st.header("📚 Upload Study Material")

youtube_url = st.text_input("YouTube Lecture URL")
manual_notes = st.text_area("Paste Notes")
uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
uploaded_txt = st.file_uploader("Upload TXT", type=["txt"])


if st.button("Process Study Material"):

    combined_text = ""

    # YouTube
    if youtube_url:
        try:
            vid = extract_video_id(youtube_url)
            combined_text += get_transcript(vid)
        except:
            st.warning("Could not fetch transcript.")

    # Manual
    if manual_notes:
        combined_text += manual_notes

    # PDF
    if uploaded_pdf:
        combined_text += read_pdf(uploaded_pdf)

    # TXT
    if uploaded_txt:
        combined_text += uploaded_txt.read().decode("utf-8")

    if combined_text.strip() == "":
        st.error("Provide some study material")
    else:
        st.session_state.vector_db = create_db(combined_text)
        st.success("Material Loaded!")


# ---------------- CHAT SECTION ----------------
st.header("💬 Ask Questions")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if question := st.chat_input("Ask study doubt..."):

    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    if st.session_state.vector_db and groq_key:

        try:
            client = Groq(api_key=groq_key)

            docs = st.session_state.vector_db.similarity_search(question, k=2)

            context = "\n".join([d.page_content for d in docs])[:3500]

            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "system",
                        "content":
                        "You are a friendly tutor. Explain simply."
                    },
                    {
                        "role": "user",
                        "content": f"Material:\n{context}\n\nQuestion:\n{question}"
                    }
                ],
                max_tokens=500
            )

            reply = response.choices[0].message.content

            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.chat_message("assistant").write(reply)

        except:
            st.error("AI failed. Try smaller content.")

    else:
        st.warning("Upload material + Enter Groq key")


# ---------------- STUDY TOOLS ----------------
st.header("✨ Study Tools")

def generate_tool(prompt_text):

    docs = st.session_state.vector_db.similarity_search(prompt_text, k=3)
    context = "\n".join([d.page_content for d in docs])[:3500]

    client = Groq(api_key=groq_key)

    result = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": context}
        ],
        max_tokens=400
    )

    return result.choices[0].message.content


if st.button("Generate Summary") and st.session_state.vector_db and groq_key:
    try:
        st.write(generate_tool("Summarize"))
    except:
        st.error("Summary too large.")


if st.button("Generate Quiz") and st.session_state.vector_db and groq_key:
    try:
        st.write(generate_tool("Create 5 quiz questions"))
    except:
        st.error("Quiz generation failed.")
