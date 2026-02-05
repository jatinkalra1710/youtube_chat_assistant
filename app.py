import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from groq import Groq


# ---------------- PAGE ----------------
st.set_page_config(page_title="Chat with YouTube FREE", layout="wide")
st.title("📺 Chat with YouTube (Free & Stable)")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    groq_key = st.text_input("Enter GROQ API Key", type="password")


# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# ---------------- EMBEDDING ----------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()


def embed_text(texts):
    return embed_model.encode(texts).tolist()


class FreeEmbedding(Embeddings):
    def embed_documents(self, texts):
        return embed_text(texts)

    def embed_query(self, text):
        return embed_text([text])[0]


# ---------------- VIDEO ID ----------------
def extract_video_id(url):
    parsed = urlparse(url)

    if "youtu.be" in parsed.netloc:
        return parsed.path[1:]

    return parse_qs(parsed.query).get("v", [None])[0]


# ---------------- TRANSCRIPT ----------------
def get_transcript(video_id):

    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.list(video_id)

    try:
        transcript = transcript_list.find_transcript(['en'])
    except:
        transcript = transcript_list.find_generated_transcript(['en'])

    data = transcript.fetch()

    return " ".join([i.text for i in data])


# ---------------- VECTOR DB ----------------
def create_db(text):

    if len(text) > 150000:
        text = text[:150000]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.create_documents([text])

    return FAISS.from_documents(docs, FreeEmbedding())


# ---------------- INPUT ----------------
url = st.text_input("Enter YouTube URL")

if st.button("Analyze"):

    vid = extract_video_id(url)

    if not vid:
        st.error("Invalid YouTube URL")
    else:
        with st.spinner("Fetching transcript..."):
            try:
                text = get_transcript(vid)
                st.session_state.vector_db = create_db(text)
                st.success("Video Ready!")

            except Exception:
                st.error(
                    "❌ Transcript unavailable.\n\n"
                    "Possible reasons:\n"
                    "- Video has captions disabled\n"
                    "- YouTube blocked cloud requests\n"
                    "- Private / restricted video\n\n"
                    "👉 Try another video."
                )


# ---------------- DISPLAY CHAT ----------------
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])


# ---------------- CHAT ----------------
if prompt := st.chat_input("Ask about video"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if st.session_state.vector_db and groq_key:

        try:
            client = Groq(api_key=groq_key)

            docs = st.session_state.vector_db.similarity_search(prompt, k=2)

            context = "\n".join([d.page_content for d in docs])[:4000]

            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": "Answer only using transcript context."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"
                    }
                ],
                max_tokens=500
            )

            reply = completion.choices[0].message.content

            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.chat_message("assistant").write(reply)

        except Exception as e:
            st.error("LLM request failed.")
            st.write(str(e))

    else:
        st.warning("Analyze video + enter Groq key")
