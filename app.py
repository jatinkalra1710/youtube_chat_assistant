import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Chat with YouTube", layout="wide")
st.title("📺 Chat with YouTube (Fast Mode)")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    api_key = st.text_input("Enter OpenAI API Key", type="password")


# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# ---------------- HELPER: EXTRACT VIDEO ID ----------------
def extract_video_id(url):
    parsed = urlparse(url)

    # youtu.be short links
    if "youtu.be" in parsed.netloc:
        return parsed.path[1:]

    # normal youtube links
    return parse_qs(parsed.query).get("v", [None])[0]


# ---------------- HELPER: GET TRANSCRIPT ----------------
def get_transcript(video_id):
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)

        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
        except:
            transcript = transcript_list.find_generated_transcript(['en'])

        data = transcript.fetch()

        full_text = " ".join([i.text for i in data])
        return full_text

    except Exception as e:
        raise Exception(f"Could not get transcript: {e}")


# ---------------- CACHE TRANSCRIPT ----------------
@st.cache_data(show_spinner=False)
def cached_transcript(video_id):
    return get_transcript(video_id)


# ---------------- CREATE VECTOR DB ----------------
def create_db(text, key):

    # protect from extremely long transcripts
    if len(text) > 200000:
        text = text[:200000]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(openai_api_key=key)

    return FAISS.from_documents(docs, embeddings)


# ---------------- INPUT ----------------
url = st.text_input("YouTube URL:")


if st.button("Analyze"):

    if not api_key:
        st.error("Need API Key")

    else:
        vid = extract_video_id(url)

        if not vid:
            st.error("Invalid YouTube URL")
        else:
            with st.spinner("Fetching transcript..."):
                try:
                    text = cached_transcript(vid)
                    st.session_state.vector_db = create_db(text, api_key)
                    st.success("Transcript loaded! Ask questions below.")

                except Exception as e:
                    st.error(str(e))


# ---------------- DISPLAY CHAT ----------------
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])


# ---------------- CHAT INPUT ----------------
if prompt := st.chat_input("Ask something about the video..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if st.session_state.vector_db:

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=api_key
        )

        docs = st.session_state.vector_db.similarity_search(prompt, k=3)
        context = "\n".join([d.page_content for d in docs])

        reply = llm.invoke([
            SystemMessage(content=f"Answer using this video transcript context:\n{context}"),
            HumanMessage(content=prompt)
        ]).content

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)

    else:
        st.warning("Please analyze a video first.")
