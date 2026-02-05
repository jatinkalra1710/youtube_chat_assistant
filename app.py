import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from youtube_transcript_api import YouTubeTranscriptApi

st.set_page_config(page_title="Chat with YouTube", layout="wide")
st.title("📺 Chat with YouTube (Fast Mode)")

with st.sidebar:
    api_key = st.text_input("Enter OpenAI API Key", type="password")

if "messages" not in st.session_state: st.session_state.messages = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None

def get_transcript(video_id):
    try:
        # Use list_transcripts to be safer
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try finding English manually first, then fallback to auto-generated
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
        except:
            transcript = transcript_list.find_generated_transcript(['en'])
            
        return " ".join([i['text'] for i in transcript.fetch()])
    except Exception as e:
        raise Exception(f"Could not get transcript: {e}")

def create_db(text, key):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=key))

url = st.text_input("YouTube URL:")
if st.button("Analyze"):
    if not api_key: st.error("Need API Key")
    elif "v=" not in url and "youtu.be" not in url: st.error("Invalid URL")
    else:
        vid = url.split("v=")[1].split("&")[0] if "v=" in url else url.split("/")[-1].split("?")[0]
        with st.spinner("Fetching transcript..."):
            try:
                text = get_transcript(vid)
                st.session_state.vector_db = create_db(text, api_key)
                st.success("Done! Chat below.")
            except Exception as e: st.error(str(e))

for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if st.session_state.vector_db:
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
        docs = st.session_state.vector_db.similarity_search(prompt, k=3)
        context = "\n".join([d.page_content for d in docs])
        reply = llm.invoke([SystemMessage(content=f"Context: {context}"), HumanMessage(content=prompt)]).content
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)
