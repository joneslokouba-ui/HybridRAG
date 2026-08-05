import streamlit as st
from src.rag_pipeline import build_vectorstore, query_rag
import re


def highlight_text(text, query):
    keywords = query.split()
    for word in keywords:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(f"**{word}**", text)
    return text


@st.cache_resource
def load_vectorstore(file_paths):
    return build_vectorstore(list(file_paths))


st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")

st.title("📄 Hybrid RAG Assistant (Groq + LangChain)")
st.markdown("""
**A multi-document Retrieval-Augmented Generation (RAG) system  
with conversational memory, hybrid similarity scoring, and source transparency.**""")

# --- Groq API key from Streamlit Secrets ---
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error(
        "No GROQ_API_KEY found in Secrets. Add it under App settings -> Secrets "
        "as: GROQ_API_KEY = \"your-key-here\""
    )
    st.stop()

uploaded_files = st.file_uploader(
    "Upload text files",
    type=["txt"],
    accept_multiple_files=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

vectorstore = None

if uploaded_files:
    file_paths = []
    for file in uploaded_files:
        file_path = f"temp_{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.read())
        file_paths.append(file_path)

    try:
        vectorstore = load_vectorstore(tuple(file_paths))
    except Exception as e:
        st.error(f"Vectorstore build failed: {e}")
        st.stop()

    st.success(f"Indexed {len(file_paths)} file(s) successfully.")
    st.divider()

if st.session_state.chat_history:
    st.write("### 💬 Conversation History")
    for turn in st.session_state.chat_history:
        st.markdown(f"**Q:** {turn['question']}")
        st.markdown(f"**A:** {turn['answer']}")
        st.write("---")

query = st.text_input("Ask a question about the documents")
alpha = st.slider(
    "Hybrid Weight (Vector vs Keyword)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
)

if query:
    if vectorstore is None:
        st.warning("Upload at least one document before asking a question.")
        st.stop()

    try:
        response, docs, scores = query_rag(
            vectorstore,
            query,
            st.session_state.chat_history,
            groq_api_key=groq_api_key,
            alpha=alpha,
        )
    except Exception as e:
        st.error(f"Query failed: {e}")
        st.stop()

    st.session_state.chat_history.append({
        "question": query,
        "answer": response,
    })

    st.write("### 🤖 Answer")
    st.write(response)

    st.write("### 📚 Retrieved Sources")
    st.caption("Hybrid retrieval: vector similarity + keyword reinforcement")
    for i, (doc, score) in enumerate(zip(docs, scores)):
        source_file = doc.metadata.get("source_file", "Unknown")
        st.markdown(
            f"**Source {i+1} | File: {source_file} | Similarity Score: {score:.4f}**"
        )
        highlighted = highlight_text(doc.page_content, query)
        st.markdown(highlighted)
        st.write("---")
