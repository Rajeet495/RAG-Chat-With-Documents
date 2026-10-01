import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Chat With Your Documents",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Chat With Your Documents")
st.caption("Upload a PDF and ask questions about its content.")


# -----------------------------
# Groq Client
# -----------------------------

try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()


# -----------------------------
# Load Embedding Model
# -----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# -----------------------------
# Text Chunking
# -----------------------------

def create_chunks(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -----------------------------
# Extract PDF Text
# -----------------------------

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Create Vector Database
# -----------------------------

def create_vector_database(chunks):

    embeddings = embedding_model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# -----------------------------
# Retrieve Relevant Chunks
# -----------------------------

def retrieve_context(query, chunks, index, k=3):

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        min(k, len(chunks))
    )

    retrieved_chunks = []

    for index_number in indices[0]:

        if index_number != -1:
            retrieved_chunks.append(
                chunks[index_number]
            )

    return retrieved_chunks


# -----------------------------
# Generate Answer
# -----------------------------

def generate_answer(query, context):

    context_text = "\n\n".join(context)

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the provided document context.

If the answer cannot be found in the context, say:

"I could not find the answer in the uploaded document."

Do not invent information.

DOCUMENT CONTEXT:
{context_text}

USER QUESTION:
{query}

ANSWER:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content


# -----------------------------
# PDF Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)


# -----------------------------
# Process Document
# -----------------------------

if uploaded_file:

    if (
        "processed_file" not in st.session_state
        or st.session_state.processed_file
        != uploaded_file.name
    ):

        with st.spinner("Processing document..."):

            try:

                text = extract_pdf_text(
                    uploaded_file
                )

                if not text.strip():

                    st.error(
                        "No readable text was found in this PDF."
                    )

                    st.stop()

                chunks = create_chunks(text)

                index = create_vector_database(
                    chunks
                )

                st.session_state.chunks = chunks

                st.session_state.index = index

                st.session_state.processed_file = (
                    uploaded_file.name
                )

                st.success(
                    f"Document processed successfully! "
                    f"{len(chunks)} chunks created."
                )

            except Exception as e:

                st.error(
                    "An error occurred while processing the document."
                )

                st.code(str(e))


# -----------------------------
# Question Answering
# -----------------------------

if (
    "chunks" in st.session_state
    and "index" in st.session_state
):

    st.divider()

    st.subheader("💬 Ask Questions")

    question = st.chat_input(
        "Ask something about your document..."
    )

    if question:

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching document and generating answer..."
            ):

                try:

                    retrieved_chunks = retrieve_context(
                        question,
                        st.session_state.chunks,
                        st.session_state.index
                    )

                    answer = generate_answer(
                        question,
                        retrieved_chunks
                    )

                    st.markdown(answer)

                    with st.expander(
                        "🔍 View Retrieved Context"
                    ):

                        for i, chunk in enumerate(
                            retrieved_chunks,
                            start=1
                        ):

                            st.markdown(
                                f"**Chunk {i}**"
                            )

                            st.write(chunk)

                except Exception as e:

                    st.error(
                        "Something went wrong while generating the answer."
                    )

                    st.code(str(e))