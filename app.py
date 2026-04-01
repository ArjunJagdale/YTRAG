# =============================================================================
# YouTube Transcript RAG QA
# =============================================================================
# pip install langchain langchain-community langchain-openai \
#             sentence-transformers faiss-cpu youtube-transcript-api gradio
# python yt_rag.py
# =============================================================================

import re
from urllib.parse import urlparse, parse_qs

import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI


# =============================================================================
# STEP 1 — EXTRACT VIDEO ID
# =============================================================================

def extract_video_id(url: str) -> str:
    url = url.strip()

    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]

# =============================================================================
# STEP 2 — FETCH TRANSCRIPT
# =============================================================================

def fetch_transcript(video_id: str) -> list[dict]:
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    return transcript.to_raw_data()


# =============================================================================
# STEP 3 — BAKE TIMESTAMPS INTO TEXT
# =============================================================================

def sec_to_mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"

def bake_timestamps(segments: list[dict]) -> str:
    parts = []
    for seg in segments:
        ts = sec_to_mmss(seg["start"])
        parts.append(f"[{ts}] {seg['text']}")
    return "\n".join(parts)


# =============================================================================
# STEP 4 — SPLIT INTO CHUNKS
# =============================================================================

def build_chunks(text: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 200,
        separators = ["\n\n", "\n", " ", ""]
    )
    return splitter.create_documents([text])


# =============================================================================
# STEP 5 — BUILD FAISS INDEX
# =============================================================================

def build_index(documents: list[Document]) -> FAISS:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(documents, embedding_model)


# =============================================================================
# STEP 6 — RAG CHAIN
# =============================================================================

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(index: FAISS, api_key: str):
    retriever = index.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k": 4, "fetch_k": 10, "lambda_mult": 0.7},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant that answers questions about YouTube video content. "
         "You are given transcript excerpts with inline timestamps in [MM:SS] format. "
         "Answer the question using only the provided context. "
         "Reference the timestamps naturally in your answer so the user knows where to look. "
         "If the context does not contain the answer, say so clearly. "
         "Be concise and direct."),
        ("human", "Transcript excerpts:\n{context}\n\nQuestion: {question}")
    ])

    llm = ChatOpenAI(
        model = "openai/gpt-4o-mini",
        temperature = 0,
        max_tokens = 512,
        openai_api_key = api_key,
        openai_api_base = "https://openrouter.ai/api/v1"
    )

    return (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# =============================================================================
# MAIN HANDLER
# =============================================================================

def answer_question(youtube_url: str, question: str, api_key: str) -> str:
    if not youtube_url.strip():
        return "Please enter a YouTube URL."
    if not question.strip():
        return "Please enter a question."
    if not api_key.strip():
        return "Please enter your OpenRouter API key."

    try:
        video_id = extract_video_id(youtube_url)
    except ValueError as e:
        return f"Invalid URL: {e}"

    try:
        segments = fetch_transcript(video_id)
    except Exception as e:
        return f"Could not fetch transcript: {e}"

    text = bake_timestamps(segments)
    documents = build_chunks(text)
    index = build_index(documents)
    chain= build_rag_chain(index, api_key)

    # debug — print retrieved chunks to CLI
    retriever = index.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k": 4, "fetch_k": 10}
    )
    retrieved = retriever.invoke(question)
    print("\n" + "="*60)
    print("RETRIEVED CHUNKS")
    print("="*60)
    for i, doc in enumerate(retrieved):
        print(f"\n--- Chunk {i+1} ---")
        print(doc.page_content)
    print("="*60 + "\n")

    try:
        return chain.invoke(question)
    except Exception as e:
        return f"LLM error: {e}"


# =============================================================================
# GRADIO UI
# =============================================================================

def launch():
    with gr.Blocks(title="YouTube Transcript QA") as app:

        gr.Markdown("""
        # YouTube Transcript QA

        Ask any question about a YouTube video.
        Answers are grounded in the transcript and include timestamps.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                url_input = gr.Textbox(
                    label = "YouTube URL",
                    placeholder = "https://www.youtube.com/watch?v=..."
                )
                key_input = gr.Textbox(
                    label = "OpenRouter API Key",
                    placeholder = "sk-or-...",
                    type = "password"
                )
                question_input = gr.Textbox(
                    label = "Question",
                    placeholder = "What does the speaker say about X?",
                    lines = 3
                )
                submit_btn = gr.Button("Ask", variant="primary")

            with gr.Column(scale=1):
                answer_output = gr.Textbox(
                    label = "Answer",
                    lines = 12
                )

        submit_btn.click(
            fn = answer_question,
            inputs = [url_input, question_input, key_input],
            outputs = answer_output
        )

        gr.Markdown("""
        ---
        Pipeline: Extract video ID → Fetch transcript → Bake timestamps →
        RecursiveCharacterTextSplitter → HuggingFace embeddings → FAISS →
        MMR retrieval → GPT-4o-mini via OpenRouter
        """)

    app.launch()


if __name__ == "__main__":
    launch()
