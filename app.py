# =============================================================================
# YouTube Transcript RAG QA
# =============================================================================
# Local install:
#   pip install langchain langchain-community langchain-openai \
#               sentence-transformers faiss-cpu \
#               youtube-transcript-api gradio
#
# Run:
#   python yt_rag.py
# =============================================================================

import re
from urllib.parse import urlparse, parse_qs

import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI


# =============================================================================
# STEP 1 — EXTRACT VIDEO ID
# =============================================================================

def extract_video_id(url: str) -> str:
    """
    Parses a YouTube URL and returns just the video ID string.

    url.strip()
        Removes any leading or trailing whitespace the user may have added.

    if "youtu.be" in url
        Detects the short URL format: https://youtu.be/VIDEO_ID?si=...
        url.split("/")[-1]  → splits on "/" and takes the last segment → "VIDEO_ID?si=..."
        .split("?")[0]      → removes everything from "?" onwards → leaves just the video ID.

    urlparse(url)
        Breaks the full URL into named components: scheme, netloc, path, query, fragment.
        e.g. urlparse("https://youtube.com/watch?v=abc123") gives:
             path="/watch", query="v=abc123"

    if parsed.query
        Checks whether a query string exists (e.g. "v=VIDEO_ID&t=30s").
        parse_qs(parsed.query) → converts the query string into a dict:
            {"v": ["VIDEO_ID"], "t": ["30s"]}
        qs["v"][0] → takes the first value of key "v" = the video ID.

    re.search(r"/(shorts|embed)/([^/?]+)", parsed.path)
        Handles /shorts/VIDEO_ID and /embed/VIDEO_ID URL formats.
        (shorts|embed)  → matches either word literally.
        ([^/?]+)        → captures one or more characters that are NOT "/" or "?",
                          which isolates the video ID.
        match.group(2)  → returns the second capture group = the video ID.

    raise ValueError
        If none of the formats matched, the URL is unrecognised.
    """
    url = url.strip()

    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]

    parsed = urlparse(url)

    if parsed.query:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]

    match = re.search(r"/(shorts|embed)/([^/?]+)", parsed.path)
    if match:
        return match.group(2)

    raise ValueError("Could not extract video ID from the provided URL.")


# =============================================================================
# STEP 2 — FETCH TRANSCRIPT
# =============================================================================

def fetch_transcript(video_id: str) -> list[dict]:
    """
    Fetches raw caption data for a YouTube video and returns it as a list of dicts.

    YouTubeTranscriptApi()
        Instantiates the API client. No authentication is needed for public videos.

    api.fetch(video_id)
        Makes an HTTP request to YouTube's internal caption endpoint using the video ID.
        Returns a Transcript object — not plain Python data yet.

    transcript.to_raw_data()
        Converts the Transcript object into a plain list of dicts.
        Each dict has exactly three keys:
            "text"     → the caption text for this time segment
            "start"    → float, seconds from the start of the video where this segment begins
            "duration" → float, how many seconds this segment lasts
        Example: {"text": "hello everyone", "start": 4.64, "duration": 3.2}
    """
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    return transcript.to_raw_data()


# =============================================================================
# STEP 3 — CUSTOM CHUNKER
# =============================================================================

def build_chunks(segments: list[dict], max_chars: int = 500) -> list[Document]:
    """
    Groups transcript segments into LangChain Document objects, preserving timestamps.

    Why not RecursiveCharacterTextSplitter?
        The standard splitter takes plain text and discards all metadata.
        Passing segments through it would lose start_time and end_time completely.
        Since timestamps are the core feature of this tool, we need a custom chunker
        that accumulates segments by character count (same concept as chunk_size)
        while tracking the timestamp of the first and last segment in each chunk.

    Why no overlap?
        Overlapping transcript chunks would feed the LLM the same spoken content
        twice but with different timestamps, producing confusing answers.

    documents, buffer, start_time
        documents  → accumulates the final list of Document objects to return.
        buffer     → temporary list holding segment texts for the chunk being built.
        start_time → records the "start" time of the first segment in the current chunk.

    if start_time is None: start_time = seg["start"]
        On the very first segment of each new chunk, capture its start time.
        This becomes the start_time in the chunk's metadata.

    buffer.append(seg["text"])
        Adds the current segment's text to the running buffer.

    current_length = sum(len(t) for t in buffer)
        Counts the total number of characters accumulated in the buffer.
        This is the equivalent of chunk_size in RecursiveCharacterTextSplitter.

    if current_length >= max_chars
        Once the buffer hits or exceeds the character limit, flush it to a Document.
        end_time = seg["start"] + seg["duration"]
            The end time is the start of the last segment plus how long it lasts.
        Document(page_content, metadata)
            LangChain's standard document object. page_content holds the text,
            metadata holds the timestamp dict.
        Reset buffer and start_time to start the next chunk from scratch.

    if buffer (after the loop ends)
        The final group of segments may not have reached max_chars.
        Whatever remains in the buffer is flushed into one last Document.
    """
    documents  = []
    buffer     = []
    start_time = None

    for seg in segments:
        if start_time is None:
            start_time = seg["start"]

        buffer.append(seg["text"])
        current_length = sum(len(t) for t in buffer)

        if current_length >= max_chars:
            end_time = seg["start"] + seg["duration"]
            documents.append(Document(
                page_content = " ".join(buffer),
                metadata     = {"start_time": start_time, "end_time": end_time}
            ))
            buffer     = []
            start_time = None

    if buffer:
        end_time = segments[-1]["start"] + segments[-1]["duration"]
        documents.append(Document(
            page_content = " ".join(buffer),
            metadata     = {"start_time": start_time, "end_time": end_time}
        ))

    return documents


# =============================================================================
# STEP 4 — FORMAT DOCS WITH TIMESTAMPS
# =============================================================================

def sec_to_mmss(seconds: float) -> str:
    """
    Converts a float number of seconds into a MM:SS formatted string.

    int(seconds)
        Truncates the float to whole seconds, discarding milliseconds.

    s // 60
        Integer division — gives the number of complete minutes.
        e.g. 83 // 60 = 1 minute.

    s % 60
        Modulo — gives the leftover seconds after full minutes are removed.
        e.g. 83 % 60 = 23 seconds.

    :02d
        Format specifier that zero-pads the seconds to always be two digits.
        e.g. 4 → "04", ensuring "1:04" not "1:4".
    """
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def format_docs(docs: list[Document]) -> str:
    """
    Converts a list of retrieved Documents into a single timestamped context string.

    Why this is needed:
        FAISS retrieves Document objects. The LLM only ever receives page_content —
        it never sees metadata directly. Without this step, start_time and end_time
        stay hidden in metadata and the LLM cannot reference timestamps in its answer.

    sec_to_mmss(doc.metadata["start_time"]) / ["end_time"]
        Converts the float timestamps stored during chunking into readable MM:SS strings.

    f"[{start} - {end}] {doc.page_content}"
        Prepends the timestamp range directly to the chunk's text.
        The LLM now sees e.g. "[1:23 - 1:51] the speaker talks about sleep tracking..."
        and can reference these markers naturally in its response.

    "\n\n".join(chunks)
        Joins all formatted chunks with a blank line between them,
        making each chunk visually distinct in the context the LLM receives.
    """
    chunks = []
    for doc in docs:
        start = sec_to_mmss(doc.metadata["start_time"])
        end   = sec_to_mmss(doc.metadata["end_time"])
        chunks.append(f"[{start} - {end}] {doc.page_content}")
    return "\n\n".join(chunks)


# =============================================================================
# STEP 5 — BUILD FAISS INDEX
# =============================================================================

def build_index(documents: list[Document]) -> FAISS:
    """
    Embeds all Document chunks into vectors and stores them in a FAISS index.

    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Loads a local sentence-transformer model that converts text to dense float vectors.
        "all-MiniLM-L6-v2" produces 384-dimensional vectors — compact and fast.
        Runs entirely on CPU with no API key required.
        The model downloads once (~90MB) on first use and is cached for future runs.

    FAISS.from_documents(documents, embedding_model)
        Iterates over every Document and calls embedding_model.embed_query(page_content).
        Stores the resulting vectors in a FAISS IndexFlatL2 (exact nearest-neighbour search).
        Also stores the original Document objects in an in-memory docstore so they can be
        returned alongside their vectors when a query is made.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(documents, embedding_model)


# =============================================================================
# STEP 6 — BUILD RAG CHAIN
# =============================================================================

def build_rag_chain(index: FAISS, api_key: str):
    """
    Constructs the full LCEL retrieval-augmented generation pipeline.

    index.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
        Wraps the FAISS index in a LangChain Retriever interface.
        search_type="mmr"
            Maximal Marginal Relevance: selects results that are both relevant to the
            query AND diverse from each other, avoiding returning near-duplicate chunks.
        fetch_k=10
            MMR first pulls 10 candidate chunks from FAISS using raw similarity search.
        k=4
            From those 10 candidates, MMR re-ranks and returns the best 4.

    ChatPromptTemplate.from_messages([("system", ...), ("human", ...)])
        Defines a two-role prompt template.
        "system" message → instructs the LLM to answer only from the provided context,
            reference timestamps, and be concise.
        "human" message  → injects {context} (the timestamped transcript chunks) and
            {question} (the user's query) at runtime.

    ChatOpenAI(openai_api_base="https://openrouter.ai/api/v1")
        ChatOpenAI is built for OpenAI's API format. Overriding openai_api_base
        redirects all requests to OpenRouter, which supports the same API contract
        but routes to many different model providers.
        Swapping models is just changing the model= string — nothing else changes.

    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        This dict is the entry point of the LCEL chain. The input is the user's question string.

        "context" key:
            retriever receives the question string → queries FAISS → returns top-k Documents.
            RunnableLambda(format_docs) wraps format_docs (a plain Python function) so it
            can participate in the | pipe chain.
            The result is a single string of timestamped transcript chunks.

        "question" key:
            RunnablePassthrough() takes the input question and passes it through unchanged,
            so it's available as {question} in the prompt template.

    | prompt | llm | StrOutputParser()
        prompt          → fills {context} and {question} into the template → list of messages.
        llm             → sends the messages to the model → returns an AIMessage object.
        StrOutputParser → extracts the .content string from the AIMessage → plain text output.
    """
    retriever = index.as_retriever(
        search_type   = "mmr",
        search_kwargs = {"k": 4, "fetch_k": 10}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant that answers questions about YouTube video content. "
         "You are given transcript excerpts with timestamps in [MM:SS - MM:SS] format. "
         "Answer the question using only the provided context. "
         "Reference the timestamps naturally in your answer so the user knows where to look. "
         "If the context does not contain the answer, say so clearly. "
         "Be concise and direct."),
        ("human", "Transcript excerpts:\n{context}\n\nQuestion: {question}")
    ])

    llm = ChatOpenAI(
        model           = "openai/gpt-4o-mini",
        temperature     = 0,
        max_tokens      = 512,
        openai_api_key  = api_key,
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

    documents = build_chunks(segments, max_chars=500)
    index     = build_index(documents)
    chain     = build_rag_chain(index, api_key)

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
                    label       = "YouTube URL",
                    placeholder = "https://www.youtube.com/watch?v=..."
                )
                key_input = gr.Textbox(
                    label       = "OpenRouter API Key",
                    placeholder = "sk-or-...",
                    type        = "password"
                )
                question_input = gr.Textbox(
                    label       = "Question",
                    placeholder = "What does the speaker say about X?",
                    lines       = 3
                )
                submit_btn = gr.Button("Ask", variant="primary")

            with gr.Column(scale=1):
                answer_output = gr.Textbox(
                    label = "Answer",
                    lines = 12
                )

        submit_btn.click(
            fn      = answer_question,
            inputs  = [url_input, question_input, key_input],
            outputs = answer_output
        )

        gr.Markdown("""
        ---
        Pipeline: Extract video ID → Fetch transcript → Custom chunker →
        HuggingFace embeddings → FAISS → MMR retrieval → GPT-4o-mini via OpenRouter
        """)

    app.launch()


if __name__ == "__main__":
    launch()
