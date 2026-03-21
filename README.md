# YouTube Transcript RAG QA

Ask questions about any YouTube video. Answers are grounded in the transcript and include timestamps so you can verify exactly where in the video the information comes from.

## Quick Flow
```
User types: "What helped the speaker sleep better?"
                        ↓
          retriever.invoke(question)
          embeds the question → searches FAISS → finds 4 nearest vectors
          fetches the 4 Documents from docstore (text + metadata both come back)
                        ↓
          RunnableLambda(format_docs) runs HERE
          receives those 4 Documents
          reads metadata["start_time"] and metadata["end_time"] from each
          injects timestamps into page_content as plain text:
          "[0:00 - 0:42] hi friends today I'm going to share with..."
                        ↓
          this timestamped string fills {context} in the prompt
                        ↓
          LLM receives it as plain text and references timestamps in its answer
```
---

## How it works

```
YouTube URL → Extract video ID → Fetch transcript → Custom chunker →
HuggingFace embeddings → FAISS index → MMR retrieval → GPT-4o-mini → Answer with timestamps
```

The transcript is split into chunks that preserve start and end timestamps. When you ask a question, the most relevant chunks are retrieved from FAISS and passed to the LLM with their timestamps already embedded in the context — so the model can reference them naturally in its answer.

---

## Setup

**Install dependencies:**
```bash
pip install langchain langchain-community langchain-openai \
            sentence-transformers faiss-cpu \
            youtube-transcript-api gradio
```

**Run:**
```bash
python yt_rag.py
```

Then open `http://localhost:7860` in your browser.

---

## Usage

1. Paste any YouTube URL — standard, short (`youtu.be`), shorts, or embed format
2. Enter your [OpenRouter](https://openrouter.ai) API key
3. Type your question and click **Ask**

> Note: The video must have captions enabled. Most popular videos do. The embedding model (`all-MiniLM-L6-v2`) downloads once on first run (~90MB) and is cached after that.

---

## Screenshots

**In-context question — answer with timestamps:**
<img width="1854" height="874" alt="Screenshot 2026-03-21 123829" src="https://github.com/user-attachments/assets/f53c9b53-9762-4f9d-b770-e5f0b9e8fdae" />

---

**Out-of-context question — model says it doesn't know:**
<img width="1833" height="872" alt="Screenshot 2026-03-21 124227" src="https://github.com/user-attachments/assets/c52b924f-5098-42bd-b50f-170f071e5e2a" />

---

## Stack

| Component | Library |
|---|---|
| Transcript fetch | `youtube-transcript-api` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| LLM | GPT-4o-mini via OpenRouter |
| Chain | LangChain LCEL |
| UI | Gradio |
