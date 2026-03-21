# YouTube Transcript RAG QA

Ask questions about any YouTube video. Answers are grounded in the transcript and include timestamps.

---

## How it works

```
YouTube URL → Extract video ID → Fetch transcript → Bake timestamps into text →
RecursiveCharacterTextSplitter → HuggingFace embeddings → FAISS →
MMR retrieval → GPT-4o-mini via OpenRouter → Answer with timestamps
```

Timestamps are baked directly into the transcript text as `[MM:SS]` markers before splitting. This means `RecursiveCharacterTextSplitter` can split with overlap freely — timestamps survive in every chunk and the LLM can reference them naturally in its answer.

---

## Setup

```bash
pip install langchain langchain-community langchain-openai \
            sentence-transformers faiss-cpu \
            youtube-transcript-api gradio
```

```bash
python yt_rag.py
```

Open `http://localhost:7860` in your browser.

---

## Usage

1. Paste any YouTube URL
2. Enter your [OpenRouter](https://openrouter.ai) API key
3. Type a question and click **Ask**

The video must have captions enabled. The embedding model downloads once on first run (~90MB) and is cached after that.

---

## Screenshots

**In-context question — answer with timestamps:**

<img width="1894" height="882" alt="image" src="https://github.com/user-attachments/assets/89d1e95a-98b0-4c4f-bd0b-c28a36ec696d" />

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
