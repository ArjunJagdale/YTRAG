# 🧠 YouTube Comments RAG Agent

### Retrieve • Understand • Analyze YouTube Conversations with a LangChain-Powered AI Agent

This project is an end-to-end **Retrieval-Augmented Generation (RAG)** system built for analyzing YouTube comments in a deeply semantic way.
It scrapes comments, embeds them, loads them into a vector store, and allows an intelligent agent to retrieve + reason over them using natural language queries.

The UI is powered by **Gradio**, while the retrieval, memory, prompting, and agent orchestration is handled by **LangChain**.

---
# Demo

https://github.com/user-attachments/assets/a8dd7f90-06d3-4114-917d-11600d90a902

---

# 🚀 Features

* **Scrape YouTube comments** (up to 500) using `youtube-comment-downloader`
* **Convert comments into LangChain documents** and chunk them semantically
* **Embed text using HuggingFace MiniLM** and store vectors in FAISS
* **Query comments using natural language**, not keywords
* **Agent with tools** for:

  * Searching by author
  * Searching by keyword
  * Retrieving all comments for holistic analysis
* **Conversational memory** so the agent remembers context over multiple queries
* **OpenRouter LLM integration** for cost-effective GPT-based inference
* **Clean UI** for loading, previewing, and chatting with comment data

---

# 🧩 How It Works (Technical Overview)

## 1. 🔍 Scraping Layer

Comments are fetched using **youtube-comment-downloader**, capturing:

* author
* comment text
* timestamp

They are loaded into a `pandas` DataFrame and stored globally.

---

## 2. 📄 Document Creation & Chunking (LangChain)

Each comment is converted into a **LangChain `Document`** with metadata:

```python
Document(
  page_content="Author: ... Comment: ... Time: ...",
  metadata={"author": ..., "time": ...}
)
```

These documents are then split using:

**`RecursiveCharacterTextSplitter`**

* chunk size: `1000`
* overlap: `100`
* hierarchical splitting rules for cleaner semantic units

This makes them suitable for high-quality retrieval.

---

## 3. 🧬 Embeddings (HuggingFace)

Embeddings are generated using:

**`HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`**

Why?

* Fast
* Lightweight
* High quality for semantic similarity
* Works offline / no paid API

*HuggingFace is not the orchestrator — it simply powers the embedding step.*

---

## 4. 📚 Vector Store (FAISS)

All chunk embeddings are stored in a **FAISS vector store**, enabling:

* fast similarity search
* dense retrieval
* robust matching beyond keywords

LangChain’s vectorstore abstraction wraps FAISS, making querying seamless.

---

## 5. 🛠️ Tooling Layer (LangChain Tools)

Three custom tools are exposed to the agent:

1. `search_comments_by_author(author_name)`
2. `search_comments_by_keyword(keyword)`
3. `get_all_comments(limit=100)`

LangChain registers these as **runnable tools**, enabling the agent to decide:

* when to call a tool
* which tool to call
* how to use tool outputs in reasoning

This is true *agentic behavior*, not simple prompt chaining.

---

## 6. 🧠 Agent + Memory (LangChain)

A **Tool-Calling Agent** is created via:

```
create_tool_calling_agent()
```

Backed by:

* **ConversationBufferMemory** (chat history persistence)
* **ChatPromptTemplate** (system + user + memory + scratchpad)
* **OpenRouter GPT-4o-mini** as the LLM backend

The agent chooses retrieval strategies automatically:

* Direct lookup (if query mentions specific authors/keywords)
* Holistic reading (via `get_all_comments`)
* Tool sequences (multi-step reasoning)

---

## 7. 💬 Natural-Language Interaction

Users can ask:

* “What are people complaining about?”
* “Any helpful explanations in the comments?”
* “Show me sarcastic reactions”
* “What themes dominate the discussion?”
* “Did someone ask a question?”

The agent:

1. Retrieves relevant chunks from FAISS
2. Calls tools where needed
3. Synthesizes a reasoning-rich answer via OpenRouter

---

## 8. 🖥️ Frontend (Gradio)

A polished UI with three stages:

1. **Load Comments & Initialize Agent**
2. **Preview Comments**
3. **Interact with the RAG Agent**

This keeps the workflow simple and intuitive.

---

# 🛠️ Installation

### 1. Clone the repo

```bash
git clone <repo-url>
cd <project>
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
python app.py
```

---

# 🔑 API Key (OpenRouter)

To use GPT-4o-mini or any model via OpenRouter, set:

```
sk-or-v1-xxxxxxxxxxx
```

OpenRouter provides:

* multi-provider unified API
* cheaper GPT-style models
* rate-limit transparency

*They power inference — LangChain handles orchestration.*

---

# 📦 Requirements

```
langchain==0.3.27
langchain-core==0.3.76
langchain-community==0.3.29
langchain-openai==0.3.11
langchain-text-splitters==0.3.11
youtube-comment-downloader
gradio
langchain-huggingface
faiss-cpu
sentence-transformers
pandas
```

---

# 🧪 Architecture Overview

```
          ┌──────────────────────────┐
          │  YouTube Comment Scraper │
          └─────────────┬────────────┘
                        ▼
             Pandas DataFrame
                        ▼
      ┌──────────────────────────────────┐
      │    LangChain Document Layer      │
      └──────────────────────────────────┘
                        ▼
      RecursiveCharacterTextSplitter
                        ▼
                HF MiniLM Embeddings
                        ▼
                   FAISS Vector DB
                        ▼
     ┌──────────────────────────────────┐
     │      LangChain Agent System      │
     │  - Tools                         │
     │  - Memory                        │
     │  - OpenRouter LLM                │
     └──────────────────────────────────┘
                        ▼
                 Natural Language Answers
```

---

# 🏁 Summary

This system combines:

### LangChain — the brain

Document pipelines, retrieval abstraction, tool-based agent reasoning, memory, orchestration

### HuggingFace — the muscle

Embeddings for vector similarity

### OpenRouter — the voice

LLM reasoning & natural language dialog

Together, they form a practical, flexible, and elegant RAG workflow specialized for YouTube comment analysis.

---
