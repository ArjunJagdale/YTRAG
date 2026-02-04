# YT-RAG

**Retrieval-Augmented Generation for YouTube videos using LangChain**

Ask questions about any YouTube video and get answers grounded in the actual transcript — with exact timestamps.

---
## Demo
> Input - OPENROUTER API | YT URL | QUESTION
>
> <img width="1115" height="599" alt="image" src="https://github.com/user-attachments/assets/c03229c0-f768-4498-964c-990640fb648c" />
>
> OUTPUT - ANSWER GENERATED BASED ON TRANSCRIPT

## What It Does

Input: YouTube URL + Question  
Output: Answer based **only** on retrieved transcript chunks

No hallucinations. No guessing. Pure RAG.

---

## How We Built RAG with LangChain

```
Ingest → Chunk → Embed → Store → Retrieve → Prompt → Generate
```

### 1. **Ingest** — Fetch Transcript
```python
from youtube_transcript_api import YouTubeTranscriptApi

segments = fetch_transcript(youtube_url)
# Returns: [{"text": "...", "start": 0.0, "duration": 2.5}, ...]
```

### 2. **Chunk** — Timestamp-Aware Splitting
```python
from langchain.schema import Document

documents = build_chunks(segments, max_chars=500)
# Returns: Document(page_content="...", metadata={"start_time": 0, "end_time": 15})
```

We don't use `RecursiveCharacterTextSplitter` because we need **custom logic** to preserve timestamps while chunking.

### 3. **Embed** — Generate Vectors
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)
```

LangChain handles the embedding API calls and batch processing.

### 4. **Store** — Vector Database
```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)
```

FAISS (via LangChain) stores embeddings for fast similarity search.

### 5. **Retrieve** — Semantic Search
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
docs = retriever.get_relevant_documents(question)
```

LangChain's retriever interface makes it easy to swap search strategies.

### 6. **Prompt** — Context Injection
```python
from langchain.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_template("""
You are answering questions about a YouTube video.
Use ONLY the retrieved transcript chunks below.

Transcript chunks:
{context}

Question:
{question}
""")

messages = PROMPT.format_messages(context=context, question=question)
```

Retrieved chunks are formatted with timestamps and injected into the prompt.

### 7. **Generate** — LLM Response
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

response = llm.invoke(messages)
```

LangChain abstracts the LLM API call and response parsing.

---

## Why LangChain?

LangChain provides:
- **`Document` abstraction** — standardized text + metadata handling
- **Embedding interface** — swap models without rewriting code
- **Vector store integration** — FAISS, Pinecone, Chroma, etc.
- **Retriever pattern** — clean separation of search logic
- **Prompt templates** — reusable, testable prompt engineering

Result: **Modular, inspectable, production-ready RAG**.

---

## Run It

### Local
```bash
pip install -r requirements.txt
python app.py
```

### Google Colab
```bash
!pip install langchain langchain-community langchain-openai youtube-transcript-api faiss-cpu gradio
# Then run youtube_rag_qa_colab.py
```

Get your **OpenRouter API key** at [openrouter.ai](https://openrouter.ai)

---

## 🛠️ Tech Stack

- **LangChain** — RAG orchestration
- **OpenRouter** — LLM + embeddings API
- **FAISS** — vector store
- **YouTube Transcript API** — data ingestion
- **Gradio** — UI

---

## Key Design Decisions

 **Timestamp preservation** — Custom chunking logic maintains exact timing metadata  
 **Retrieval-only answers** — LLM can't hallucinate; it only uses retrieved context  
 **No caching** — Each query re-processes the video (trade-off for simplicity)  
 **LangChain abstractions** — Makes the system extensible and testable  

---

## Example

**URL:** `https://youtu.be/0bUieoJ6FI4?si=09lUFlLLX-tnKKZV`  
**Question:** "for roasted veg, lentil % Chickpea bowl, what temperature to consider for pre heating? also tell me more about the ingredients!"  
**Answer:**
> For the roasted vegetable, chickpea, and lentil bowl, you should preheat your oven to 400°F. The ingredients include half a cauliflower, various vegetables that are roasted to enhance their sweetness and juiciness, black lentils which are high in protein and fiber...

---

## License

MIT

---

**Built by [ArjunJagdale](https://github.com/ArjunJagdale)**
