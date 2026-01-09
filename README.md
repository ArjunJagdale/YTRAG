# 🎥 YouTube Comments RAG Agent

**Retrieve • Understand • Analyze YouTube Conversations with AI**

A production-grade RAG (Retrieval-Augmented Generation) system for intelligent analysis of YouTube comments through natural language queries. Built with LangChain, OpenRouter, and FAISS vector search.

---

## 📺 Demo

https://github.com/user-attachments/assets/a8dd7f90-06d3-4114-917d-11600d90a902

---

## 🎯 Architecture Overview

```
YouTube Comments → Document Processing → Vector Embeddings → FAISS Index → Agent Reasoning → Natural Language Answers
```

**Complete RAG Pipeline:**
1. **Scrape** comments using `youtube-comment-downloader`
2. **Chunk** documents semantically with `RecursiveCharacterTextSplitter`
3. **Embed** using HuggingFace `all-MiniLM-L6-v2`
4. **Store** in FAISS for fast similarity search
5. **Retrieve** via hybrid tools + vector search
6. **Generate** contextual answers with LangChain agents

---

## 🚀 Key Features

### 1. **LLM Routing & Fallback System**
Production-ready infrastructure for model resilience:

```python
Primary Model → Automatic Fallback → Secondary Model
       ↓
Evaluation Logging (CSV)
```

- **Sequential Routing**: Auto-fallback on failure (gpt-4o-mini → mistral-7b)
- **Observability**: Logs latency, success rate, output metrics for every request
- **Future-Ready**: Architected for eval-driven routing and cost-aware selection

**Evaluation Metrics (logged to `llm_eval_logs.csv`):**
- Request latency
- Success/failure status
- Output token length
- Error traces

---

### 2. **Hybrid Retrieval System**
Combines vector search with structured tools:

| Tool | Purpose |
|------|---------|
| `search_comments_by_author()` | Find all comments by specific user |
| `search_comments_by_keyword()` | Search by terms/phrases |
| `get_all_comments()` | Retrieve full dataset for holistic analysis |

**Why Hybrid?** Vector search misses exact matches; tools ensure precision alongside semantic understanding.

---

### 3. **Intelligent Agent System**
LangChain agent with tool orchestration:

- **Conversational Memory**: Multi-turn context awareness
- **Tool Selection**: Auto-chooses optimal retrieval strategy
- **Error Handling**: Graceful degradation with retry logic
- **Smart Analysis**: Understands sentiment, tone, intent beyond keywords

**Agent Capabilities:**
- Identifies questions, explanations, complaints without keyword reliance
- Synthesizes multi-source information
- Provides evidence-backed structured analysis

---

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM Provider** | OpenRouter | Multi-model access + routing |
| **Vector Store** | FAISS | High-speed similarity search |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Semantic text representation |
| **Agent Framework** | LangChain | Tool orchestration + memory |
| **UI** | Gradio | Interactive web interface |
| **Data Source** | `youtube-comment-downloader` | Comment extraction |

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenRouter API key ([get one here](https://openrouter.ai/))

### Setup

```bash
# Clone repository
git clone https://github.com/ArjunJagdale/YTRAG.git
cd YTRAG

# Install dependencies
pip install -r requirements.txt

# Add your OpenRouter API key
# (paste directly in the UI when launching)
```

### Dependencies
```txt
langchain==0.3.27
langchain-openai==0.3.11
langchain-community==0.3.29
langchain-huggingface
youtube-comment-downloader
gradio
faiss-cpu
sentence-transformers
pandas
```

---

## 🎮 Usage

### Launch
```bash
python app.py
```

Interface opens at `http://localhost:7860`

### Workflow

**Step 1: Load Comments**
1. Enter OpenRouter API key
2. Paste YouTube URL
3. Set max comments (10-500)
4. Click "Load Comments & Initialize Agent"

**What Happens:**
- Comments scraped and converted to LangChain Documents
- Text chunked (1000-char chunks, 100-char overlap)
- Embedded using `all-MiniLM-L6-v2`
- FAISS index built
- Agent initialized with routing + tools

**Step 2: Query with Natural Language**

```
"What's the overall sentiment?"
"Has anyone asked questions?"
"Show me detailed explanations"
"Find complaints about X"
"What topics are most discussed?"
```

---

## 🏗️ Technical Deep Dive

### Document Processing

**Chunking Strategy:**
- **Size**: 1000 chars (balances context vs. precision)
- **Overlap**: 100 chars (prevents boundary info loss)
- **Separators**: `["\n\nAuthor:", "\n\n", "\n", " "]` (semantic boundaries)

**Why all-MiniLM-L6-v2?**
- 14,000+ tokens/sec inference
- 384-dim embeddings with strong semantic capture
- 80MB model (local, no API costs)

### LLM Routing Architecture

```python
class LLMRouter:
    """Sequential router with evaluation hooks"""
    def generate(self, messages):
        for provider in self.providers:
            try:
                response = provider.generate(messages)
                self._log_eval(success=True, latency=...)
                return response
            except:
                self._log_eval(success=False, ...)
        raise RuntimeError("All providers failed")
```

**Routing Strategy (Current):**
- Sequential fallback
- Manual heuristic selection

**Planned:**
- Eval-driven routing
- Cost-aware model selection
- Query complexity-based routing

---

## 📈 Evaluation & Monitoring

### Performance Analysis
```python
import pandas as pd

logs = pd.read_csv('llm_eval_logs.csv')

# Success rate by provider
logs.groupby('provider')['success'].mean()

# Average latency
logs[logs['success'] == True]['latency_sec'].mean()

# Failure patterns
logs[logs['success'] == False]['error'].value_counts()
```

---

## 🔮 Roadmap

- [ ] Eval-driven routing based on query complexity
- [ ] Cost tracking per request
- [ ] Response streaming for long-form analysis
- [ ] Multi-video comparison
- [ ] Temporal sentiment tracking

---

## 🎓 Resources

- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [FAISS Documentation](https://faiss.ai/)
- [OpenRouter API Docs](https://openrouter.ai/docs)

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

Built with [LangChain](https://www.langchain.com/) • [OpenRouter](https://openrouter.ai/) • [FAISS](https://faiss.ai/) • [HuggingFace](https://huggingface.co/) • [Gradio](https://gradio.app/)

---

**For questions or feedback:** [Open an issue](https://github.com/ArjunJagdale/YTRAG/issues)
