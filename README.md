# 🎥 YouTube Comments RAG Chatbot

This project enables you to **scrape YouTube comments**, embed them using **HuggingFace sentence embeddings**, store them in a **FAISS vector database**, and chat with them using **Retrieval-Augmented Generation (RAG)** powered by **LangChain** and **OpenRouter LLMs**.  
The interface is built with **Gradio** for easy interaction — no terminal knowledge required.

---

## Demo Video

https://github.com/user-attachments/assets/0b639356-c0d6-4979-9444-95fae4e46a6c

## 🚀 Features

- **Scrape YouTube comments** from any public video  
- **Convert comments into embeddings** using `all-MiniLM-L6-v2`
- Store embeddings in an in-memory **FAISS vector store**
- **Query comments intelligently** with a RAG pipeline
- Use **LangChain RetrievalQA** to ground responses in actual comments
- Chat with insights like:
  - What is the general sentiment?
  - Who is mentioned most?
  - Are users discussing a specific topic?
  - What are the most common reactions or opinions?

---

## 🧠 How RAG Works Here

| Step | Process |
|-----|---------|
| 1. Scrape | Pull up to N comments using `youtube-comment-downloader` |
| 2. Embed | Convert comments into vector embeddings using HuggingFace |
| 3. Store | Save vectors in a FAISS similarity index |
| 4. Retrieve | For each user query, find the most relevant comments |
| 5. Generate | Use an LLM (via OpenRouter) to answer using retrieved context |

This ensures the chatbot **does not hallucinate** and answers **based directly on real YouTube comments**.

---

## 🏗 Tech Stack

| Component | Library / Model |
|----------|----------------|
| LLM | `ChatOpenAI` via OpenRouter |
| Embeddings | `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) |
| Vector DB | `FAISS` |
| RAG Chain | `LangChain` `RetrievalQA` |
| UI | `Gradio` |
| Data Source | `youtube-comment-downloader` |

---

## 🔑 Requirements

Install dependencies:

```bash
pip install gradio pandas youtube-comment-downloader langchain-openai langchain-community faiss-cpu
````

You'll also need an **OpenRouter API key** (free tier available):
[https://openrouter.ai](https://openrouter.ai)

---

## ▶️ Usage

Run the app:

```bash
python app.py
```

Then in the UI:

1. Enter your **OpenRouter API Key**
2. Enter a **YouTube video URL**
3. Select number of comments to load
4. Click **"Load Comments"**
5. Ask questions in the chat box 🎤

---

## 💬 Example Questions

```
What are people saying about the editing quality?
Which commenters are discussing the main topic?
Is there any negative criticism mentioned?
Did anyone reference another YouTuber?
```

The model will answer using **only retrieved comment evidence**.

---

## 📦 Project Structure

```
.
├── app.py               # main application script (Gradio UI)
├── requirements.txt     # required libs to get started
├── README.md            # documentation
```
