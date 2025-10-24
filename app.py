import os
import pandas as pd
import gradio as gr
from youtube_comment_downloader import YoutubeCommentDownloader
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = None
qa_chain = None

def scrape_youtube_comments(video_url, max_comments=500):
    """Scrape YouTube comments using youtube-comment-downloader"""
    downloader = YoutubeCommentDownloader()
    comments_gen = downloader.get_comments_from_url(video_url, sort_by=1)

    comments = []
    for i, comment in enumerate(comments_gen):
        comments.append({
            "author": comment.get("author"),
            "comment": comment.get("text"),
            "time": comment.get("time")
        })
        if i + 1 >= max_comments:
            break

    df = pd.DataFrame(comments)
    return df

def load_comments_to_vectorstore(video_url, max_comments, openrouter_api_key):
    """Load YouTube comments into vector store"""
    global vectorstore, qa_chain

    try:
        # Scrape comments
        df = scrape_youtube_comments(video_url, max_comments)

        if df.empty:
            return "❌ No comments found!", None

        # Convert to LangChain documents
        documents = []
        for _, row in df.iterrows():
            content = f"Author: {row['author']}\nTime: {row['time']}\nComment: {row['comment']}"
            documents.append(Document(page_content=content, metadata={
                "author": row['author'],
                "time": row['time']
            }))

        # Split documents - keep comments intact
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\nAuthor:", "\n\n", "\n", " "]
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store
        vectorstore = FAISS.from_documents(splits, embeddings)

        # Initialize LLM with OpenRouter
        llm = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7
        )

        # Create custom prompt
        prompt_template = """Use the following YouTube comments to answer the question.
Pay attention to author names, comment content, and timestamps.
When asked about specific users/authors, check if their name appears in the comments.
Answer directly based on what you find in the context.

Context: {context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        return f"✅ Loaded {len(df)} comments successfully!", df

    except Exception as e:
        return f"❌ Error: {str(e)}", None

def chat_with_comments(question, chat_history):
    """Chat with the loaded comments"""
    global qa_chain

    if qa_chain is None:
        return "⚠️ Please load YouTube comments first!"

    try:
        result = qa_chain({"query": question})
        answer = result["result"]
        return answer

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="YouTube Comments RAG Chatbot") as demo:
    gr.Markdown("# 🎥 YouTube Comments RAG Chatbot")
    gr.Markdown("Load YouTube comments and chat with them using LangChain + OpenRouter")

    with gr.Row():
        with gr.Column():
            api_key_input = gr.Textbox(
                label="OpenRouter API Key",
                type="password",
                placeholder="sk-or-v1-..."
            )
            url_input = gr.Textbox(
                label="YouTube Video URL",
                placeholder="https://youtu.be/..."
            )
            max_comments_input = gr.Slider(
                minimum=10,
                maximum=100,
                value=20,
                step=10,
                label="Max Comments"
            )
            load_btn = gr.Button("📥 Load Comments", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)

        with gr.Column():
            comments_df = gr.Dataframe(
                label="Loaded Comments Preview",
                headers=["author", "comment", "time"],
                interactive=False
            )

    gr.Markdown("---")
    gr.Markdown("## 💬 Chat with Comments")

    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(
        label="Ask a question about the comments",
        placeholder="What are people saying about...?"
    )
    clear = gr.Button("Clear Chat")

    # Event handlers
    def respond(message, chat_history):
        bot_message = chat_with_comments(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history

    load_btn.click(
        fn=load_comments_to_vectorstore,
        inputs=[url_input, max_comments_input, api_key_input],
        outputs=[status_output, comments_df]
    )

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()

