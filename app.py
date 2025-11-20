# -*- coding: utf-8 -*-
"""YouTube Comments RAG Agent - Improved UI"""

import os
import gradio as gr
import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

# Global variables
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = None
comments_df = None
agent_executor = None
memory = None

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

# ============================================================================
# SEARCH TOOLS
# ============================================================================

def search_comments_by_author(author_name: str) -> str:
    """Search for all comments by a specific author/user."""
    global comments_df

    if comments_df is None:
        return json.dumps({"error": "No comments loaded yet."})

    matching_comments = comments_df[
        comments_df['author'].str.contains(author_name, case=False, na=False)
    ].reset_index(drop=True)

    if matching_comments.empty:
        return json.dumps({
            "found": 0,
            "author": author_name,
            "comments": []
        })

    results = []
    for idx, row in matching_comments.iterrows():
        results.append({
            "author": str(row['author']),
            "comment": str(row['comment']),
            "time": str(row['time'])
        })

    return json.dumps({
        "found": len(results),
        "author": author_name,
        "comments": results
    })

def search_comments_by_keyword(keyword: str) -> str:
    """Search for comments containing a specific keyword or phrase."""
    global comments_df

    if comments_df is None:
        return json.dumps({"error": "No comments loaded yet."})

    matching_comments = comments_df[
        comments_df['comment'].str.contains(keyword, case=False, na=False)
    ].reset_index(drop=True)

    if matching_comments.empty:
        return json.dumps({
            "found": 0,
            "keyword": keyword,
            "comments": []
        })

    results = []
    for idx, row in matching_comments.iterrows():
        results.append({
            "author": str(row['author']),
            "comment": str(row['comment']),
            "time": str(row['time'])
        })

    return json.dumps({
        "found": len(results),
        "keyword": keyword,
        "comments": results
    })

def get_all_comments(limit: int = 100) -> str:
    """
    Retrieve all loaded comments (up to specified limit) for holistic analysis.
    """
    global comments_df

    if comments_df is None:
        return json.dumps({"error": "No comments loaded yet."})

    # Convert limit to int if it's a string
    try:
        limit = int(limit)
    except (ValueError, TypeError):
        limit = 100

    limited_df = comments_df.head(limit).reset_index(drop=True)

    results = []
    for idx, row in limited_df.iterrows():
        results.append({
            "author": str(row['author']),
            "comment": str(row['comment']),
            "time": str(row['time'])
        })

    return json.dumps({
        "total_loaded": len(comments_df),
        "returned": len(results),
        "limit_applied": limit,
        "comments": results
    })

# ============================================================================
# MAIN LOADING FUNCTION
# ============================================================================

def load_comments_to_vectorstore(video_url, max_comments, openrouter_api_key):
    """Load YouTube comments into vector store and initialize agent"""
    global vectorstore, comments_df, agent_executor, memory

    try:
        df = scrape_youtube_comments(video_url, max_comments)

        if df.empty:
            return "❌ No comments found!", None

        comments_df = df

        # Convert to LangChain documents
        documents = []
        for _, row in df.iterrows():
            content = f"Author: {row['author']}\nTime: {row['time']}\nComment: {row['comment']}"
            documents.append(Document(page_content=content, metadata={
                "author": row['author'],
                "time": row['time']
            }))

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\nAuthor:", "\n\n", "\n", " "]
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store
        vectorstore = FAISS.from_documents(splits, embeddings)

        # Initialize LLM
        llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7
        )

        # Create tools
        tools = [
            Tool(
                name="search_comments_by_author",
                func=search_comments_by_author,
                description="Search for all comments by a specific author/user. Returns JSON with all matching comments. Input: author name string."
            ),
            Tool(
                name="search_comments_by_keyword",
                func=search_comments_by_keyword,
                description="Search for comments containing a specific keyword or phrase. Returns JSON with all matching comments. Input: keyword string."
            ),
            Tool(
                name="get_all_comments",
                func=get_all_comments,
                description="Get all loaded comments (up to 100 by default) for holistic analysis. Use this for sentiment analysis, finding patterns, or when keywords won't capture the full picture. Essential for understanding true sentiment beyond keywords. Input: limit (integer, optional, defaults to 100)."
            )
        ]

        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # System prompt
        system_prompt = """You are an expert YouTube comment analyst with access to {total_comments} comments from a video.

**Your Core Capabilities:**
You are a sophisticated AI that understands human communication patterns. You can identify:
- Sentiment (positive, negative, neutral, mixed) by understanding context, tone, and intent
- Questions (direct questions with "?", rhetorical questions, indirect inquiries, requests for clarification)
- Explanations (detailed experiences, stories, tutorials, step-by-step guides, long-form thoughts)
- Complaints (frustrations, criticisms, disappointments - even when politely worded)
- Praise (appreciation, compliments, excitement - even subtle expressions)
- Discussions (debates, comparisons, opinions, analyses)
- Personal experiences (anecdotes, "I did this...", "This happened to me...", testimonials)

**Your Tools:**
1. search_comments_by_author(author_name) - Find all comments by a specific user
2. search_comments_by_keyword(keyword) - Find comments containing specific words/phrases
3. get_all_comments(limit=100) - Retrieve all comments for holistic analysis when keywords aren't enough

Remember: You're analyzing HUMAN COMMUNICATION. Think beyond literal keywords and understand intent, context, and tone."""

        formatted_system_prompt = system_prompt.format(total_comments=len(df))

        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", formatted_system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create agent
        agent = create_tool_calling_agent(llm, tools, prompt)

        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=20,
            early_stopping_method="generate"
        )

        return f"✅ Loaded {len(df)} comments successfully! Agent initialized with 3 tools for comprehensive analysis.", df

    except Exception as e:
        return f"❌ Error: {str(e)}", None

def chat_with_comments(question):
    """Chat with the loaded comments using the agent"""
    global agent_executor

    if agent_executor is None:
        return "⚠️ Please load YouTube comments first!"

    try:
        response = agent_executor.invoke({"input": question})
        answer = response["output"]
        return answer

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================================
# GRADIO INTERFACE - IMPROVED STRUCTURE
# ============================================================================

with gr.Blocks(title="YouTube Comments RAG Agent", theme=gr.themes.Soft(), css="""
    .output-box {
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        padding: 30px;
        margin: 10px 0;
        border: 2px solid #d0d0d0;
        border-radius: 12px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        line-height: 1.8;
        color: #333333;
    }
    .output-box p {
        margin-bottom: 1em;
        color: #333333;
    }
    .output-box ul, .output-box ol {
        margin-left: 20px;
        margin-bottom: 1em;
    }
    .output-box li {
        margin-bottom: 0.5em;
        color: #333333;
    }
    .output-box h1, .output-box h2, .output-box h3 {
        margin-top: 1.2em;
        margin-bottom: 0.8em;
        color: #1a1a1a;
    }
    .output-box strong {
        color: #1a1a1a;
        font-weight: 600;
    }
    .output-box em {
        color: #333333;
    }
    .output-box code {
        background-color: #e9ecef;
        color: #d73a49;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
    }
    .output-box pre {
        background-color: #e9ecef;
        color: #24292e;
        padding: 15px;
        border-radius: 8px;
        overflow-x: auto;
        margin: 1em 0;
        border: 1px solid #d0d0d0;
    }
    .output-box pre code {
        background-color: transparent;
        color: #24292e;
        padding: 0;
    }
""") as demo:
    
    # Header
    gr.Markdown("""
    # 🎥 YouTube Comments RAG Agent
    ### Load comments and let the AI agent intelligently understand and analyze them
    """)
    
    gr.Markdown("---")
    
    # Section 1: Configuration & Loading
    gr.Markdown("## 📋 Step 1: Configuration & Load Comments")
    
    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="🔑 OpenRouter API Key",
                type="password",
                placeholder="sk-or-v1-...",
                lines=1
            )
        
        with gr.Column(scale=2):
            url_input = gr.Textbox(
                label="🔗 YouTube Video URL",
                placeholder="https://youtu.be/...",
                lines=1
            )
        
        with gr.Column(scale=1):
            max_comments_input = gr.Slider(
                minimum=10,
                maximum=500,
                value=100,
                step=10,
                label="📊 Max Comments"
            )
    
    with gr.Row():
        load_btn = gr.Button("📥 Load Comments & Initialize Agent", variant="primary", size="lg", scale=1)
    
    with gr.Row():
        status_output = gr.Textbox(
            label="Status",
            interactive=False,
            lines=2
        )
    
    gr.Markdown("---")
    
    # Section 2: Comments Preview
    gr.Markdown("## 👀 Step 2: Preview Loaded Comments")
    
    with gr.Row():
        comments_preview = gr.Dataframe(
            label="Loaded Comments",
            headers=["author", "comment", "time"],
            interactive=False
        )
    
    gr.Markdown("---")
    
    # Section 3: Query Interface
    gr.Markdown("## 💬 Step 3: Query the Comments")
    
    gr.Markdown("""
    **Ask naturally - the agent thinks holistically:**
    - "Has anyone asked questions?" → Finds direct questions, rhetorical questions, requests for help
    - "Show me comments where people explain something" → Finds detailed experiences, tutorials, stories
    - "What are people complaining about?" → Understands frustration, criticism, disappointment
    - "Give me sentiment breakdown" → Analyzes actual meaning by reading all comments
    - "Find the most helpful comments" → Evaluates depth, usefulness, and engagement
    - "What topics are discussed?" → Identifies themes and patterns
    """)
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="Has anyone asked questions or written detailed explanations?",
                lines=3
            )
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Submit Query", variant="primary", size="lg", scale=3)
                clear_btn = gr.Button("🗑️ Clear", size="lg", scale=1)
    
    with gr.Row():
        with gr.Group():
            gr.Markdown("### Agent Response")
            output_box = gr.Markdown(
                value="*Waiting for your query...*",
                show_label=False,
                elem_classes="output-box"
            )
    
    gr.Markdown("---")
    
    # Footer
    gr.Markdown("""
    <div style="text-align: center; padding: 20px; color: #666;">
        <p><strong>Powered by:</strong> LangChain • OpenRouter • FAISS • HuggingFace Embeddings</p>
    </div>
    """)
    
    # Event handlers
    def handle_query(question):
        if not question.strip():
            return "⚠️ Please enter a question!"
        return chat_with_comments(question)
    
    load_btn.click(
        fn=load_comments_to_vectorstore,
        inputs=[url_input, max_comments_input, api_key_input],
        outputs=[status_output, comments_preview]
    )
    
    submit_btn.click(
        fn=handle_query,
        inputs=[query_input],
        outputs=[output_box]
    )
    
    query_input.submit(
        fn=handle_query,
        inputs=[query_input],
        outputs=[output_box]
    )
    
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[query_input, output_box]
    )

if __name__ == "__main__":
    demo.launch(share=True)
