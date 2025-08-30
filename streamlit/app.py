"""
Streamlit chat interface for the RAG microservice.
"""

import asyncio
import json
import os
import time
from typing import List, Dict, Any, Optional

import streamlit as st
import requests
from loguru import logger

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_api_base_url() -> str:
    """Get API base URL from secrets or environment variables."""
    try:
        # Try to get from Streamlit secrets first
        return st.secrets.get("API_URL", "http://localhost:8000")
    except (FileNotFoundError, AttributeError):
        # Fallback to environment variable or default
        return os.getenv("API_URL", "http://localhost:8000")

# Constants
def get_endpoints():
    """Get API endpoints."""
    api_base_url = get_api_base_url()
    return {
        "health": f"{api_base_url}/health",
        "query": f"{api_base_url}/query"
    }


def check_api_health() -> Dict[str, Any]:
    """Check if the RAG API is healthy."""
    try:
        endpoints = get_endpoints()
        response = requests.get(endpoints["health"], timeout=5)
        if response.status_code == 200:
            return {
                "status": "healthy",
                "data": response.json()
            }
        else:
            return {
                "status": "unhealthy",
                "error": f"HTTP {response.status_code}"
            }
    except requests.exceptions.RequestException as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def query_rag_api(query: str, max_sources: int = 3) -> Dict[str, Any]:
    """Query the RAG API."""
    try:
        endpoints = get_endpoints()
        payload = {
            "query": query,
            "max_sources": max_sources,
            "include_metadata": True
        }
        
        response = requests.post(
            endpoints["query"],
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json()
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "api_health" not in st.session_state:
        st.session_state.api_health = None
    
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0


def display_health_status():
    """Display API health status in sidebar."""
    st.sidebar.header("🏥 System Health")
    
    health = check_api_health()
    st.session_state.api_health = health
    
    if health["status"] == "healthy":
        st.sidebar.success("✅ API is healthy")
        
        # Display health details if available
        if "data" in health and "details" in health["data"]:
            details = health["data"]["details"]
            st.sidebar.json(details)
    else:
        st.sidebar.error(f"❌ API is unhealthy: {health.get('error', 'Unknown error')}")


def display_chat_history():
    """Display chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                sources = message["sources"]
                if sources:
                    st.write("**Sources:**")
                    for i, source in enumerate(sources, 1):
                        st.write(f"{i}. {source}")
            
            # Display metadata if available
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                if metadata:
                    with st.expander("View Details"):
                        st.json(metadata)


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("🤖 RAG Chatbot")
    st.write("Ask questions about AI, Machine Learning, and RAG systems!")
    
    # Sidebar
    with st.sidebar:
        display_health_status()
        
        st.header("⚙️ Settings")
        max_sources = st.slider(
            "Max Sources",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum number of source documents to retrieve"
        )
        
        st.header("📊 Statistics")
        st.write(f"**Queries asked:** {st.session_state.query_count}")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.rerun()
    
    # Main chat interface
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about AI/ML..."):
        # Check API health before processing
        if st.session_state.api_health and st.session_state.api_health["status"] != "healthy":
            st.error("⚠️ API is not healthy. Please check the connection.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response from RAG API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_rag_api(prompt, max_sources)
                
                if result["success"]:
                    data = result["data"]
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    confidence_score = data.get("confidence_score", 0.0)
                    metadata = data.get("metadata", {})
                    
                    # Display answer
                    st.write(answer)
                    
                    # Display confidence score
                    if confidence_score > 0:
                        st.write(f"**Confidence:** {confidence_score:.2f}")
                    
                    # Display sources
                    if sources:
                        st.write("**Sources:**")
                        for i, source in enumerate(sources, 1):
                            st.write(f"{i}. {source}")
                    
                    # Display metadata in expandable section
                    if metadata:
                        with st.expander("View Details"):
                            st.json(metadata)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "confidence_score": confidence_score,
                        "metadata": metadata
                    })
                    
                    # Update query count
                    st.session_state.query_count += 1
                    
                else:
                    error_msg = f"❌ Error: {result['error']}"
                    st.error(error_msg)
                    
                    # Add error message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Sample questions
    st.subheader("💡 Sample Questions")
    sample_questions = [
        "What is Retrieval-Augmented Generation (RAG)?",
        "How does ChromaDB work as a vector database?",
        "What are the benefits of using LangChain for RAG pipelines?",
        "Explain the role of embeddings in RAG systems",
        "How do you evaluate RAG system performance?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        col = cols[i % 2]
        if col.button(question, key=f"sample_{i}"):
            # Simulate clicking the question
            st.session_state.sample_question = question
            st.rerun()
    
    # Handle sample question selection
    if hasattr(st.session_state, 'sample_question'):
        prompt = st.session_state.sample_question
        delattr(st.session_state, 'sample_question')
        
        # Process the sample question (same logic as chat input)
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        result = query_rag_api(prompt, max_sources)
        
        if result["success"]:
            data = result["data"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": data.get("sources", []),
                "confidence_score": data.get("confidence_score", 0.0),
                "metadata": data.get("metadata", {})
            })
            st.session_state.query_count += 1
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {result['error']}"
            })
        
        st.rerun()


if __name__ == "__main__":
    main()
