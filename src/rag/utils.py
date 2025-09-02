"""
Utility functions for the RAG pipeline.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from loguru import logger
from llama_index.core import Document
from sentence_transformers import SentenceTransformer


class DocumentProcessor:
    """Utility class for processing documents."""
    
    @staticmethod
    def load_json_documents(file_path: str) -> List[Document]:
        """Load documents from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            for i, item in enumerate(data):
                # Create document from JSON item
                if isinstance(item, dict):
                    # Handle different JSON structures
                    if 'question' in item and 'answer' in item:
                        # Q&A format
                        text = f"Question: {item['question']}\nAnswer: {item['answer']}"
                        metadata = {
                            "source": item.get("source", f"document_{i}"),
                            "type": "qa_pair",
                            "question": item['question'],
                            "answer": item['answer']
                        }
                    elif 'content' in item:
                        # Content format
                        text = item['content']
                        metadata = {
                            "source": item.get("source", f"document_{i}"),
                            "type": "content",
                            "title": item.get("title", "")
                        }
                    else:
                        # Generic format
                        text = str(item)
                        metadata = {
                            "source": f"document_{i}",
                            "type": "generic"
                        }
                    
                    doc = Document(
                        text=text,
                        metadata=metadata,
                        doc_id=f"doc_{i}"
                    )
                    documents.append(doc)
                
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load documents from {file_path}: {e}")
            raise
    
    @staticmethod
    def load_text_documents(directory: str) -> List[Document]:
        """Load text documents from a directory."""
        try:
            directory = Path(directory)
            documents = []
            
            for file_path in directory.glob("**/*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = Document(
                    text=content,
                    metadata={
                        "source": str(file_path),
                        "type": "text_file",
                        "filename": file_path.name
                    },
                    doc_id=f"file_{file_path.stem}"
                )
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} text documents from {directory}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load text documents from {directory}: {e}")
            raise


class EmbeddingUtils:
    """Utility class for embedding operations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with SentenceTransformer model."""
        self.model = SentenceTransformer(model_name)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise


class TextUtils:
    """Utility class for text processing."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters if needed
        # text = re.sub(r'[^\w\s\.\?\!,;:]', '', text)
        
        return text.strip()
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 1000) -> str:
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclamation = truncated.rfind('!')
        
        last_sentence_end = max(last_period, last_question, last_exclamation)
        
        if last_sentence_end > max_length * 0.8:  # If we can keep 80% of text
            return truncated[:last_sentence_end + 1]
        else:
            return truncated + "..."
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text (simple implementation)."""
        # Simple keyword extraction - can be enhanced with NLP libraries
        words = text.lower().split()
        
        # Filter common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            word = word.strip('.,!?;:"()[]{}')
            if len(word) > 2 and word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in keywords[:max_keywords]]


def format_sources(sources: List[str], max_sources: int = 5) -> List[str]:
    """Format and limit the number of sources."""
    if not sources:
        return []
    
    # Remove duplicates while preserving order
    unique_sources = []
    seen = set()
    for source in sources:
        if source not in seen:
            unique_sources.append(source)
            seen.add(source)
    
    # Limit number of sources
    return unique_sources[:max_sources]


def calculate_similarity_score(text1: str, text2: str) -> float:
    """Calculate simple similarity score between two texts."""
    # Simple Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)
