"""
Ingest sample documents into the RAG system.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List # Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from llama_index.core import Document

from src.config.config import get_settings
from src.rag.pipeline import RAGPipeline
# from src.rag.utils import DocumentProcessor


async def load_documents_from_json(file_path: str) -> List[Document]:
    """Load documents from JSON file."""
    try:
        logger.info(f"Loading documents from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # Handle different JSON structures
                if 'content' in item:
                    # Document format
                    doc = Document(
                        text=item['content'],
                        metadata={
                            "source": item.get("source", f"document_{i}"),
                            "type": item.get("type", "unknown"),
                            "category": item.get("category", "general"),
                            **item.get("metadata", {})
                        },
                        doc_id=f"doc_{i}"
                    )
                elif 'question' in item and 'answer' in item:
                    # Q&A format
                    text = f"Question: {item['question']}\n\nAnswer: {item['answer']}"
                    doc = Document(
                        text=text,
                        metadata={
                            "source": item.get("source", f"qa_{i}"),
                            "type": "qa",
                            "category": item.get("category", "general"),
                            "question": item['question'],
                            "answer": item['answer'],
                            "difficulty": item.get("difficulty", "unknown")
                        },
                        doc_id=f"qa_{i}"
                    )
                else:
                    # Generic format
                    doc = Document(
                        text=str(item),
                        metadata={
                            "source": f"generic_{i}",
                            "type": "generic"
                        },
                        doc_id=f"generic_{i}"
                    )
                
                documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to load documents from {file_path}: {e}")
        raise


async def ingest_data():
    """Ingest sample data into the RAG system."""
    try:
        logger.info("Starting data ingestion process...")
        
        # Load settings
        settings = get_settings()
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(settings)
        await rag_pipeline.initialize()
        
        # Check current document count
        stats = await rag_pipeline.get_stats()
        initial_count = stats.get("document_count", 0)
        logger.info(f"Initial document count: {initial_count}")
        
        # List of data files to ingest
        data_files = [
            "data/sample_qa.json",
            "data/educational_articles.json",
            "data/combined_documents.json"
        ]
        
        total_ingested = 0
        
        for data_file in data_files:
            file_path = Path(data_file)
            
            if not file_path.exists():
                logger.warning(f"Data file not found: {file_path}")
                continue
            
            try:
                # Load documents from file
                documents = await load_documents_from_json(str(file_path))
                
                if documents:
                    # Ingest documents
                    logger.info(f"Ingesting {len(documents)} documents from {file_path.name}")
                    await rag_pipeline.ingest_documents(documents)
                    total_ingested += len(documents)
                    logger.info(f"Successfully ingested {len(documents)} documents")
                else:
                    logger.warning(f"No documents found in {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                continue
        
        # Get final stats
        final_stats = await rag_pipeline.get_stats()
        final_count = final_stats.get("document_count", 0)
        
        logger.info(f"Data ingestion completed!")
        logger.info(f"Documents ingested in this session: {total_ingested}")
        logger.info(f"Total documents in vector store: {final_count}")
        logger.info(f"Pipeline stats: {final_stats}")
        
        logger.info("✅ Data ingestion process completed successfully!")
        
        return {
            "status": "success",
            "documents_ingested": total_ingested,
            "total_documents": final_count,
            "pipeline_stats": final_stats
        }
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise


def main():
    """Main function to run data ingestion."""
    # Check if data files exist
    required_files = ["data/sample_qa.json", "data/combined_documents.json","data/educational_articles.json"]
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing required data files: {missing_files}")
        print("Please run 'python scripts/download_data.py' first to create sample data.")
        sys.exit(1)
    
    # Run ingestion
    result = asyncio.run(ingest_data())
    print(f"Data ingestion result: {result}")


if __name__ == "__main__":
    main()
