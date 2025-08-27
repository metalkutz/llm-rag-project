"""
Initialize ChromaDB for the RAG system.
"""

import asyncio
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config.config import get_settings
from src.rag.vector_store import ChromaVectorStore


async def init_chromadb():
    """Initialize ChromaDB with proper configuration."""
    try:
        logger.info("Initializing ChromaDB...")
        
        # Load settings
        settings = get_settings()
        
        # Create data directory
        data_dir = Path(settings.vector_store.persist_directory)
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created data directory: {data_dir}")
        
        # Initialize vector store
        vector_store = ChromaVectorStore(settings)
        await vector_store.initialize()
        
        # Get collection info
        collection = await vector_store.get_collection()
        logger.info(f"Collection '{collection.name}' initialized successfully")
        
        # Get stats
        stats = await vector_store.get_stats()
        logger.info(f"Vector store stats: {stats}")
        
        logger.info("✅ ChromaDB initialization completed successfully!")
        
        return {
            "status": "success",
            "collection_name": collection.name,
            "persist_directory": settings.vector_store.persist_directory,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise


def main():
    """Main function to run ChromaDB initialization."""
    result = asyncio.run(init_chromadb())
    print(f"ChromaDB initialization result: {result}")


if __name__ == "__main__":
    main()
