"""
Vector store implementation using ChromaDB.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.models.Collection import Collection
from chromadb.errors import InvalidCollectionException
from llama_index.vector_stores.chroma import ChromaVectorStore as LlamaIndexChromaVectorStore
from llama_index.core.schema import Document
from loguru import logger

from src.config.config import Settings


class ChromaVectorStore:
    """
    ChromaDB vector store wrapper for the RAG pipeline.
    """
    
    def __init__(self, settings: Settings):
        """Initialize ChromaDB vector store."""
        self.settings = settings
        self.client = None
        self.collection = None
        self.llama_index_vector_store = None
        
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            logger.info("Initializing ChromaDB vector store...")
            
            # Ensure persist directory exists
            persist_dir = Path(self.settings.vector_store.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize client
            chroma_settings = ChromaSettings(
                persist_directory=str(persist_dir),
                anonymized_telemetry=False
            )
            
            self.client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=chroma_settings
            )
            
            # Get or create collection
            collection_name = self.settings.vector_store.collection_name
            try:
                # Try to get existing collection
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=None  # We'll use LlamaIndex embeddings
                )
                logger.info(f"Loaded existing collection: {collection_name}")
            except InvalidCollectionException:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=None
                )
                logger.info(f"Created new collection: {collection_name}")
                logger.info(f"Created new collection: {collection_name}")
            
            # Create LlamaIndex vector store wrapper
            self.llama_index_vector_store = LlamaIndexChromaVectorStore(
                chroma_collection=self.collection
            )
            
            logger.info("ChromaDB vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def get_collection(self) -> Collection:
        """Get the ChromaDB collection."""
        if self.collection is None:
            raise RuntimeError("Vector store not initialized")
        return self.collection
    
    def get_vector_store(self) -> LlamaIndexChromaVectorStore:
        """Get the LlamaIndex vector store wrapper."""
        if self.llama_index_vector_store is None:
            raise RuntimeError("Vector store not initialized")
        return self.llama_index_vector_store
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            collection = await self.get_collection()
            count = collection.count()
            
            return {
                "collection_name": collection.name,
                "document_count": count,
                "persist_directory": self.settings.vector_store.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    async def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            if self.client is not None and self.collection is not None:
                self.client.delete_collection(self.collection.name)
                logger.info(f"Deleted collection: {self.collection.name}")
                self.collection = None
                self.llama_index_vector_store = None
            else:
                logger.warning("Cannot delete collection: client or collection is None")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

# import asyncio

# if __name__ == "__main__":
#     settings = Settings()
#     vector_store = ChromaVectorStore(settings)

#     async def main():
#         try:
#             # Initialize the vector store first
#             await vector_store.initialize()
#             await vector_store.delete_collection()
#             logger.info("Collection deleted successfully")
#         except Exception as e:
#             logger.error(f"Failed to delete collection: {e}")
#             raise

#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         logger.info("Operation cancelled by user")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         exit(1)