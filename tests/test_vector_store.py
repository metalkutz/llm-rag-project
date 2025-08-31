"""
Tests for vector store functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from src.config.config import Settings
from src.rag.vector_store import ChromaVectorStore


@pytest.fixture
def mock_settings():
    """Create mock settings for testing with C++ compatibility."""
    settings = Settings()
    
    # Use absolute paths for C++ integration reliability
    test_dir = Path(__file__).parent / "test_data"
    
    settings.vector_store.persist_directory = str(test_dir / "chromadb")
    settings.vector_store.collection_name = "test_collection"
    
    # Ensure paths are C++ compatible (no spaces, special chars)
    assert not any(char in settings.vector_store.persist_directory for char in [' ', '"', "'"])
    
    return settings


@pytest.mark.asyncio
async def test_chromadb_initialization(mock_settings):
    """Test ChromaDB vector store initialization."""
    with patch('src.rag.vector_store.chromadb.PersistentClient') as mock_client, \
         patch('src.rag.vector_store.Path') as mock_path:
        
        # Setup mocks
        mock_path.return_value.mkdir = MagicMock()
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_client_instance.get_collection.return_value = mock_collection
        
        # Test initialization
        vector_store = ChromaVectorStore(mock_settings)
        await vector_store.initialize()
        
        # Verify initialization
        assert vector_store.client is not None
        assert vector_store.collection is not None
        assert vector_store.llama_index_vector_store is not None
        
        # Verify client was called correctly
        mock_client.assert_called_once()
        mock_client_instance.get_collection.assert_called_once_with(
            name="test_collection",
            embedding_function=None
        )


@pytest.mark.asyncio
async def test_chromadb_create_new_collection(mock_settings):
    """Test creating a new collection when it doesn't exist."""
    with patch('src.rag.vector_store.chromadb.PersistentClient') as mock_client, \
         patch('src.rag.vector_store.Path') as mock_path:
        
        # Setup mocks
        mock_path.return_value.mkdir = MagicMock()
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Simulate collection not existing - use ChromaDB's actual exception
        class InvalidCollectionException(Exception):
            pass
        mock_client_instance.get_collection.side_effect = InvalidCollectionException("Collection not found")
        
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_client_instance.create_collection.return_value = mock_collection
        
        # Test initialization
        vector_store = ChromaVectorStore(mock_settings)
        await vector_store.initialize()
        
        # Verify new collection was created
        mock_client_instance.create_collection.assert_called_once_with(
            name="test_collection",
            embedding_function=None
        )
        
        # Verify the vector store state is properly set
        assert vector_store.client is not None
        assert vector_store.collection is not None
        assert vector_store.collection.name == "test_collection"


@pytest.mark.asyncio
async def test_get_collection_error_handling(mock_settings):
    """Test error handling when vector store is not initialized."""
    vector_store = ChromaVectorStore(mock_settings)
    
    # Test get_collection before initialization
    with pytest.raises(RuntimeError, match="Vector store not initialized"):
        await vector_store.get_collection()


@pytest.mark.asyncio
async def test_get_vector_store_error_handling(mock_settings):
    """Test error handling for LlamaIndex vector store."""
    vector_store = ChromaVectorStore(mock_settings)
    
    # Test get_vector_store before initialization
    with pytest.raises(RuntimeError, match="Vector store not initialized"):
        vector_store.get_vector_store()


@pytest.mark.asyncio
async def test_delete_collection(mock_settings):
    """Test deleting the collection."""
    with patch('src.rag.vector_store.chromadb.PersistentClient'):
        vector_store = ChromaVectorStore(mock_settings)
        
        # Mock client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        
        vector_store.client = mock_client
        vector_store.collection = mock_collection
        
        # Test deletion
        await vector_store.delete_collection()
        
        # Verify deletion was called
        mock_client.delete_collection.assert_called_once_with("test_collection")
        assert vector_store.collection is None
        assert vector_store.llama_index_vector_store is None


@pytest.mark.asyncio
async def test_get_stats(mock_settings):
    """Test getting vector store statistics."""
    with patch('src.rag.vector_store.chromadb.PersistentClient'):
        vector_store = ChromaVectorStore(mock_settings)
        
        # Mock collection
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.count.return_value = 42
        vector_store.collection = mock_collection
        
        # Test getting stats
        stats = await vector_store.get_stats()
        
        # Verify stats
        assert stats["collection_name"] == "test_collection"
        assert stats["document_count"] == 42
        assert stats["persist_directory"] == mock_settings.vector_store.persist_directory


@pytest.mark.asyncio
async def test_get_stats_error_handling(mock_settings):
    """Test error handling in get_stats."""
    with patch('src.rag.vector_store.chromadb.PersistentClient'):
        vector_store = ChromaVectorStore(mock_settings)
        
        # Mock collection that raises error
        mock_collection = MagicMock()
        mock_collection.count.side_effect = Exception("Database error")
        vector_store.collection = mock_collection
        
        # Test getting stats with error
        stats = await vector_store.get_stats()
        
        # Should return error in stats
        assert "error" in stats
        assert "Database error" in stats["error"]


@pytest.mark.asyncio
async def test_chromadb_persistence_directory_creation(mock_settings):
    """Test that persistence directory is created if it doesn't exist."""
    with patch('src.rag.vector_store.chromadb.PersistentClient') as mock_client, \
         patch('src.rag.vector_store.Path') as mock_path:
        
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_collection = MagicMock()
        mock_client_instance.get_collection.return_value = mock_collection
        
        # Test initialization
        vector_store = ChromaVectorStore(mock_settings)
        await vector_store.initialize()
        
        # Verify directory creation was attempted
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)


@pytest.mark.asyncio
async def test_initialization_error_handling(mock_settings):
    """Test error handling during initialization."""
    with patch('src.rag.vector_store.chromadb.PersistentClient') as mock_client, \
         patch('src.rag.vector_store.Path') as mock_path:
        
        # Setup path mock to succeed
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.mkdir = MagicMock()
        
        # Setup client mock to raise error
        mock_client.side_effect = Exception("ChromaDB connection failed")
        
        vector_store = ChromaVectorStore(mock_settings)
        
        # Test that initialization raises the error
        with pytest.raises(Exception, match="ChromaDB connection failed"):
            await vector_store.initialize()
        
        # Verify directory creation was attempted before the error
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # Verify vector store remains uninitialized
        assert vector_store.client is None
        assert vector_store.collection is None
        assert vector_store.llama_index_vector_store is None

@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up test data for C++/Python shared environment."""
    import shutil
    import time
    
    test_data_path = Path("test_data")
    
    # Cleanup before test
    if test_data_path.exists():
        # Add small delay for file system consistency in C++ integration
        time.sleep(0.1)
        shutil.rmtree(test_data_path, ignore_errors=True)
    
    yield
    
    # Cleanup after test
    if test_data_path.exists():
        time.sleep(0.1)
        shutil.rmtree(test_data_path, ignore_errors=True)