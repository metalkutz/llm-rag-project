"""
RAG Pipeline implementation using LlamaIndex and LangChain.
"""

# import asyncio
import uuid
from typing import Dict, List, Any #Optional
# from pathlib import Path

from loguru import logger
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings as LlamaIndexSettings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from src.rag.vector_store import ChromaVectorStore
from src.config.config import Settings


class RAGPipeline:
    """
    Production-ready RAG pipeline using LlamaIndex, LangChain, and ChromaDB.
    """
    
    def __init__(self, settings: Settings):
        """Initialize the RAG pipeline with configuration."""
        self.settings = settings
        self.vector_store = None
        self.index = None
        self.query_engine = None
        self.retriever = None
        self.embed_model = None
        self.llm = None
        self.text_splitter = None
        self.node_parser = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize all components of the RAG pipeline."""
        try:
            logger.info("Initializing RAG pipeline components...")
            
            # Initialize embedding model
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.settings.embedding.model_name,
                device=self.settings.embedding.device
            )
            
            # Initialize LLM with better generation controls
            self.llm = HuggingFaceLLM(
                model_name=self.settings.llm.model_name,
                tokenizer_name=self.settings.llm.model_name,
                device_map="auto",
                max_new_tokens=self.settings.llm.max_new_tokens,
                model_kwargs={
                    "torch_dtype": "auto",  # Use automatic mixed precision
                    "trust_remote_code": self.settings.llm.trust_remote_code,  # Allow custom model code
                },
                tokenizer_kwargs={
                    "padding_side": "left",  # Pad on the left for causal LM
                    "truncation": True,  # Truncate input sequences
                    "max_length": 900,  # Leave room for generation tokens
                    "return_overflowing_tokens": False,  # Do not return overflowing tokens
                },
                generate_kwargs={
                    "do_sample": True,
                    "temperature": self.settings.llm.temperature,
                    "top_p": self.settings.llm.top_p,
                    "pad_token_id": 50256,
                    "repetition_penalty": 1.2,  # Prevent repetition
                    "early_stopping": True,
                }
            )
            
            # Set global LlamaIndex settings
            LlamaIndexSettings.embed_model = self.embed_model
            LlamaIndexSettings.llm = self.llm
            LlamaIndexSettings.chunk_size = self.settings.document_processing.chunk_size
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.document_processing.chunk_size,
                chunk_overlap=self.settings.document_processing.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )

            # Initialize node parser for LlamaIndex
            self.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=self.settings.document_processing.chunk_size,
                chunk_overlap=self.settings.document_processing.chunk_overlap
            )

            # Initialize vector store
            self.vector_store = ChromaVectorStore(self.settings)
            await self.vector_store.initialize()
            
            # Load or create index
            await self._setup_index()
            
            self._initialized = True
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    async def _setup_index(self) -> None:
        """Setup the vector store index."""
        try:
            # Check if we have existing data
            collection = await self.vector_store.get_collection()
            doc_count = collection.count()

            logger.info(f"Found {doc_count} documents in vector store")
            
            # Create storage context with vector store
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store.get_vector_store()
            )
            
            if doc_count > 0:
                logger.info("Loading existing documents into index")
                # Create index from existing vector store
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store.get_vector_store(),
                    storage_context=storage_context
                )
            else:
                logger.info("No existing documents found, creating empty index")
                # Create empty index with storage context
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context
                )
            
            # Create query engine with proper prompt template
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=min(3, self.settings.retrieval.top_k)  # Limit to 3 sources max
            )
            
            # Create a conversation-style prompt template for DialoGPT
            qa_prompt_template = PromptTemplate(
                """User: I have some information about {query_str}. Here's what I know: {context_str}

Based on this, can you explain {query_str}?
Assistant:"""
            )
            
            # Create response synthesizer with custom prompt
            response_synthesizer = get_response_synthesizer(
                llm=self.llm,
                text_qa_template=qa_prompt_template,
                response_mode=ResponseMode.COMPACT,
                use_async=True
            )
            
            self.query_engine = RetrieverQueryEngine(
                retriever=self.retriever,
                response_synthesizer=response_synthesizer
            )
            
        except Exception as e:
            logger.error(f"Failed to setup index: {e}")
            raise
    
    async def ingest_documents(self, documents: List[Document]) -> None:
        """Ingest documents into the vector store."""
        if not self._initialized:
            raise RuntimeError("RAG pipeline not initialized")
        
        try:
            logger.info(f"Ingesting {len(documents)} documents...")

            # Use LlamaIndex's built-in document processing
            logger.info("Processing documents with LlamaIndex node parser...")
            
            # Parse documents into nodes using LlamaIndex's parser
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            logger.info(f"Generated {len(nodes)} nodes from {len(documents)} documents")
            
            # Add chunk metadata for tracking
            for i, node in enumerate(nodes):
                if not hasattr(node, 'metadata') or node.metadata is None:
                    node.metadata = {}
                
                # Add chunk tracking metadata
                node.metadata.update({
                    "chunk_id": f"{node.node_id}_{i}",
                    "processing_timestamp": str(uuid.uuid4()),
                    "chunk_method": "llamaindex_parser"
                })
            
            # Debug: Check vector store before insertion
            collection_before = await self.vector_store.get_collection()
            count_before = collection_before.count()
            logger.info(f"Vector store count before insertion: {count_before}")
            
            # Insert nodes into the index
            logger.info(f"Inserting {len(nodes)} nodes into index...")
            self.index.insert_nodes(nodes)
            logger.info("Node insertion completed")
            
            # Debug: Check vector store after insertion
            collection_after = await self.vector_store.get_collection()
            count_after = collection_after.count()
            logger.info(f"Vector store count after insertion: {count_after}")
            
            if count_after == count_before:
                logger.warning("⚠️  Vector store count unchanged after insertion - nodes may not be persisting!")
                logger.warning("This could indicate an embedding or vector store connection issue.")
            
            # Rebuild query engine to ensure it sees the new data
            await self._rebuild_query_engine()

            # Final verification
            collection_final = await self.vector_store.get_collection()
            new_count = collection_final.count()
            logger.info(f"Successfully ingested {len(nodes)} document nodes. Total in vector store: {new_count}")

        except Exception as e:
            logger.error(f"Failed to ingest documents: {e}")
            raise
    
    async def ingest_documents_manual_chunking(self, documents: List[Document]) -> None:
        """
        Alternative ingestion method using manual chunking.
        Provides more control over the chunking process.
        """
        if not self._initialized:
            raise RuntimeError("RAG pipeline not initialized")
        
        try:
            logger.info(f"Ingesting {len(documents)} documents with manual chunking...")
            
            # Process documents through text splitter first
            processed_docs = []
            for doc in documents:
                chunks = self.text_splitter.split_text(doc.text)
                for i, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        text=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_id": f"{doc.doc_id}_{i}",
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "original_doc_id": doc.doc_id
                        },
                        doc_id=f"{doc.doc_id}_chunk_{i}"
                    )
                    processed_docs.append(chunk_doc)
            
            # Convert processed documents to nodes
            nodes = self.node_parser.get_nodes_from_documents(processed_docs)
            
            # Insert nodes into the index
            self.index.insert_nodes(nodes)
            
            logger.info(f"Successfully ingested {len(nodes)} document chunks using manual method")
            
        except Exception as e:
            logger.error(f"Failed to ingest documents with manual chunking: {e}")
            raise

    async def _rebuild_query_engine(self) -> None:
        """Rebuild the query engine after ingestion."""
        try:
            logger.info("Rebuilding query engine with updated index...")
            
            # Create new retriever with updated index
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.settings.retrieval.top_k
            )
            
            # Create new query engine
            self.query_engine = RetrieverQueryEngine(retriever=retriever)
            
            logger.info("Query engine rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Failed to rebuild query engine: {e}")
            raise   

    async def query(self, query_text: str) -> Dict[str, Any]:
        """
        Query the RAG pipeline.
        
        Args:
            query_text: The question to ask
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self._initialized:
            raise RuntimeError("RAG pipeline not initialized")
        
        try:
            logger.info(f"Processing query: {query_text}")
            query_id = str(uuid.uuid4())
            
            # Query the engine
            response = self.query_engine.query(query_text)
            
            # Extract sources
            sources = []
            source_metadata = []
            
            for node in response.source_nodes:
                source_info = node.metadata.get("source", "Unknown")
                sources.append(source_info)
                source_metadata.append({
                    "source": source_info,
                    "score": node.score if hasattr(node, 'score') else 0.0,
                    "chunk_id": node.metadata.get("chunk_id", ""),
                    "node_id": node.node_id,
                    "text_snippet": node.text[:200] + "..." if len(node.text) > 200 else node.text
                })
            
            # Calculate confidence score (simple heuristic)
            confidence_score = min(1.0, len(sources) * 0.2) if sources else 0.0
            
            # Get response text correctly
            response_text = str(response).strip()
            
            # Check for empty responses
            if not response_text or response_text.lower() in ['empty response', 'none', '']:
                response_text = "I found relevant information but was unable to generate a complete response."
            
            result = {
                "answer": response_text,
                "sources": sources,
                "confidence_score": confidence_score,
                "query_id": query_id,
                "metadata": {
                    "source_details": source_metadata,
                    "response_length": len(response_text),
                    "num_sources": len(sources)
                }
            }
            
            logger.info(f"Query processed successfully. Query ID: {query_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the RAG pipeline."""
        health_status = {
            "initialized": self._initialized,
            "vector_store": False,
            "index": False,
            "query_engine": False
        }
        
        try:
            if self.vector_store:
                collection = await self.vector_store.get_collection()
                health_status["vector_store"] = True
                health_status["document_count"] = collection.count()
            
            if self.index:
                health_status["index"] = True
            
            if self.query_engine:
                health_status["query_engine"] = True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
        
        return health_status
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if not self._initialized:
            return {"error": "Pipeline not initialized"}
        
        try:
            collection = await self.vector_store.get_collection()
            return {
                "document_count": collection.count(),
                "embedding_model": self.settings.embedding.model_name,
                "llm_model": self.settings.llm.model_name,
                "chunk_size": self.settings.document_processing.chunk_size,
                "retrieval_top_k": self.settings.retrieval.top_k,
                "node_parser_available": self.node_parser is not None
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
