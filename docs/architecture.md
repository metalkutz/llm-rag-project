# RAG System Architecture

## Overview

The RAG (Retrieval-Augmented Generation) microservice is designed as a modular, scalable system that combines document retrieval with generative AI to provide accurate, contextual responses to user queries.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI API   │    │  RAG Pipeline   │
│                 │    │                 │    │                 │
│  - Chat UI      │◄──►│  - REST API     │◄──►│  - LlamaIndex   │
│  - Source View  │    │  - Validation   │    │  - LangChain    │
│  - Health Check │    │  - Error Handle │    │  - Embeddings   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   Vector Store  │
                                               │                 │
                                               │   - ChromaDB    │
                                               │   - Persistence │
                                               │   - Similarity  │
                                               └─────────────────┘
```

## Components

### 1. API Layer (`src/api/`)

- **FastAPI Application**: RESTful API with async support
- **Pydantic Models**: Request/response validation and serialization
- **Health Checks**: System health monitoring
- **Error Handling**: Comprehensive error management
- **CORS Support**: Cross-origin resource sharing

#### Endpoints

- `GET /health` - Health check endpoint
- `POST /query` - Main RAG query endpoint
- `GET /` - API information

### 2. RAG Pipeline (`src/rag/`)

- **Pipeline Orchestration**: Main RAG workflow coordination
- **Document Processing**: Text chunking and preprocessing
- **Vector Store Integration**: ChromaDB management
- **Embedding Generation**: Sentence transformer embeddings
- **Retrieval**: Similarity-based document retrieval
- **Generation**: LLM-based answer generation

#### Key Components

- `RAGPipeline`: Main orchestrator class
- `ChromaVectorStore`: Vector database wrapper
- `DocumentProcessor`: Document ingestion utilities
- `EmbeddingUtils`: Embedding generation utilities

### 3. Configuration (`src/config/`)

- **Settings Management**: Centralized configuration using Pydantic
- **Environment Support**: Environment variable and YAML file support
- **Validation**: Configuration validation and type checking

### 4. Streamlit UI (`streamlit/`)

- **Chat Interface**: Real-time conversational UI
- **Source Attribution**: Display of source documents
- **Health Monitoring**: API health status display
- **Settings**: User configurable parameters

## Data Flow

1. **User Query**: User submits question through Streamlit UI
2. **API Request**: UI sends POST request to `/query` endpoint
3. **Query Processing**: FastAPI validates request and passes to RAG pipeline
4. **Embedding**: Query is converted to vector embedding
5. **Retrieval**: Similar documents are retrieved from ChromaDB
6. **Generation**: LLM generates answer using retrieved context
7. **Response**: Answer and sources are returned to UI
8. **Display**: UI shows answer with source attribution

## Storage Architecture

### Vector Database (ChromaDB)

- **Collections**: Documents organized in named collections
- **Embeddings**: High-dimensional vectors for semantic search
- **Metadata**: Document source, type, and additional information
- **Persistence**: Data persisted to local directory

### Document Structure

```json
{
  "text": "Document content...",
  "metadata": {
    "source": "document_source.pdf",
    "type": "article",
    "category": "AI/ML",
    "chunk_id": "doc_1_chunk_0"
  },
  "doc_id": "unique_document_id"
}
```

## Deployment Architecture

### Development

- Local development with hot reloading
- Separate API and UI processes
- Local ChromaDB persistence
- Environment-based configuration

### Docker Compose

- Multi-container setup
- API and UI services
- Shared volumes for data persistence
- Network isolation
- Health checks and dependencies

### Production Considerations

- Container orchestration (Kubernetes)
- Load balancing
- Horizontal scaling
- Persistent storage
- Monitoring and observability
- Security and authentication

## Security Considerations

1. **Input Validation**: All inputs validated using Pydantic models
2. **CORS Configuration**: Properly configured for production
3. **Error Handling**: No sensitive information in error responses
4. **Rate Limiting**: Can be added for production deployment
5. **Authentication**: Extensible for API key or JWT authentication

## Performance Optimizations

1. **Async Processing**: Fully async API for high concurrency
2. **Embedding Caching**: Reuse embeddings where possible
3. **Vector Index Optimization**: Efficient similarity search
4. **Connection Pooling**: Database connection management
5. **Response Streaming**: For large responses (future enhancement)

## Monitoring and Observability

1. **Health Checks**: System health monitoring
2. **Logging**: Structured logging with configurable levels
3. **Metrics**: Performance and usage metrics (extensible)
4. **Error Tracking**: Comprehensive error logging
5. **API Documentation**: Auto-generated OpenAPI docs

## Extensibility

The system is designed for easy extension:

1. **Multiple Vector Stores**: Pluggable vector store interface
2. **Different LLMs**: Support for various language models
3. **Custom Embeddings**: Configurable embedding models
4. **Additional Data Sources**: Extensible document loaders
5. **Custom Processing**: Pluggable text processing pipelines
