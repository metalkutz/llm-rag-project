# RAG Microservice 🤖

A production-ready Retrieval-Augmented Generation (RAG) microservice built with FastAPI, Streamlit, LlamaIndex, LangChain, and ChromaDB. This system combines the power of large language models with external knowledge retrieval to provide accurate, contextual responses.

## ✨ Features

- **FastAPI REST API** with async support, Pydantic validation, and comprehensive error handling
- **Streamlit Chat UI** with real-time conversation, source attribution, and health monitoring
- **RAG Pipeline** using LlamaIndex for data ingestion and LangChain for orchestration
- **Vector Database** powered by ChromaDB with persistent storage
- **Embedding Models** via Hugging Face Transformers and SentenceTransformers
- **Docker Support** with development and production configurations
- **Comprehensive Testing** with pytest, coverage reports, and CI/CD ready
- **Production Ready** with health checks, logging, and monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Git

### 1. Clone and Setup

```bash
git clone https://github.com/metalkutz/llm-rag-project.git
cd llm-rag-project

# Copy environment configuration
cp .env.example .env

# Install dependencies
make install
```

### 2. Download and Prepare Data

```bash
# Download sample AI/ML educational data
make download-data

# Initialize ChromaDB vector database
make init-db

# Ingest documents into the database
make ingest-data

# Or run all data setup steps at once
make setup-data
```

### 3. Run the Application

#### Option A: Local Development

```bash
# Terminal 1: Start the API server
make run-api

# Terminal 2: Start the Streamlit UI
make run-ui
```

#### Option B: Docker Compose (Recommended)

```bash
# Start both services with Docker
make run-docker-dev

# Or use docker-compose directly
docker-compose -f docker/docker-compose.yml up --build
```

### 4. Access the Application

- **Chat Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
llm-rag-project/
├── src/                          # Source code
│   ├── api/                      # FastAPI application
│   │   ├── main.py              # API server and endpoints
│   │   ├── models.py            # Pydantic models
│   ├── rag/                      # RAG pipeline components
│   │   ├── pipeline.py          # Main RAG orchestrator
│   │   ├── vector_store.py      # ChromaDB integration
│   └── config/                   # Configuration management
│       ├── config.py            # Settings and config loader
│       └── settings.yaml        # Default configuration
├── streamlit/                    # Streamlit UI application
│   └── app.py                   # Chat interface
├── scripts/                      # Automation scripts
│   ├── download_data.py         # Sample data creation
│   ├── init_chromadb.py         # Database initialization
│   ├── import_pdf.py            # Import data from pdfs
│   └── ingest_data.py           # Document ingestion
├── tests/                        # Test suite
│   ├── test_llm-rag.py          # API and RAG pipeline tests
│   └── test_vector_store.py     # Vector store tests
├── docker/                       # Docker configurations
│   ├── Dockerfile               # Production image
│   ├── Dockerfile.dev           # Development image
│   ├── Dockerfile.ui            # Streamlit service image
│   ├── docker-compose.dev.yml   # Multi-service dev image
│   └── docker-compose.yml       # Multi-service setup
├── data/                         # Data directory
│   └── sample_qa.json           # Generated sample data
├── requirements.txt              # Python dependencies
├── Makefile                      # Development commands
├── .env.example                  # Environment template
└── .gitignore                    # Git ignore rules
```

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
LLM_MODEL_NAME=google/gemma-2-2b-it # requires HF token 

# Vector Store
VECTOR_STORE_PERSIST_DIRECTORY=data/chromadb
VECTOR_STORE_COLLECTION_NAME=rag_documents

# UI Configuration
STREAMLIT_API_URL=http://localhost:8000
```

### YAML Configuration

Edit `src/config/settings.yaml` for detailed configuration:

```yaml
api:
  host: "0.0.0.0"
  port: 8000

embedding:
  model_name: "all-MiniLM-L6-v2"
  device: "cpu"

llm:
  model_name: "google/gemma-2-2b-it"
  max_new_tokens: 256
  temperature: 0.7

vector_store:
  type: "chromadb"
  persist_directory: "data/chromadb"
  collection_name: "rag_documents"
```

## 🧪 Testing

```bash
# Run all tests with coverage
make test

# Run specific test files
pytest tests/test_llm-rag.py -v
pytest tests/test_vector_store.py -v
```
## 🐳 Docker Development

### Development with Docker

```bash
# Build development image
make build-docker-dev

# Run development environment
make up-dev

# View logs
make logs-dev

# Stop services
make down-dev
```

## 📊 Sample Data

The system comes with sample data about RAG topics:

- **Q&A Pairs**: 10 educational Q&A pairs covering RAG, embeddings, vector databases, etc.
- **Articles**: Educational articles about transformers, RAG systems, and vector databases
- **Combined Dataset**: Mixed content for comprehensive testing

### Sample Questions to Try

- "What is Retrieval-Augmented Generation (RAG)?"
- "How does ChromaDB work as a vector database?"
- "What are the benefits of using LangChain for RAG pipelines?"
- "Explain the role of embeddings in RAG systems"
- "How do you evaluate RAG system performance?"

## 🚀 Deployment

### Local Development

```bash
# Full development setup
make quickstart
```

### Docker Compose

```bash
# Production deployment
docker-compose -f docker/docker-compose.yml up -d
```

### Production Considerations

- Use environment-specific configuration files
- Set up proper logging and monitoring
- Configure authentication and rate limiting
- Use persistent volumes for data
- Set up load balancing for high availability
- Configure HTTPS with proper certificates

## 📚 API Documentation

### Main Endpoints

#### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "message": "RAG microservice is running",
  "details": {
    "api_version": "1.0.0",
    "rag_pipeline": "ready"
  }
}
```

#### Query RAG System
```http
POST /query
Content-Type: application/json

{
  "query": "What is machine learning?",
  "max_sources": 3,
  "include_metadata": true
}
```

Response:
```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "sources": ["AI Research Papers", "ML Textbook"],
  "confidence_score": 0.85,
  "query_id": "query-123",
  "metadata": {
    "source_details": [...],
    "response_length": 150,
    "num_sources": 2
  }
}
```

## 📈 Performance

- **Async API**: Fully asynchronous FastAPI for high concurrency
- **Efficient Embeddings**: Optimized sentence transformers
- **Vector Search**: Fast similarity search with ChromaDB
- **Caching**: Configurable caching for repeated queries
- **Streaming**: Response streaming for large outputs (extensible)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for RAG pipeline components
- [LangChain](https://python.langchain.com/) for LLM orchestration
- [ChromaDB](https://www.trychroma.com/) for vector database
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Streamlit](https://streamlit.io/) for the UI framework
- [Hugging Face](https://huggingface.co/) for transformer models
