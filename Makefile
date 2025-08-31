# Makefile for RAG Microservice
.PHONY: install download-data init-db ingest-data setup-data 
.PHONY: test test_vector_store 
.PHONY: run-api run-ui run-dev build up down logs-api logs-ui docker-clean

# Installation and Setup
install: ## Install Python dependencies
	pip install -r requirements.txt

# Data Management
download-data: ## Download and create sample data
	python scripts/download_data.py

init-db: ## Initialize ChromaDB vector database
	python scripts/init_chromadb.py

ingest-data: ## Ingest sample data into vector database
	python scripts/ingest_data.py

setup-data: download-data init-db ingest-data ## Complete data setup pipeline
	@echo "✅ Data setup completed!"

# Testing and Quality
test: ## Run tests with pytest
	pytest tests/

test_vector_store: ## Run vector store tests
	pytest tests/test_vector_store.py

# Local Development
run-api: ## Run the FastAPI server locally
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui: ## Run the Streamlit UI locally
	streamlit run streamlit/app.py --server.address 0.0.0.0 --server.port 8501

run-dev: ## Run both API and UI in development mode
	@echo "Starting API and UI servers..."
	@echo "API will be available at http://localhost:8000"
	@echo "UI will be available at http://localhost:8501"
	@echo "Press Ctrl+C to stop both servers"
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 & \
	streamlit run streamlit/app.py --server.address 0.0.0.0 --server.port 8501 & \
	wait

# Docker Operations
build: ## Build the Docker image
	docker build -f docker/Dockerfile.dev -t rag-microservice:dev .

up: ## Start the Docker containers
	docker-compose up

down: ## Stop the Docker containers
	docker-compose down

logs-api: ## View API logs
	docker-compose -f docker/docker-compose.yml logs -f rag-api

logs-ui: ## View UI logs
	docker-compose -f docker/docker-compose.yml logs -f rag-ui

docker-clean: ## Clean up Docker resources
	docker-compose -f docker/docker-compose.yml down -v
	docker system prune -f