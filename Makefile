# Makefile for RAG Microservice
.PHONY: install download-data init-db ingest-data setup-data 
.PHONY: test test_vector_store 
.PHONY: run-api run-ui run-dev 
.PHONY: build-dev build-prod up-dev up-prod down-dev down-prod
.PHONY: logs-dev logs-prod-api logs-prod-ui logs-prod-chromadb
.PHONY: docker-clean docker-clean-dev docker-clean-prod

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
build-dev: ## Build the development Docker image
	docker build -f docker/Dockerfile.dev -t rag-microservice:dev .

build-prod: ## Build the production Docker image
	docker build -f docker/Dockerfile -t rag-microservice:prod .

up-dev: ## Start development Docker containers
	cd docker && docker compose -f docker-compose.dev.yml up -d

up-prod: ## Start production Docker containers
	cd docker && docker compose -f docker-compose.yml up -d

down-dev: ## Stop development Docker containers
	cd docker && docker compose -f docker-compose.dev.yml down

down-prod: ## Stop production Docker containers
	cd docker && docker compose -f docker-compose.yml down

logs-dev: ## View development container logs
	cd docker && docker compose -f docker-compose.dev.yml logs -f

logs-prod-api: ## View production API logs
	cd docker && docker compose -f docker-compose.yml logs -f rag-api

logs-prod-ui: ## View production UI logs
	cd docker && docker compose -f docker-compose.yml logs -f rag-ui

logs-prod-chromadb: ## View production ChromaDB logs
	cd docker && docker compose -f docker-compose.yml logs -f chromadb

docker-clean-dev: ## Clean up development Docker resources
	cd docker && docker compose -f docker-compose.dev.yml down -v
	docker image rm rag-microservice:dev 2>/dev/null || true

docker-clean-prod: ## Clean up production Docker resources
	cd docker && docker compose -f docker-compose.yml down -v
	docker image rm rag-microservice:prod 2>/dev/null || true

docker-clean: docker-clean-dev docker-clean-prod ## Clean up all Docker resources
	docker system prune -f