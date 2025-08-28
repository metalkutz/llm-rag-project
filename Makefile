# Makefile for RAG Microservice

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