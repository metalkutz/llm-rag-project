#!/bin/bash
set -e

# Production startup script for Hugging Face authentication
# More robust error handling and logging for production

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to get token from various sources with priority order
get_hf_token() {
    # Priority 1: Docker secrets (recommended for production)
    if [ -f "/run/secrets/huggingface_token" ]; then
        log_message "Using Hugging Face token from Docker secrets"
        cat /run/secrets/huggingface_token
        return 0
    fi
    
    # Priority 2: Environment variable
    if [ ! -z "$HUGGINGFACE_TOKEN" ]; then
        log_message "Using Hugging Face token from environment variable"
        echo "$HUGGINGFACE_TOKEN"
        return 0
    fi
    
    # Priority 3: .env file (fallback for development)
    if [ -f .env ]; then
        local token=$(grep "^HUGGINGFACE_TOKEN=" .env | cut -d"=" -f2- | tr -d "\"'")
        if [ ! -z "$token" ]; then
            log_message "Using Hugging Face token from .env file"
            echo "$token"
            return 0
        fi
    fi
    
    # No token found
    return 1
}

# Main authentication logic
authenticate_huggingface() {
    log_message "Starting Hugging Face authentication process..."
    
    local token
    if token=$(get_hf_token); then
        log_message "Authenticating with Hugging Face..."
        
        # Use --token flag directly for better error handling
        if echo "$token" | huggingface-cli login --token "$token" 2>/dev/null; then
            log_message "Hugging Face authentication successful"
            return 0
        else
            log_message "ERROR: Hugging Face authentication failed"
            return 1
        fi
    else
        log_message "WARNING: No Hugging Face token found. Some models may not be accessible."
        log_message "To use gated models, provide token via Docker secrets, environment variable, or .env file"
        return 0  # Don't fail the container start for missing token
    fi
}

# Pre-flight checks
log_message "Starting RAG service container..."
log_message "Python version: $(python --version)"
log_message "Current working directory: $(pwd)"

# Perform Hugging Face authentication
authenticate_huggingface

# Validate ChromaDB accessibility (production assumes external/existing DB)
validate_chromadb_access() {
    log_message "Validating ChromaDB access..."
    
    # Check if connecting to external ChromaDB service
    if [ ! -z "$CHROMA_SERVER_HOST" ]; then
        log_message "Attempting to connect to ChromaDB service at $CHROMA_SERVER_HOST"
        # You could add a health check here if needed
        # curl -f http://$CHROMA_SERVER_HOST:8000/api/v1/heartbeat || return 1
        log_message "ChromaDB service connection configured"
    # Check if using local persistent storage
    elif [ -d "data/chromadb" ] && [ "$(ls -A data/chromadb 2>/dev/null)" ]; then
        log_message "Found existing ChromaDB data directory with content"
    else
        log_message "WARNING: No ChromaDB data found and no external service configured"
        log_message "Ensure ChromaDB is properly set up before starting the application"
        # Don't fail in production - let the application handle DB initialization
    fi
}

validate_chromadb_access

# Validate required directories exist
if [ ! -d "logs" ]; then
    log_message "Creating logs directory..."
    mkdir -p logs
fi

log_message "Container initialization complete. Starting main application..."

# Execute the main command
exec "$@"
