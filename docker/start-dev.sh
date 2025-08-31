#!/bin/bash
set -e

# Function to get token from various sources
get_hf_token() {
    # First try Docker secrets
    if [ -f "/run/secrets/huggingface_token" ]; then
        cat /run/secrets/huggingface_token
    # Fallback to environment variable
    elif [ ! -z "$HUGGINGFACE_TOKEN" ]; then
        echo "$HUGGINGFACE_TOKEN"
    # Fallback to .env file
    elif [ -f .env ]; then
        grep "^HUGGINGFACE_TOKEN=" .env | cut -d"=" -f2- | tr -d "\"'"
    else
        echo ""
    fi
}

# Get the token
HF_TOKEN=$(get_hf_token)

# Login to Hugging Face if token is available
if [ ! -z "$HF_TOKEN" ]; then
    echo "Logging into Hugging Face..."
    echo "$HF_TOKEN" | huggingface-cli login --token "$HF_TOKEN"
    echo "Hugging Face login successful"
else
    echo "Warning: No Hugging Face token found in secrets, environment, or .env file"
fi

# Run data ingestion for development
echo "Running data ingestion script..."
if [ -f "scripts/ingest_data.py" ]; then
    python scripts/ingest_data.py
    echo "Data ingestion completed"
else
    echo "Warning: scripts/ingest_data.py not found, skipping data ingestion"
fi

# Execute the main command
exec "$@"
