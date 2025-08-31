"""
Configuration management for the RAG microservice.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings

class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    model_config = ConfigDict(extra="ignore")
    type: str = Field(
        default="chromadb", 
        description="Vector store type"
    )
    persist_directory: str = Field(
        default="data/chromadb",
        description="Directory to persist vector store"
    )
    collection_name: str = Field(
        default="rag_documents",
        description="Collection name"
    )

class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_config = ConfigDict(extra="ignore")
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model name"
    )
    device: str = Field(
        default="cpu", 
        description="Device to run model on"
        )
    batch_size: int = Field(
        default=32, 
        description="Batch size for embeddings"
        )

class QuantizationConfig(BaseModel):
    """Quantization configuration for LLM optimization."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = Field(
        default=True,
        description="Enable quantization"
    )
    load_in_4bit: bool = Field(
        default=True,
        description="Enable 4-bit quantization"
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="4-bit quantization type (nf4 or fp4)"
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Enable nested quantization"
    )
    compute_dtype: str = Field(
        default="float16",
        description="Computation data type"
    )

class LLMConfig(BaseModel):
    """Language model configuration."""
    model_config = ConfigDict(extra="ignore")
    model_name: str = Field(
        default="microsoft/DialoGPT-medium",
        description="Hugging Face model name"
    )
    max_new_tokens: int = Field(
        default=256, 
        description="Maximum new tokens"
    )
    temperature: float = Field(
        default=0.7, 
        description="Generation temperature"
    )
    top_p: float = Field(
        default=0.9, 
        description="Top-p sampling"
    )
    device: str = Field(
        default="cpu", 
        description="Device to run model on"
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Allow loading remote code for custom models"
    )
    quantization: QuantizationConfig = Field(
        default_factory=QuantizationConfig,
        description="Quantization settings"
    )

class DocumentProcessingConfig(BaseModel):
    """Document processing configuration."""
    model_config = ConfigDict(extra="ignore")
    chunk_size: int = Field(default=512, description="Text chunk size")
    chunk_overlap: int = Field(default=50, description="Chunk overlap size")
    max_documents: int = Field(default=1000, description="Maximum documents to process")

class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    model_config = ConfigDict(extra="ignore")
    top_k: int = Field(
        default=3, 
        description="Number of documents to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.5,
        description="Similarity threshold for retrieval"
    )

class StreamlitConfig(BaseModel):
    """Streamlit UI configuration."""
    host: str = Field(
        default="0.0.0.0", 
        description="Streamlit host"
        )
    port: int = Field(
        default=8501, 
        description="Streamlit port"
        )
    api_url: str = Field(
        default="http://localhost:8000",
        description="RAG API URL"
    )

class APIConfig(BaseModel):
    """API configuration."""
    host: str = Field(
        default="0.0.0.0", 
        description="API host"
        )
    port: int = Field(
        default=8000, 
        description="API port"
        )
    reload: bool = Field(
        default=False, 
        description="Enable auto-reload"
        )
    workers: int = Field(
        default=1, 
        description="Number of workers"
    )

class LoggingConfig(BaseModel):
    """Logging configuration."""
    model_config = ConfigDict(extra="ignore")
    level: str = Field(
        default="INFO", 
        description="Logging level"
    )
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="Log format"
    )
    file: Optional[str] = Field(default=None, description="Log file path")

class Settings(BaseSettings):
    """Main settings class."""
    # Component configurations
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    document_processing: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    # Authentication
    huggingface_token: Optional[str] = Field(
        default=None,
        description="Hugging Face authentication token"
    )
    
    # Environment
    environment: str = Field(
        default="development", 
        description="Environment name"
        )
    debug: bool = Field(
        default=False, 
        description="Debug mode"
        )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

# Global settings instance
_settings: Optional[Settings] = None

def get_settings(config_path: str = "src/config/settings.yaml") -> Settings:
    """Get global settings instance."""
    global _settings
    
    if _settings is None:
        # Load YAML configuration
        yaml_config = load_yaml_config(config_path)
        
        # Create settings with YAML config as defaults
        _settings = Settings(**yaml_config)
    
    return _settings


def reload_settings(config_path: str = "src/config/settings.yaml") -> Settings:
    """Reload settings from configuration files."""
    global _settings
    _settings = None
    return get_settings(config_path)
