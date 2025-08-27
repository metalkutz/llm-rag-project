"""
Configuration management for the RAG microservice.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    type: str = Field(default="chromadb", description="Vector store type")
    persist_directory: str = Field(
        default="data/chromadb",
        description="Directory to persist vector store"
    )
    collection_name: str = Field(
        default="rag_documents",
        description="Collection name"
    )

class Settings(BaseSettings):
    """Main settings class."""

    # Component configurations
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    
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