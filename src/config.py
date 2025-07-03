import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    gmail_client_id: str = Field(..., description="Gmail OAuth client ID")
    gmail_client_secret: str = Field(..., description="Gmail OAuth client secret")
    
    embedding_provider: str = Field(default="ollama", description="Embedding provider: 'ollama' or 'openai'")
    
    ollama_model: str = Field(default="nomic-embed-text", description="Ollama model for embeddings")
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_instance_count: int = Field(default=1, description="Number of Ollama instances running on sequential ports")
    
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    
    chroma_persist_directory: Path = Field(default=Path("data/chroma"), description="ChromaDB storage directory")
    
    credentials_path: Path = Field(default=Path("credentials/token.json"), description="OAuth token storage path")
    
    batch_size: int = Field(default=100, description="Batch size for processing emails")
    max_results_per_query: int = Field(default=50, description="Maximum search results")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    global _settings
    _settings = None 
