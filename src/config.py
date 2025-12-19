"""
Configuration module for the Molecular Pipeline.
Loads configuration from environment variables using .env file.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
_env_loaded = False

def _ensure_env_loaded():
    """Ensure .env file is loaded."""
    global _env_loaded
    if not _env_loaded:
        # Look for .env file in project root (parent of src)
        project_root = Path(__file__).parent.parent
        env_path = project_root / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}, using system environment variables")
        _env_loaded = True


def get_ollama_config() -> Dict[str, Any]:
    """
    Get Ollama configuration from environment variables.
    
    Returns:
        Dictionary with ollama configuration
    """
    _ensure_env_loaded()
    return {
        'api_base': os.getenv('OLLAMA_API_BASE', 'http://10.15.0.10:11436'),
        'model': os.getenv('OLLAMA_MODEL', 'gpt-oss:20b'),
        'api_key': os.getenv('OLLAMA_API_KEY', 'ollama'),
        'temperature': float(os.getenv('OLLAMA_TEMPERATURE', '0'))
    }


def get_output_config() -> Dict[str, Any]:
    """
    Get output configuration from environment variables.
    
    Returns:
        Dictionary with output configuration
    """
    _ensure_env_loaded()
    return {
        'directory': os.getenv('OUTPUT_DIRECTORY', 'output'),
        'filename': os.getenv('OUTPUT_FILENAME', 'pathogen_info_output.json')
    }


def get_ollama_llm() -> Any:
    """
    Get LangChain BaseChatModel for Ollama.
    
    Returns:
        LangChain BaseChatModel instance
    """
    from langchain_ollama import ChatOllama
    from langchain_core.language_models.chat_models import BaseChatModel
    
    ollama_config = get_ollama_config()
    model = ollama_config['model'].replace('ollama/', '')
    base_url = ollama_config['api_base']
    
    llm: BaseChatModel = ChatOllama(
        model=model,
        base_url=base_url,
        format='json',
        temperature=ollama_config['temperature'],
    )
    
    return llm


def get_perplexity_config() -> Dict[str, Any]:
    """
    Get Perplexity configuration from environment variables.
    
    Returns:
        Dictionary with perplexity configuration
    """
    _ensure_env_loaded()
    return {
        'api_key': os.getenv('PERPLEXITY_API_KEY', ''),
        'max_tokens': int(os.getenv('PERPLEXITY_MAX_TOKENS', '50000')),
        'max_tokens_per_page': int(os.getenv('PERPLEXITY_MAX_TOKENS_PER_PAGE', '4096')),
        'max_search_results': int(os.getenv('PERPLEXITY_MAX_SEARCH_RESULTS', '10'))
    }


def get_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0) -> Any:
    """
    Get LangChain ChatOpenAI instance.
    
    Args:
        model: OpenAI model name (default: "gpt-4o-mini")
        temperature: Temperature for the model (default: 0)
    
    Returns:
        LangChain ChatOpenAI instance
    """
    from langchain_openai import ChatOpenAI
    
    _ensure_env_loaded()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file")
    
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
    )
    
    return llm


def get_database_config() -> Dict[str, Any]:
    """
    Get database configuration from environment variables.
    
    Returns:
        Dictionary with database configuration including SSH tunnel settings
    """
    _ensure_env_loaded()
    return {
        # SSH Tunnel Configuration
        'ssh_host': os.getenv('DB_SSH_HOST', ''),
        'ssh_port': int(os.getenv('DB_SSH_PORT', '22')),
        'ssh_username': os.getenv('DB_SSH_USERNAME', ''),
        'ssh_password': os.getenv('DB_SSH_PASSWORD', ''),
        'ssh_key_path': os.getenv('DB_SSH_KEY_PATH', ''),  # Path to SSH private key file (optional, alternative to password)
        'ssh_key_passphrase': os.getenv('DB_SSH_KEY_PASSPHRASE', ''),  # Passphrase for SSH key (if key is encrypted)
        
        # Database Configuration (after SSH tunnel)
        'db_type': 'postgresql',  # Only PostgreSQL is supported
        'db_host': os.getenv('DB_HOST', 'localhost'),  # Usually localhost after SSH tunnel
        'db_port': int(os.getenv('DB_PORT', '5432')),  # PostgreSQL default port
        'db_name': os.getenv('DB_NAME', ''),
        'db_username': os.getenv('DB_USERNAME', ''),
        'db_password': os.getenv('DB_PASSWORD', ''),
    }