import os
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)


class LLMConfig(BaseModel):
    """LLM configuration"""
    model: str = Field(default="gpt-3.5-turbo")
    base_url: Optional[str] = Field(default=None)
    api_key: str = Field(...)
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=4000)
    seed: Optional[int] = Field(default=42)


class DatabaseConfig(BaseModel):
    """Database configuration"""
    database_url: str = Field(default="sqlite:///data/Chinook.db")
    connection_timeout: int = Field(default=30)
    max_query_time: int = Field(default=60)


class SecurityConfig(BaseModel):
    """Security configuration"""
    allow_only_select: bool = Field(default=True)
    max_retry_attempts: int = Field(default=3)
    enable_query_validation: bool = Field(default=True)
    blocked_patterns: list = Field(default_factory=lambda: [
        r'\bINSERT\b', r'\bUPDATE\b', r'\bDELETE\b', 
        r'\bDROP\b', r'\bCREATE\b', r'\bALTER\b'
    ])


class WorkflowConfig(BaseModel):
    """Workflow configuration"""
    enable_debug: bool = Field(default=False)
    enable_tracing: bool = Field(default=True)
    max_execution_time: int = Field(default=300)  # 5 minutes
    result_limit: int = Field(default=100)
    recursion_limit: int = Field(default=20)  # LangGraph recursion limit


class Settings(BaseModel):
    """Application settings"""
    
    # LLM settings
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(
        model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
        base_url=os.getenv('OPENAI_BASE_URL'),
        api_key=os.getenv('OPENAI_API_KEY', ''),
        temperature=float(os.getenv('LLM_TEMPERATURE', '0.1')),
        max_tokens=int(os.getenv('LLM_MAX_TOKENS', '4000'))
    ))
    
    # Database settings
    database: DatabaseConfig = Field(default_factory=lambda: DatabaseConfig(
        database_url=os.getenv('DATABASE_URL', 'sqlite:///data/Chinook.db'),
        connection_timeout=int(os.getenv('DB_TIMEOUT', '30'))
    ))
    
    # Security settings
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Workflow settings  
    workflow: WorkflowConfig = Field(default_factory=lambda: WorkflowConfig(
        enable_debug=os.getenv('DEBUG', 'False').lower() == 'true',
        enable_tracing=os.getenv('ENABLE_TRACING', 'True').lower() == 'true'
    ))
    
    @classmethod
    def load_from_env(cls) -> "Settings":
        """Load settings from environment variables"""
        return cls()
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        
        # Check required fields
        if not self.llm.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        if not self.database.database_url:
            raise ValueError("DATABASE_URL is required")
        
        return True


# Global settings instance
settings = Settings.load_from_env()