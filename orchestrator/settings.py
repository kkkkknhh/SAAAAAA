"""
Centralized settings module for SAAAAAA orchestrator.
This module loads configuration from environment variables and .env file.
Only the orchestrator should read from this module - core modules should not import this.
"""

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in the repository root
REPO_ROOT: Final[Path] = Path(__file__).parent.parent
ENV_FILE: Final[Path] = REPO_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


class Settings:
    """Application settings loaded from environment variables."""

    # Application Settings
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "5000"))
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "dev-secret-key")

    # Database Configuration
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "saaaaaa")
    DB_USER: str = os.getenv("DB_USER", "saaaaaa_user")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # Authentication
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_HOURS: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

    # External Services
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Processing Configuration
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
    TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "300"))

    # Feature Flags
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    ENABLE_MONITORING: bool = os.getenv("ENABLE_MONITORING", "false").lower() == "true"
    ENABLE_RATE_LIMITING: bool = (
        os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
    )


# Global settings instance
settings = Settings()
