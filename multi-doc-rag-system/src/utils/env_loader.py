"""Helpers to load environment variables from a .env file."""
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv


def load_env(env_path: str | None = None) -> None:
    """Load environment variables from `env_path` or automatically find a .env file.

    This mutates ``os.environ``.
    """
    if env_path:
        env_file = Path(env_path)
        if not env_file.exists():
            raise FileNotFoundError(f"Specified env file not found: {env_path}")
        load_dotenv(env_file)
    else:
        # find_dotenv returns '' when not found; load_dotenv handles that gracefully
        load_dotenv(find_dotenv())


def get_groq_api_key(env_path: str | None = None) -> str | None:
    """Ensure env is loaded and return the `GROQ_API_KEY` value (or None)."""
    load_env(env_path)
    return os.environ.get("GROQ_API_KEY")
