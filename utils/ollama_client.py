"""Shared Ollama client configured with an API destination, not a bind address."""

from ollama import Client

from config.settings import settings


ollama_client = Client(host=settings.OLLAMA_BASE_URL)
