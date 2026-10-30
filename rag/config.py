"""Configuración del sistema, leída de variables de entorno (archivo .env).

Todo lo que cambia entre "local" y "nube" vive aquí: para pasar de Ollama a
OpenRouter u OpenAI basta con cambiar una URL, una clave y un nombre de modelo.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv es opcional: en Docker las variables ya vienen de env_file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass
class Config:
    # --- Rutas ---
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))

    # --- Transcripción (Whisper) ---
    # faster-whisper: corre en CPU, también dentro de Docker
    # mlx: nativo en Apple Silicon (GPU), fuera de Docker
    # openai: API compatible con OpenAI (OpenAI, Groq...)
    transcriber: str = field(default_factory=lambda: _env("TRANSCRIBER", "faster-whisper"))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "large-v3-turbo"))
    whisper_language: str = field(default_factory=lambda: _env("WHISPER_LANGUAGE", "es"))
    transcribe_base_url: str = field(default_factory=lambda: _env("TRANSCRIBE_BASE_URL", "https://api.openai.com/v1"))
    transcribe_api_key: str = field(default_factory=lambda: _env("TRANSCRIBE_API_KEY", ""))

    # --- Chunking ---
    chunk_max_chars: int = field(default_factory=lambda: int(_env("CHUNK_MAX_CHARS", "700")))
    chunk_overlap_segments: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP_SEGMENTS", "1")))

    # --- Embeddings (cualquier API compatible con OpenAI; Ollama incluido) ---
    embed_base_url: str = field(default_factory=lambda: _env("EMBED_BASE_URL", "http://localhost:11434/v1"))
    embed_api_key: str = field(default_factory=lambda: _env("EMBED_API_KEY", "ollama"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "bge-m3"))

    # --- Base vectorial ---
    vector_store: str = field(default_factory=lambda: _env("VECTOR_STORE", "qdrant"))  # qdrant | numpy
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY", ""))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))

    # --- Reranking local ---
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
    max_query_retries: int = field(default_factory=lambda: int(_env("MAX_QUERY_RETRIES", "3")))

    # --- LLM (cualquier API compatible con OpenAI; Ollama incluido) ---
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "ollama"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3.8:27b"))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.2")))
    # Parámetros extra que se envían tal cual al proveedor (JSON).
    # Ej. para apagar el "thinking" en Ollama: {"reasoning_effort": "none"}
    llm_extra_body: dict = field(default_factory=lambda: json.loads(_env("LLM_EXTRA_BODY", "{}") or "{}"))

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def collection(self) -> str:
        """Una colección por modelo de embeddings: los vectores de modelos
        distintos no son comparables (ni siquiera tienen la misma dimensión)."""
        safe = "".join(c if c.isalnum() else "_" for c in self.embed_model)
        return f"videos_{safe}"


def load_config() -> Config:
    cfg = Config()
    for d in (cfg.videos_dir, cfg.transcripts_dir, cfg.index_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cfg
