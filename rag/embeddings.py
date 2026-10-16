"""Etapa 3 · Embeddings: texto -> vector de números.

Textos con significado parecido quedan como vectores que apuntan en
direcciones parecidas. Usamos la API de embeddings compatible con OpenAI,
que Ollama también expone: el mismo código sirve para bge-m3 en local o
para text-embedding-3-small en la nube.
"""

import numpy as np
from openai import OpenAI

from .config import Config


class Embedder:
    def __init__(self, cfg: Config):
        self.client = OpenAI(base_url=cfg.embed_base_url, api_key=cfg.embed_api_key or "none")
        self.model = cfg.embed_model

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch_size):
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + batch_size])
            vectors.extend(item.embedding for item in resp.data)
        return np.array(vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
