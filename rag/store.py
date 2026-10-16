"""Etapa 4 · Almacén vectorial: guardar vectores y buscar los más parecidos.

Dos implementaciones con la MISMA interfaz, intercambiables con VECTOR_STORE:

  - NumpyStore  : todo a mano, en ~30 líneas. Sirve para ver que la "búsqueda
                  semántica" es, en el fondo, un producto punto y un ordenamiento.
  - QdrantStore : base de datos vectorial de producción (índice HNSW, filtros,
                  persistencia, escalado). Corre en Docker.

Cada resultado es un dict: {"score", "video", "start", "end", "text", "id"}.
"""

import json
import uuid

import numpy as np

from .config import Config


def get_store(cfg: Config):
    if cfg.vector_store == "numpy":
        return NumpyStore(cfg)
    if cfg.vector_store == "qdrant":
        return QdrantStore(cfg)
    raise ValueError(f"VECTOR_STORE desconocido: {cfg.vector_store}")


def _normalize(m: np.ndarray) -> np.ndarray:
    """Vectores de longitud 1: así el producto punto ES la similitud coseno."""
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


# ----------------------------------------------------------------------------
# NumPy: búsqueda por fuerza bruta, hecha a mano
# ----------------------------------------------------------------------------
class NumpyStore:
    def __init__(self, cfg: Config):
        self.path = cfg.index_dir / f"{cfg.collection}.npz"
        self.meta_path = cfg.index_dir / f"{cfg.collection}.json"
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.payloads: list[dict] = []
        if self.path.exists():
            self.vectors = np.load(self.path)["vectors"]
            self.payloads = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def add(self, video: str, chunks: list[dict], vectors: np.ndarray) -> None:
        self.delete_video(video)  # reindexar un video reemplaza lo anterior
        new = _normalize(vectors.astype(np.float32))
        self.vectors = new if len(self.payloads) == 0 else np.vstack([self.vectors, new])
        self.payloads += [{"video": video, **c} for c in chunks]
        self._save()

    def search(self, query_vector: np.ndarray, k: int) -> list[dict]:
        if not self.payloads:
            return []
        q = _normalize(query_vector.astype(np.float32))
        scores = self.vectors @ q                # similitud coseno con TODOS los chunks
        best = np.argsort(-scores)[:k]           # los k más altos
        return [{"score": float(scores[i]), **self.payloads[i]} for i in best]

    def delete_video(self, video: str) -> None:
        keep = [i for i, p in enumerate(self.payloads) if p["video"] != video]
        if len(keep) != len(self.payloads):
            self.vectors = self.vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
            self.payloads = [self.payloads[i] for i in keep]
            self._save()

    def count(self) -> int:
        return len(self.payloads)

    def _save(self) -> None:
        np.savez(self.path, vectors=self.vectors)
        self.meta_path.write_text(json.dumps(self.payloads, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------
# Qdrant: base de datos vectorial
# ----------------------------------------------------------------------------
class QdrantStore:
    def __init__(self, cfg: Config):
        from qdrant_client import QdrantClient

        # QDRANT_URL=":memory:" -> modo embebido en memoria (útil para pruebas)
        if cfg.qdrant_url == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key or None)
        self.collection = cfg.collection

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def add(self, video: str, chunks: list[dict], vectors: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        self._ensure_collection(vectors.shape[1])
        self.delete_video(video)
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video}:{c['id']}")),
                vector=v.tolist(),
                payload={"video": video, **c},
            )
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(self.collection, points=points, wait=True)

    def search(self, query_vector: np.ndarray, k: int) -> list[dict]:
        if not self.client.collection_exists(self.collection):
            return []
        res = self.client.query_points(self.collection, query=query_vector.tolist(), limit=k, with_payload=True)
        return [{"score": p.score, **p.payload} for p in res.points]

    def delete_video(self, video: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        if not self.client.collection_exists(self.collection):
            return
        flt = Filter(must=[FieldCondition(key="video", match=MatchValue(value=video))])
        self.client.delete(self.collection, points_selector=FilterSelector(filter=flt), wait=True)

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count
