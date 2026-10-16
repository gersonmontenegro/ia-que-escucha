"""Pruebas sin red: embeddings y LLM falsos, NumPy y Qdrant en memoria.

    pytest -q
"""

import json
import re

import numpy as np
import pytest

from rag.chunking import chunk_segments
from rag.config import Config
from rag.llm import build_messages, cited_numbers, strip_thinking
from rag.pipeline import RAG
from rag.store import NumpyStore, QdrantStore
from rag.transcribe import format_ts

SEGMENTS = [
    {"start": 0.0, "end": 5.0, "text": "Hoy hablamos de un nuevo telescopio espacial."},
    {"start": 5.0, "end": 11.0, "text": "El telescopio detectó agua en un exoplaneta lejano."},
    {"start": 11.0, "end": 18.0, "text": "Los científicos usaron espectroscopía infrarroja."},
    {"start": 18.0, "end": 25.0, "text": "Mientras tanto, en biología, se secuenció un genoma de tortuga."},
    {"start": 25.0, "end": 31.0, "text": "La tortuga vive más de cien años gracias a genes de reparación."},
]


class FakeEmbedder:
    """Bolsa de palabras con hashing: suficiente para que 'tortuga' encuentre 'tortuga'."""

    def embed(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                if len(w) > 3:
                    out[i, hash(w) % 64] += 1
        return out

    def embed_one(self, text):
        return self.embed([text])[0]


class FakeLLM:
    def stream(self, messages):
        yield "<think>pensando…"
        yield "<think>pensando…</think>La tortuga"
        yield "<think>pensando…</think>La tortuga vive mucho [2][1]."


@pytest.fixture
def cfg(tmp_path):
    c = Config(data_dir=tmp_path, chunk_max_chars=120, chunk_overlap_segments=1, top_k=2)
    for d in (c.videos_dir, c.transcripts_dir, c.index_dir):
        d.mkdir(parents=True)
    return c


def test_chunking_conserva_tiempos_y_solapa():
    chunks = chunk_segments(SEGMENTS, max_chars=120, overlap_segments=1)
    assert len(chunks) > 1
    assert chunks[0]["start"] == 0.0
    assert chunks[-1]["end"] == 31.0
    # el último segmento de un chunk reaparece al inicio del siguiente
    last_text = chunks[0]["text"].split(". ")[-1]
    assert chunks[1]["text"].startswith(last_text.rstrip("."))
    assert all(len(c["text"]) <= 120 * 2 for c in chunks)


def test_chunking_sin_solapamiento_no_repite():
    chunks = chunk_segments(SEGMENTS, max_chars=120, overlap_segments=0)
    joined = " ".join(c["text"] for c in chunks)
    assert joined == " ".join(s["text"] for s in SEGMENTS)


@pytest.mark.parametrize("store_cls", ["numpy", "qdrant"])
def test_ingesta_y_busqueda(cfg, store_cls):
    cfg.vector_store = store_cls
    cfg.qdrant_url = ":memory:"
    store = NumpyStore(cfg) if store_cls == "numpy" else QdrantStore(cfg)
    rag = RAG(cfg, embedder=FakeEmbedder(), store=store, llm=FakeLLM())

    video = cfg.videos_dir / "noticia.mp4"
    (cfg.transcripts_dir / "noticia.json").write_text(
        json.dumps({"video": video.name, "engine": "fake", "model": "x", "seconds": 1, "segments": SEGMENTS})
    )
    log = list(rag.ingest(video))
    assert "caché" in log[0]
    n = store.count()
    assert n > 0

    # reindexar el mismo video no duplica
    list(rag.ingest(video))
    assert store.count() == n

    hits = rag.retrieve("¿Cuántos años vive la tortuga?")
    assert "tortuga" in hits[0]["text"].lower()
    assert hits[0]["video"] == "noticia.mp4"
    assert hits[0]["score"] >= hits[-1]["score"]

    final = list(rag.answer("¿Cuántos años vive la tortuga?", hits))[-1]
    assert final == "La tortuga vive mucho [2][1]."
    assert cited_numbers(final) == [2, 1]


def test_numpy_persiste(cfg):
    store = NumpyStore(cfg)
    store.add("a.mp4", [{"id": 0, "start": 0, "end": 1, "text": "hola"}], np.ones((1, 4)))
    assert NumpyStore(cfg).count() == 1


def test_prompt_y_utilidades():
    msgs = build_messages("¿qué?", [{"video": "v.mp4", "start": 65, "end": 130, "text": "texto"}])
    assert "[1] (v.mp4, 01:05–02:10)" in msgs[1]["content"]
    assert build_messages("¿qué?", None)[0]["content"].startswith("Eres un asistente útil")
    assert strip_thinking("<think>a</think> hola") == "hola"
    assert strip_thinking("<think>sin cerrar") == ""
    assert format_ts(3725) == "1:02:05"
