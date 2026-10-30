"""El pipeline completo, de punta a punta.

    INGESTA:   audio/video ─► Whisper ─► segmentos ─► chunks ─► embeddings ─► almacén
    CONSULTA:  pregunta ─► búsqueda ─► reranking ─► evaluación LangGraph ─► respuesta + citas

Uso por línea de comandos:
    python -m rag.pipeline ingest data/videos/conversacion.mp3
    python -m rag.pipeline ask "¿Qué descubrieron?"
"""

import argparse
import sys
import time
from pathlib import Path

from .chunking import chunk_segments
from .config import Config, load_config
from .embeddings import Embedder
from .llm import LLM, build_messages, strip_thinking
from .query_graph import build_query_graph
from .rerank import CrossEncoderReranker
from .store import get_store
from .transcribe import load_transcript, transcribe


class RAG:
    def __init__(self, cfg: Config | None = None, embedder=None, store=None, llm=None, reranker=None):
        self.cfg = cfg or load_config()
        # Las dependencias se pueden inyectar (las pruebas usan versiones falsas)
        self.embedder = embedder or Embedder(self.cfg)
        self.store = store or get_store(self.cfg)
        self.llm = llm or LLM(self.cfg)
        self.reranker = reranker or CrossEncoderReranker(self.cfg.reranker_model)
        self._query_graph = None
        self.last_query_info = {}

    # ------------------------------------------------------------- ingesta
    def ingest(self, video: Path, force_transcribe: bool = False):
        """Generador: va devolviendo mensajes de progreso y al final el resumen."""
        t0 = time.time()
        data = None if force_transcribe else load_transcript(self.cfg, video)
        if data:
            yield f"🗂️ Transcripción en caché ({data['engine']}, tardó {data['seconds']} s la primera vez)"
        else:
            yield f"🎧 Transcribiendo con {self.cfg.transcriber} ({self.cfg.whisper_model})…"
            data = transcribe(self.cfg, video)
            yield f"✅ Transcripción lista: {len(data['segments'])} segmentos en {data['seconds']} s"

        t_chunks = time.perf_counter()
        chunks = chunk_segments(data["segments"], self.cfg.chunk_max_chars, self.cfg.chunk_overlap_segments)
        chunk_seconds = time.perf_counter() - t_chunks
        yield (
            f"✂️ {len(chunks)} fragmentos extraídos en {chunk_seconds:.3f} s "
            f"(máx. {self.cfg.chunk_max_chars} caracteres, solapamiento {self.cfg.chunk_overlap_segments})"
        )

        t1 = time.time()
        vectors = self.embedder.embed([c["text"] for c in chunks])
        yield f"🔢 {len(vectors)} embeddings de dimensión {vectors.shape[1]} con {self.cfg.embed_model} en {time.time() - t1:.1f} s"

        self.store.add(video.name, chunks, vectors)
        yield f"📦 Guardado en {self.cfg.vector_store}: {self.store.count()} chunks en total · ingesta en {time.time() - t0:.1f} s"

    # ------------------------------------------------------------- consulta
    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        query_vector = self.embedder.embed_one(question)
        return self.store.search(query_vector, k or self.cfg.top_k)

    def answer(self, question: str, hits: list[dict] | None):
        """Respuesta en streaming. Con hits=None responde SIN RAG (solo memoria del modelo)."""
        for partial in self.llm.stream(build_messages(question, hits)):
            yield strip_thinking(partial)

    def prepare_answer(self, question: str) -> tuple[list[dict], str]:
        """Ejecuta recuperación adaptativa y respuesta; fallback si se agotan reintentos."""
        if self._query_graph is None:
            self._query_graph = build_query_graph(self)
        state = self._query_graph.invoke({
            "question": question,
            "query": question,
            "hits": [],
            "retries": 0,
            "timings": {},
        })
        self.last_query_info = {
            "retries": state.get("retries", 0),
            "timings": state.get("timings", {}),
        }
        return state.get("hits", []), state.get("answer")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG sobre audio y video")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="Transcribir e indexar audio o video")
    p_ing.add_argument("source", type=Path)
    p_ing.add_argument("--force", action="store_true", help="Ignorar la transcripción en caché")
    p_ask = sub.add_parser("ask", help="Hacer una pregunta")
    p_ask.add_argument("question")
    p_ask.add_argument("--no-rag", action="store_true", help="Responder sin recuperar contexto")
    args = parser.parse_args()

    rag = RAG()
    if args.cmd == "ingest":
        for msg in rag.ingest(args.source, args.force):
            print(msg)
    else:
        if args.no_rag:
            hits = None
        else:
            hits, answer = rag.prepare_answer(args.question)
            if not hits:
                print(answer)
                return
        for h in hits or []:
            score = h.get("rerank_score", h["score"])
            print(f"  {score:.3f}  [{h['start']:.0f}s] {h['text'][:90]}…")
        if args.no_rag:
            last = ""
            for partial in rag.answer(args.question, hits):
                if not partial.startswith(last):
                    last = ""
                    sys.stdout.write("\n")
                sys.stdout.write(partial[len(last):])
                sys.stdout.flush()
                last = partial
            print()
        else:
            print(answer)


if __name__ == "__main__":
    main()
