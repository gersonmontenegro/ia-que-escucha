# Retos: ¿Cómo enseñarle a una IA a escuchar?

Ejercicios para hacer **después de la charla**, sobre este mismo repositorio.
Van de menor a mayor dificultad. Cada uno trae su solución plegada: intenta
resolverlo antes de abrirla.

| # | Reto | Nivel | Qué aprendes |
|---|------|-------|--------------|
| 1 | Chunking por oraciones | 🟢 Básico | Cómo el corte del texto afecta lo que se recupera |
| 2 | Conectar servicios en la nube | 🟢 Básico | Que el RAG no depende de un proveedor |
| 3 | Medir el retriever (hit@k) | 🟡 Intermedio | Evaluar en vez de "a ojo" |
| 4 | Búsqueda híbrida (BM25 + vectores) | 🟡 Intermedio | Por qué los embeddings fallan con nombres propios y cifras |
| 5 | Reranking con el LLM | 🟡 Intermedio | Recuperar mucho y luego filtrar bien |
| 6 | HyDE: buscar con una respuesta inventada | 🔴 Avanzado | Reformular la consulta para acercarla a los documentos |
| 7 | Filtros por video y por tramo de tiempo | 🔴 Avanzado | Metadatos + vectores, como en producción |

**Antes de empezar:** levanta el sistema (`make up`), indexa un video y haz 3 o 4
preguntas. Anota las respuestas: te servirán para comparar después de cada reto.

---

## Reto 1 · Chunking por oraciones 🟢

`rag/chunking.py` corta los chunks cuando se llega a `CHUNK_MAX_CHARS`, aunque sea
a mitad de una idea. Escribe `chunk_by_sentences(segments, max_chars)` que solo
corte cuando un segmento **termina en punto, signo de interrogación o exclamación**,
y que aun así no deje crecer un chunk sin límite.

Luego úsala en `RAG.ingest` en lugar de `chunk_segments`, reindexa y compara las
respuestas.

<details>
<summary>💡 Pista</summary>

Agrega segmentos a la ventana uno por uno. Cierra la ventana si (a) el último
segmento termina una oración **y** la ventana ya tiene un tamaño razonable, o (b)
la ventana se pasó bastante del máximo, sin importar dónde termine.
</details>

<details>
<summary>✅ Solución</summary>

En `rag/chunking.py`:

```python
def chunk_by_sentences(segments: list[dict], max_chars: int = 700) -> list[dict]:
    """Agrupa segmentos, pero solo corta un chunk cuando termina una oración."""
    chunks: list[dict] = []
    window: list[dict] = []
    for seg in segments:
        window.append(seg)
        size = sum(len(s["text"]) + 1 for s in window)
        ends_sentence = seg["text"].rstrip().endswith((".", "?", "!", "…"))
        if (ends_sentence and size >= max_chars * 0.6) or size >= max_chars * 1.5:
            chunks.append(_make_chunk(window, len(chunks)))
            window = []
    if window:
        chunks.append(_make_chunk(window, len(chunks)))
    return chunks
```

En `rag/pipeline.py`, dentro de `ingest`:

```python
from .chunking import chunk_by_sentences
...
chunks = chunk_by_sentences(data["segments"], self.cfg.chunk_max_chars)
```

Reindexa con el botón "Transcribir e indexar" (la transcripción sale de la caché,
así que es rápido). **Para pensar:** ¿qué pasa si Whisper no puso bien la
puntuación? ¿Para qué servía el solapamiento y por qué aquí lo quitamos?
</details>

---

## Reto 2 · Conectar servicios en la nube 🟢

Haz que el sistema use un LLM en la nube **sin tocar una sola línea de Python**.
Después, cambia también el modelo de embeddings a `text-embedding-3-small`.

<details>
<summary>💡 Pista</summary>

Todo el código habla con APIs compatibles con OpenAI. Mira `.env.example`.
Al cambiar el modelo de embeddings, piensa qué pasa con los vectores que ya
estaban guardados.
</details>

<details>
<summary>✅ Solución</summary>

En `.env` (ejemplo con OpenRouter, que sirve el mismo Qwen de la demo):

```env
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL=qwen/qwen3.8-27b
```

Y para embeddings en OpenAI:

```env
EMBED_BASE_URL=https://api.openai.com/v1
EMBED_API_KEY=sk-...
EMBED_MODEL=text-embedding-3-small
```

Reinicia (`docker compose up -d`) y **vuelve a indexar el video**. Los vectores
de `bge-m3` (1024 dimensiones) y los de `text-embedding-3-small` (1536) viven en
espacios distintos: no se pueden comparar. Por eso `Config.collection` crea una
colección distinta por modelo (`videos_bge_m3`, `videos_text_embedding_3_small`);
puedes verlas en el panel de Qdrant: http://localhost:6333/dashboard

**Para pensar:** ¿qué datos salen ahora de tu máquina? ¿Cuánto costaría indexar
100 horas de video?
</details>

---

## Reto 3 · Medir el retriever (hit@k) 🟡

Cambiar parámetros "a ojo" engaña. Escribe 8–10 preguntas sobre el video y anota
**en qué segundo** está la respuesta. Luego calcula **hit@k**: la fracción de
preguntas en las que algún chunk recuperado contiene ese segundo.

Usa la métrica para encontrar la mejor combinación de `CHUNK_MAX_CHARS`
(300, 700, 1200) y `TOP_K` (2, 4, 8).

<details>
<summary>💡 Pista</summary>

Guarda las preguntas en `data/preguntas.json` como
`[{"pregunta": "...", "segundo": 312, "video": "noticia.mp4"}]`. Un chunk
"acierta" si `start <= segundo <= end`. Recuerda reindexar al cambiar el tamaño
de chunk.
</details>

<details>
<summary>✅ Solución</summary>

Crea `scripts/evaluar.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.pipeline import RAG
from rag.transcribe import format_ts


def evaluate(rag, questions: list[dict], k: int) -> float:
    """hit@k: fracción de preguntas cuyo minuto esperado cae dentro de algún chunk recuperado."""
    hits_ok = 0
    for q in questions:
        hits = rag.retrieve(q["pregunta"], k)
        ok = any(
            h["start"] <= q["segundo"] <= h["end"] and h["video"] == q.get("video", h["video"])
            for h in hits
        )
        hits_ok += ok
        print(f"{'✅' if ok else '❌'} {q['pregunta']}  (esperado {format_ts(q['segundo'])})")
    return hits_ok / len(questions)


if __name__ == "__main__":
    rag = RAG()
    questions = json.loads(Path("data/preguntas.json").read_text(encoding="utf-8"))
    for k in (2, 4, 8):
        print(f"\n--- top-k = {k}  (chunk = {rag.cfg.chunk_max_chars} caracteres)")
        print(f"hit@{k} = {evaluate(rag, questions, k):.0%}")
```

Ejecútalo con `docker compose exec app python scripts/evaluar.py`. Para cada
tamaño de chunk: cambia `CHUNK_MAX_CHARS` en `.env`, reinicia, reindexa y vuelve
a correr el script. Arma una tabla de 3×3.

**Para pensar:** un `TOP_K` alto casi siempre sube hit@k. ¿Por qué no usar
siempre 20? (Pista: mira el prompt que recibe el LLM y cuánto tarda en responder.)
</details>

---

## Reto 4 · Búsqueda híbrida: BM25 + vectores 🟡

Los embeddings capturan el **significado**, pero a veces fallan con palabras
exactas: nombres propios, siglas, cifras ("JWST", "K2-18b", "3,2 %"). BM25, el
algoritmo clásico de los buscadores, hace justo lo contrario. Implementa BM25 a
mano y combina ambos rankings con **Reciprocal Rank Fusion (RRF)**.

<details>
<summary>💡 Pista</summary>

RRF no mezcla puntajes (que están en escalas distintas), mezcla **posiciones**:
cada documento suma `1 / (60 + posición)` por cada lista en la que aparece. Los
chunks para BM25 los puedes reconstruir desde `data/transcripts/`.
</details>

<details>
<summary>✅ Solución</summary>

Crea `rag/hybrid.py`:

```python
import json
import math
import re
from collections import Counter

from .chunking import chunk_segments


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 2]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for d in self.docs:
            tf = Counter(d)
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


def all_chunks(cfg) -> list[dict]:
    """Reconstruye los chunks de todos los videos desde las transcripciones en caché."""
    chunks = []
    for path in sorted(cfg.transcripts_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for c in chunk_segments(data["segments"], cfg.chunk_max_chars, cfg.chunk_overlap_segments):
            chunks.append({"video": data["video"], **c})
    return chunks


def hybrid_retrieve(rag, question: str, k: int = 4, rrf_k: int = 60) -> list[dict]:
    """Fusiona el ranking vectorial y el de BM25 con Reciprocal Rank Fusion."""
    chunks = all_chunks(rag.cfg)
    key = lambda c: (c["video"], c["id"])

    vector_hits = rag.retrieve(question, k=20)
    bm25 = BM25([c["text"] for c in chunks]).scores(question)
    bm25_rank = sorted(range(len(chunks)), key=lambda i: -bm25[i])[:20]

    fused: dict = {}
    for rank, h in enumerate(vector_hits):
        fused.setdefault(key(h), {**h, "score": 0.0})["score"] += 1 / (rrf_k + rank + 1)
    for rank, i in enumerate(bm25_rank):
        if bm25[i] > 0:
            fused.setdefault(key(chunks[i]), {**chunks[i], "score": 0.0})["score"] += 1 / (rrf_k + rank + 1)
    return sorted(fused.values(), key=lambda h: -h["score"])[:k]
```

Para usarlo en la app, en `do_ask` de `app.py` cambia
`hits = rag.retrieve(question)` por `hits = hybrid_retrieve(rag, question, cfg.top_k)`.
Mide con el script del reto 3 si mejora.

**Para pensar:** fíjate que la columna "Similitud" ahora muestra puntajes RRF
(alrededor de 0,03). ¿Por qué ya no se parecen a una similitud coseno?
</details>

---

## Reto 5 · Reranking con el LLM 🟡

El buscador vectorial es rápido pero "grueso". Una técnica muy usada es
**recuperar muchos candidatos (10) y reordenarlos** con un modelo más fino,
quedándose con los 4 mejores. Usa el propio LLM como juez de relevancia.

<details>
<summary>💡 Pista</summary>

Por cada candidato, pídele al LLM un número del 0 al 10 con `rag.llm.complete(...)`
y `temperature=0`. Extrae el número con una expresión regular: los modelos a
veces agregan texto aunque les pidas que no lo hagan.
</details>

<details>
<summary>✅ Solución</summary>

Crea `rag/rerank.py`:

```python
import re

RERANK_PROMPT = """Califica de 0 a 10 qué tan útil es el fragmento para responder la pregunta.
Responde SOLO con el número.

Pregunta: {q}

Fragmento: {t}"""


def rerank(rag, question: str, candidates: int = 10, k: int = 4) -> list[dict]:
    hits = rag.retrieve(question, k=candidates)
    for h in hits:
        reply = rag.llm.complete(
            [{"role": "user", "content": RERANK_PROMPT.format(q=question, t=h["text"])}], temperature=0
        )
        m = re.search(r"\d+(\.\d+)?", reply)
        h["rerank"] = float(m.group()) if m else 0.0
    return sorted(hits, key=lambda h: -h["rerank"])[:k]
```

**Para pensar:** son 10 llamadas extra al LLM por pregunta. En producción se
usa un *cross-encoder* dedicado (por ejemplo `bge-reranker-v2-m3`, de la misma
familia que nuestros embeddings), mucho más rápido. ¿Cómo lo agregarías?
</details>

---

## Reto 6 · HyDE: buscar con una respuesta inventada 🔴

Una pregunta ("¿qué animal vive mucho?") y el fragmento que la responde ("la
tortuga gigante supera los cien años…") no se parecen tanto como textos. **HyDE**
(*Hypothetical Document Embeddings*) le pide al LLM que **invente** una respuesta
plausible y busca con ella: una respuesta, aunque sea falsa, se parece más a otra
respuesta que a una pregunta.

<details>
<summary>💡 Pista</summary>

La respuesta inventada nunca se le muestra al usuario: solo se usa para calcular
el vector de búsqueda. Puedes concatenar pregunta + respuesta inventada.
</details>

<details>
<summary>✅ Solución</summary>

```python
HYDE_PROMPT = """Escribe un párrafo corto (3 frases) que podría decir el presentador de una
noticia científica para responder esta pregunta. No importa si no sabes la respuesta
real: inventa una respuesta plausible.

Pregunta: {q}"""


def hyde_retrieve(rag, question: str, k: int = 4) -> list[dict]:
    hypothetical = rag.llm.complete([{"role": "user", "content": HYDE_PROMPT.format(q=question)}])
    vector = rag.embedder.embed_one(question + "\n" + hypothetical)
    return rag.store.search(vector, k)
```

**Para pensar:** ¿es irónico usar una alucinación para *evitar* alucinaciones?
¿En qué tipo de preguntas HyDE empeora los resultados? Compruébalo con el reto 3.
</details>

---

## Reto 7 · Filtros por video y por tramo de tiempo 🔴

Indexa **dos videos**. Luego permite preguntas como *"¿qué dice el segundo video
en sus primeros dos minutos?"*: la búsqueda debe filtrar por `video` y por rango
de tiempo **antes** de comparar vectores. Impleméntalo para los dos almacenes
(NumPy y Qdrant) y agrega a la app un desplegable para elegir el video.

<details>
<summary>💡 Pista</summary>

Un chunk se cruza con el tramo `[desde, hasta]` si `end >= desde` y
`start <= hasta`. En Qdrant usa `Filter`, `FieldCondition`, `MatchValue` y
`Range`, y pásalo como `query_filter` en `query_points`.
</details>

<details>
<summary>✅ Solución</summary>

En `NumpyStore` (`rag/store.py`):

```python
def search_filtered(self, query_vector, k, video=None, desde=None, hasta=None):
    idx = [
        i for i, p in enumerate(self.payloads)
        if (video is None or p["video"] == video)
        and (desde is None or p["end"] >= desde)
        and (hasta is None or p["start"] <= hasta)
    ]
    if not idx:
        return []
    q = _normalize(query_vector.astype(np.float32))
    scores = self.vectors[idx] @ q
    order = np.argsort(-scores)[:k]
    return [{"score": float(scores[j]), **self.payloads[idx[j]]} for j in order]
```

En `QdrantStore`:

```python
def search_filtered(self, query_vector, k, video=None, desde=None, hasta=None):
    from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

    must = []
    if video:
        must.append(FieldCondition(key="video", match=MatchValue(value=video)))
    if desde is not None:
        must.append(FieldCondition(key="end", range=Range(gte=desde)))
    if hasta is not None:
        must.append(FieldCondition(key="start", range=Range(lte=hasta)))
    res = self.client.query_points(
        self.collection, query=query_vector.tolist(), limit=k,
        query_filter=Filter(must=must) if must else None, with_payload=True,
    )
    return [{"score": p.score, **p.payload} for p in res.points]
```

En `app.py`, agrega en la pestaña "Preguntar" un
`gr.Dropdown(choices=["(todos)"] + list_videos())`, pásalo como entrada de
`do_ask` y cambia la recuperación por:

```python
video = None if video_sel == "(todos)" else video_sel
hits = rag.store.search_filtered(rag.embedder.embed_one(question), cfg.top_k, video=video)
```

**Para pensar:** en NumPy filtramos con un `for` sobre *todos* los chunks; con
un millón de chunks eso es lento. Qdrant filtra **dentro** del índice HNSW. Para
que sea eficiente a gran escala se crean índices de payload:
`client.create_payload_index(collection, "video", "keyword")`. ¿Cuándo vale la
pena crearlos?
</details>
