# AGENTS.md — contexto para agentes de código

Para el estado operativo más reciente, cambios ya realizados y pendientes
concretos, consultar [`docs/contexto-actual.md`](docs/contexto-actual.md).

## Qué es este proyecto

Material para una **conferencia de 45 minutos** titulada **"¿Cómo enseñarle a una IA
a escuchar?"**: 35 minutos de exposición y 10 de preguntas, dirigida a ~100
estudiantes de Ingeniería Informática de todos los semestres (presencial +
transmisión en vivo). El ponente ejecuta la demo **en su propio Mac**, proyectado
en pantalla; el público no programa durante la charla. Después, los asistentes
pueden clonar el repo y resolver `docs/retos.md`.

El sistema es un **RAG sobre audio y video**: Whisper transcribe archivos de audio
(por ejemplo, una conversación en MP3) o videos (la demo original usa un MP4 de
~10 min con una noticia científica reciente, elegida para que el LLM no la
conozca), se indexan, y las respuestas citan el minuto exacto con un reproductor
que salta a él.

## Hardware y entorno del ponente

- MacBook Pro M2 Max, 32 GB de RAM unificada.
- Ollama **nativo** (no en Docker: Docker en macOS no accede a la GPU).
  Ollama ≥ 0.19 usa MLX en Apple Silicon, pero solo con ≥ 32 GB: verificarlo.
- LLM: Qwen3.8 27B (`qwen3.8:27b` en Ollama; confirmar el tag con `ollama list`).
- Embeddings: `bge-m3` vía Ollama (endpoint OpenAI-compatible `/v1/embeddings`).
- Qdrant y la app Gradio en Docker (`docker-compose.yml`).

## Decisiones de diseño (respetarlas)

1. **Código propio y simple, sin LangChain/LlamaIndex.** El objetivo es didáctico:
   cada etapa debe poder leerse en pantalla. Preferir funciones cortas y legibles
   a abstracciones. Solo se usan librerías para lo que no es el tema (Whisper,
   cliente OpenAI, cliente Qdrant, Gradio).
2. **Todo proveedor es intercambiable por `.env`** usando APIs compatibles con
   OpenAI (`base_url` + `api_key` + `model`). Por defecto, todo local.
3. **Dos almacenes con la misma interfaz:** `NumpyStore` (búsqueda a mano, para
   explicar) y `QdrantStore` (producción). Se elige con `VECTOR_STORE`.
4. **Tres motores de Whisper:** `faster-whisper` (CPU, Docker), `mlx` (nativo en el
   Mac, vía `make transcribe-mac`) y `openai` (API). Las transcripciones se guardan
   en caché en `data/transcripts/<fuente>.json`; la app las reutiliza. Audio y
   video se eligen desde el mismo desplegable de ingesta.
5. **Dockerizar todo lo posible**, excepto Ollama y MLX en el Mac.
6. **Idioma:** código con nombres en inglés; comentarios, docstrings, interfaz y
   documentación en **español**.
7. Las pruebas (`pytest -q`) no requieren red: usan embeddings y LLM falsos, y
   Qdrant en memoria (`QDRANT_URL=":memory:"`). Mantenerlas verdes.

## Estructura

```
rag/config.py       configuración desde .env
rag/transcribe.py   etapa 1: audio/video -> segmentos {start, end, text}
rag/chunking.py     etapa 2: segmentos -> chunks con tiempos
rag/embeddings.py   etapa 3: texto -> vectores (API OpenAI-compatible)
rag/store.py        etapa 4: NumpyStore y QdrantStore
rag/llm.py          etapa 5: prompt con contexto, streaming, citas [n]
rag/pipeline.py     clase RAG (ingest / retrieve / answer) + CLI
app.py              Gradio: pestaña Ingesta y pestaña Preguntar
docs/retos.md       7 retos con soluciones (probadas con fakes)
tests/              pruebas sin red
```

## Comandos

```bash
make help        # lista de atajos
make up          # levanta Qdrant + app (http://localhost:7860)
make check       # verifica Ollama, Qdrant y app
make test        # pruebas dentro del contenedor
make transcribe-mac VIDEO=archivo.mp4   # Whisper con MLX nativo (también MP3)
python -m rag.pipeline ingest data/videos/floristeria.mp3
python -m rag.pipeline ask "pregunta" [--no-rag]
```

## Estado actual

Verificado en un entorno Linux sin GPU ni acceso a Hugging Face:
- Pruebas unitarias (chunking, NumPy, Qdrant en memoria, prompt, citas) ✅
- App Gradio contra un servidor falso OpenAI-compatible: ingesta, recuperación,
  respuesta en streaming, comparación sin RAG, URL del video con soporte de
  `Range` (necesario para saltar al minuto) ✅
- Extracción de audio con PyAV ✅
- Soluciones de los retos, con fakes ✅

Verificado en el Mac del ponente:
- `docker compose build` en arm64; faster-whisper descargó el modelo y transcribió
  el video de `data/videos/`.
- Whisper real: faster-whisper en Docker y mlx-whisper nativo; tiempos registrados
  en `docs/tiempos.md`.
- Ollama 0.34.4: tags `qwen3.8:27b-mlx` y `bge-m3:latest`; `/v1/embeddings`
  respondió con vectores de 1024 dimensiones. La API de chat aceptó
  `reasoning_effort: "none"`.
- Borrador de diapositivas para 35 minutos y guion del ponente creados en
  `slides/` y `docs/guion.md`.

**NO verificado todavía** (hay que hacerlo en el Mac del ponente):
- Tiempos de embeddings, primer token y respuesta completa del LLM con y sin
  razonamiento.
- Confirmar de manera explícita el backend de inferencia de Ollama: se observó
  uso de GPU/Metal, pero no una identificación inequívoca de MLX para Qwen.
- Que el reproductor salte al minuto en el navegador del ponente (media fragment `#t=`).
- Nombres de clases de la tabla LangChain/LlamaIndex del README.
