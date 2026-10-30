# Contexto actual del proyecto

Nota de traspaso para continuar en una sesión nueva. Para las reglas de diseño y
la descripción del proyecto, leer primero [`AGENTS.md`](../AGENTS.md).

## Sistema RAG actual

- App Gradio: <http://localhost:7860>. Se levanta con `make up`; revisar con
  `make check`. Ollama corre nativo en el Mac y debe estar abierto aparte.
- El archivo de demo es `data/videos/video.mp4` (10 min 16 s). La transcripción
  MLX está en `data/transcripts/video.json`; se indexó en Qdrant como 19 chunks.
  Si la colección está vacía, reindexar con `make ingest VIDEO=video.mp4`.
- La app también admite audio (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`,
  `.wma`); `data/videos/floristeria.mp3` aparece en el desplegable de ingesta.
  Puede indexarse con `make ingest VIDEO=floristeria.mp3` y las citas abren un
  reproductor de audio en el instante correspondiente.
- `.env` local selecciona `qwen3.8:27b-mlx`, `bge-m3`, Qdrant y
  `LLM_EXTRA_BODY={"reasoning_effort":"none"}`. `.env` está ignorado por Git;
  comprobar los valores locales antes de cambiar proveedores.
- Tras reiniciar el Mac, abrir Ollama, ejecutar `make up` y comprobar `make check`.

## Cambios recientes de código

- Flujo adaptativo de consulta implementado con LangGraph: recuperación vectorial,
  reranking local (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`), evaluación del
  contexto con el LLM, reformulación y hasta 3 reintentos (4 búsquedas contando la inicial). Si no hay
  contexto suficiente, devuelve `Eso no se dice en el video.`. El cross-encoder
  carga bajo demanda; Docker conserva su caché en el volumen `hf_cache`.
- Configuración: `RERANKER_MODEL` y `MAX_QUERY_RETRIES` (`.env.example`). Pruebas
  del flujo inyectan reranker y LLM falsos, sin descargar el modelo.
- El juez de contexto también propone la consulta reformulada en esa misma llamada
  (incluida corrección de erratas como `jupite` → `Júpiter`), evitando una llamada
  LLM independiente por reintento. La UI expone tiempos acumulados de reranking,
  evaluación y generación para diagnosticar latencia.
- `make test` usa `python -m pytest`; en el flujo actual hay 12 pruebas aprobadas.
- `requirements.txt` fija `av<19`, compatible con faster-whisper 1.2.1; PyAV 19
  había retirado el argumento que usaba esa versión.
- `make mac-setup` usa Python 3.12 para el entorno MLX.
- La ingesta muestra tiempo de extracción de chunks. La consulta muestra tiempo
  de recuperación, tiempo de respuesta, tokens y tokens/s promedio; el promedio
  incluye la espera inicial.
- El reproductor detiene/descarga el medio anterior al saltar a otra cita y crea
  uno nuevo en el instante citado. **Queda por confirmar visualmente en Chrome**
  que video y audio saltan al tiempo correcto y que nunca se solapan reproducciones;
  si falla, usar el plan B de `docs/guion.md`.

## Charla

- Formato actual: 35 minutos de exposición + 10 de preguntas.
- PPTX: `slides/conferencia-rag-video.pptx`; texto: `slides/contenido.md`;
  regenerador: `slides/generar.py` (dependencia en `slides/requirements.txt`).
- Guion por diapositiva y plan B: `docs/guion.md`.
- Diagrama Archify: `.archify/architecture-rag-conferencia-20261013-223433/arquitectura-rag.html`.
  Pasó `finalize` y `visual-check`; el reporte sugiere dos cruces de rutas para
  una iteración posterior.

## Pendientes conocidos

- Medir latencias de embeddings, primer token y respuesta completa del LLM con y
  sin razonamiento; la transcripción sí está medida en `docs/tiempos.md`.
- Confirmar explícitamente el backend MLX usado por Ollama para Qwen; se observó
  GPU/Metal, pero no una identificación inequívoca de MLX.
- Probar en Chrome el salto al tiempo citado y la reproducción visual del video.
- Preparar 5–6 preguntas de demo a partir de toda la transcripción.
- Fijar la versión de `qdrant/qdrant` antes de la charla.
- Medir el tiempo de carga inicial y el reranking del cross-encoder en el Mac; por
  defecto corre dentro del contenedor en CPU. Validar calidad de ranking con
  preguntas de la demo en español.

## Instrucción útil para la siguiente sesión

Seguir las decisiones de `AGENTS.md`. Para preguntas de uso o explicación del
sistema, responder basándose primero en este resumen y en el código actual; si se
pide modificar la app, inspeccionar la función afectada antes de editarla.
