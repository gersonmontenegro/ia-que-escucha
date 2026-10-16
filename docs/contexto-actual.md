# Contexto actual del proyecto

Nota de traspaso para continuar en una sesión nueva. Para las reglas de diseño y
la descripción del proyecto, leer primero [`AGENTS.md`](../AGENTS.md).

## Sistema RAG actual

- App Gradio: <http://localhost:7860>. Se levanta con `make up`; revisar con
  `make check`. Ollama corre nativo en el Mac y debe estar abierto aparte.
- El archivo de demo es `data/videos/video.mp4` (10 min 16 s). La transcripción
  MLX está en `data/transcripts/video.json`; se indexó en Qdrant como 19 chunks.
  Si la colección está vacía, reindexar con `make ingest VIDEO=video.mp4`.
- `.env` local selecciona `qwen3.8:27b-mlx`, `bge-m3`, Qdrant y
  `LLM_EXTRA_BODY={"reasoning_effort":"none"}`. `.env` está ignorado por Git;
  comprobar los valores locales antes de cambiar proveedores.
- Tras reiniciar el Mac, abrir Ollama, ejecutar `make up` y comprobar `make check`.

## Cambios recientes de código

- `make test` usa `python -m pytest`; resultado de la última ejecución: 6 pruebas
  aprobadas.
- `requirements.txt` fija `av<19`, compatible con faster-whisper 1.2.1; PyAV 19
  había retirado el argumento que usaba esa versión.
- `make mac-setup` usa Python 3.12 para el entorno MLX.
- La ingesta muestra tiempo de extracción de chunks. La consulta muestra tiempo
  de recuperación, tiempo de respuesta, tokens y tokens/s promedio; el promedio
  incluye la espera inicial.
- El reproductor detiene/descarga el video anterior al saltar a otra cita y crea
  uno nuevo en el instante citado. **Queda por confirmar visualmente en Chrome**
  que el video se reproduce y que nunca se solapan audios; si falla, usar el plan
  B de `docs/guion.md`.

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

## Instrucción útil para la siguiente sesión

Seguir las decisiones de `AGENTS.md`. Para preguntas de uso o explicación del
sistema, responder basándose primero en este resumen y en el código actual; si se
pide modificar la app, inspeccionar la función afectada antes de editarla.
