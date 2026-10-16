# Atajos para preparar y correr la demo.  Uso:  make <objetivo>
VIDEO ?= noticia.mp4
Q ?= ¿De qué trata el video?
PYTHON ?= python3.12

.PHONY: help env models up down logs test check ingest ask mac-setup transcribe-mac

help:            ## Lista los objetivos disponibles
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

env:             ## Crea .env a partir de .env.example
	@test -f .env || cp .env.example .env && echo ".env listo"

models:          ## Descarga los modelos de Ollama (nativo en el Mac)
	ollama pull bge-m3
	@echo "LLM disponible:" && ollama list

up: env          ## Construye y levanta Qdrant + la app en Docker
	docker compose up -d --build
	@echo "App:    http://localhost:7860"
	@echo "Qdrant: http://localhost:6333/dashboard"

down:            ## Detiene los contenedores (los datos se conservan)
	docker compose down

logs:            ## Muestra los logs de la app
	docker compose logs -f app

test:            ## Corre las pruebas dentro del contenedor
	docker compose run --rm app python -m pytest -q

check:           ## Verifica que Ollama, Qdrant y la app respondan
	@curl -sf http://localhost:11434/api/tags >/dev/null && echo "✅ Ollama" || echo "❌ Ollama (¿ollama serve?)"
	@curl -sf http://localhost:6333/healthz >/dev/null && echo "✅ Qdrant" || echo "❌ Qdrant"
	@curl -sf http://localhost:7860 >/dev/null && echo "✅ App" || echo "❌ App"

ingest:          ## Transcribe e indexa data/videos/$(VIDEO) desde la terminal
	docker compose exec app python -m rag.pipeline ingest data/videos/$(VIDEO)

ask:             ## Pregunta desde la terminal: make ask Q="..."
	docker compose exec app python -m rag.pipeline ask "$(Q)"

mac-setup:       ## Prepara el entorno nativo para transcribir con MLX (GPU del Mac)
	$(PYTHON) -m venv .venv-mac && .venv-mac/bin/pip install -r requirements-mac.txt

transcribe-mac:  ## Transcribe $(VIDEO) con MLX nativo; la app usará esa transcripción en caché
	.venv-mac/bin/python -m rag.transcribe data/videos/$(VIDEO) --engine mlx
