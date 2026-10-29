"""Etapa 1 · Transcripción: video o audio -> segmentos con marcas de tiempo.

Cada segmento es un dict: {"start": 12.4, "end": 17.9, "text": "..."}.
Las marcas de tiempo son lo que después nos permite citar "el minuto exacto".

Tres motores intercambiables (variable TRANSCRIBER):
  - faster-whisper : CPU, funciona dentro de Docker.
  - mlx            : GPU de Apple Silicon, solo nativo en el Mac (no en Docker).
  - openai         : API compatible con OpenAI (OpenAI, Groq, ...).

Uso por línea de comandos (útil para transcribir nativo en el Mac con MLX):
    python -m rag.transcribe data/videos/conversacion.mp3 --engine mlx
"""

import argparse
import json
import time
from pathlib import Path

from .config import Config, load_config

# Nombres de modelo de cada motor para "large-v3-turbo"
MLX_REPOS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


def transcript_path(cfg: Config, video: Path) -> Path:
    return cfg.transcripts_dir / f"{video.stem}.json"


def load_transcript(cfg: Config, video: Path) -> dict | None:
    """Devuelve la transcripción guardada en caché, si existe."""
    path = transcript_path(cfg, video)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def transcribe(cfg: Config, video: Path, engine: str | None = None) -> dict:
    """Transcribe un archivo de audio o video y guarda el resultado en caché."""
    engine = engine or cfg.transcriber
    t0 = time.time()

    if engine == "faster-whisper":
        segments = _faster_whisper(cfg, video)
    elif engine == "mlx":
        segments = _mlx(cfg, video)
    elif engine == "openai":
        segments = _openai_api(cfg, video)
    else:
        raise ValueError(f"Motor de transcripción desconocido: {engine}")

    result = {
        "video": video.name,
        "engine": engine,
        "model": cfg.whisper_model,
        "seconds": round(time.time() - t0, 1),
        "segments": segments,
    }
    transcript_path(cfg, video).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _clean(segments) -> list[dict]:
    return [
        {"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2), "text": s["text"].strip()}
        for s in segments
        if s["text"].strip()
    ]


def _faster_whisper(cfg: Config, video: Path) -> list[dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(cfg.whisper_model, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(video), language=cfg.whisper_language, vad_filter=True)
    # `segments` es un generador: la transcripción ocurre mientras lo recorremos
    return _clean({"start": s.start, "end": s.end, "text": s.text} for s in segments)


def _mlx(cfg: Config, video: Path) -> list[dict]:
    import mlx_whisper  # solo existe en macOS con Apple Silicon

    repo = MLX_REPOS.get(cfg.whisper_model, cfg.whisper_model)
    out = mlx_whisper.transcribe(str(video), path_or_hf_repo=repo, language=cfg.whisper_language)
    return _clean(out["segments"])


def _openai_api(cfg: Config, video: Path) -> list[dict]:
    from openai import OpenAI

    # Las APIs limitan el tamaño del archivo (~25 MB): enviamos directamente
    # los archivos de audio y extraemos solo el audio de los videos.
    audio = video if video.suffix.lower() in AUDIO_EXT else extract_audio(
        video, cfg.transcripts_dir / f"{video.stem}.wav"
    )
    client = OpenAI(base_url=cfg.transcribe_base_url, api_key=cfg.transcribe_api_key)
    with open(audio, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=cfg.whisper_model,  # p. ej. "whisper-1" (OpenAI) o "whisper-large-v3-turbo" (Groq)
            file=f,
            language=cfg.whisper_language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    segments = [s if isinstance(s, dict) else s.model_dump() for s in resp.segments]
    return _clean(segments)


def extract_audio(video: Path, out: Path) -> Path:
    """Extrae el audio a WAV mono 16 kHz usando PyAV (viene con faster-whisper)."""
    import av

    with av.open(str(video)) as src, av.open(str(out), "w") as dst:
        stream = dst.add_stream("pcm_s16le", rate=16000, layout="mono")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in src.decode(audio=0):
            for f in resampler.resample(frame):
                for packet in stream.encode(f):
                    dst.mux(packet)
        for packet in stream.encode(None):
            dst.mux(packet)
    return out


def format_ts(seconds: float) -> str:
    """125.3 -> '02:05'"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio o video con Whisper")
    parser.add_argument("source", type=Path)
    parser.add_argument("--engine", choices=["faster-whisper", "mlx", "openai"])
    args = parser.parse_args()

    cfg = load_config()
    res = transcribe(cfg, args.source, args.engine)
    print(f"{len(res['segments'])} segmentos en {res['seconds']} s con {res['engine']}")
    for seg in res["segments"][:5]:
        print(f"  [{format_ts(seg['start'])}] {seg['text']}")
