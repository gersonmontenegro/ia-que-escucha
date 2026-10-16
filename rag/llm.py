"""Etapa 5 · Generación: pregunta + fragmentos recuperados -> respuesta con citas.

El "truco" del RAG está aquí: no le pedimos al modelo que recuerde, le damos
el texto relevante dentro del prompt y le pedimos que responda SOLO con eso,
citando cada fragmento como [1], [2]...
"""

import re
from collections.abc import Iterator

from openai import OpenAI

from .config import Config
from .transcribe import format_ts

SYSTEM_RAG = """Eres un asistente que responde preguntas sobre videos a partir de su transcripción.
Reglas:
- Responde SOLO con la información de los fragmentos que se te entregan.
- Cita cada afirmación con el número del fragmento entre corchetes, por ejemplo [1] o [2][3].
- Si los fragmentos no contienen la respuesta, di claramente que el video no lo menciona.
- Responde en español, de forma breve y directa."""

SYSTEM_SIN_RAG = """Eres un asistente útil. Responde en español, de forma breve y directa."""


def build_context(hits: list[dict]) -> str:
    """Convierte los fragmentos recuperados en el bloque de contexto del prompt."""
    blocks = []
    for n, h in enumerate(hits, start=1):
        blocks.append(f"[{n}] ({h['video']}, {format_ts(h['start'])}–{format_ts(h['end'])})\n{h['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[dict] | None) -> list[dict]:
    if hits is None:  # modo "sin RAG": el modelo solo tiene su memoria
        return [{"role": "system", "content": SYSTEM_SIN_RAG}, {"role": "user", "content": question}]
    user = f"Fragmentos de la transcripción:\n\n{build_context(hits)}\n\nPregunta: {question}"
    return [{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": user}]


class LLM:
    def __init__(self, cfg: Config):
        self.client = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key or "none")
        self.model = cfg.llm_model
        self.temperature = cfg.llm_temperature
        self.extra_body = cfg.llm_extra_body
        self.last_completion_tokens: int | None = None

    def complete(self, messages: list[dict], temperature: float | None = None) -> str:
        """Respuesta completa de una sola vez (sin streaming)."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            extra_body=self.extra_body or None,
        )
        return strip_thinking(resp.choices[0].message.content or "")

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Devuelve la respuesta acumulada a medida que llega (para mostrarla en vivo)."""
        self.last_completion_tokens = None
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=self.extra_body or None,
        )
        text = ""
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
                yield text
            if chunk.usage and chunk.usage.completion_tokens is not None:
                self.last_completion_tokens = chunk.usage.completion_tokens


def strip_thinking(text: str) -> str:
    """Quita el razonamiento <think>...</think> que algunos modelos mezclan en la respuesta."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL)  # bloque aún sin cerrar
    return text.strip()


def cited_numbers(answer: str) -> list[int]:
    """Números de fragmento citados en la respuesta, en orden de aparición: '[2][1]' -> [2, 1]."""
    seen: list[int] = []
    for m in re.finditer(r"\[(\d+)\]", answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen
