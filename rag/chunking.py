"""Etapa 2 · Chunking: segmentos de Whisper -> fragmentos ("chunks") para indexar.

Los segmentos de Whisper son muy cortos (unos segundos, media frase). Si
indexáramos cada uno por separado, cada vector tendría muy poco contexto.
Aquí los agrupamos en ventanas de hasta `max_chars` caracteres, conservando
el inicio del primer segmento y el final del último: así cada chunk sabe en
qué minuto empieza y termina.

El solapamiento (`overlap_segments`) repite los últimos segmentos de un chunk
al comienzo del siguiente, para que una idea partida en el borde no se pierda.
"""


def chunk_segments(segments: list[dict], max_chars: int = 700, overlap_segments: int = 1) -> list[dict]:
    chunks: list[dict] = []
    window: list[dict] = []

    for seg in segments:
        size = sum(len(s["text"]) + 1 for s in window)
        # Si el segmento ya no cabe, cerramos la ventana actual
        if window and size + len(seg["text"]) > max_chars:
            chunks.append(_make_chunk(window, len(chunks)))
            # Arrancamos la siguiente ventana con los últimos segmentos (solapamiento)
            window = window[-overlap_segments:] if overlap_segments > 0 else []
        window.append(seg)

    if window:
        chunks.append(_make_chunk(window, len(chunks)))
    return chunks


def _make_chunk(window: list[dict], index: int) -> dict:
    return {
        "id": index,
        "start": window[0]["start"],
        "end": window[-1]["end"],
        "text": " ".join(s["text"] for s in window),
    }
