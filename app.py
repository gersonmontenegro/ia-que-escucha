"""App de demostración (Gradio): ¿Cómo enseñarle a una IA a escuchar?

Pestaña 1 · Ingesta:   transcribe e indexa un video, mostrando cada etapa.
Pestaña 2 · Preguntar: muestra los fragmentos recuperados ANTES de la respuesta,
                       la respuesta con citas, y un reproductor que salta al minuto.
"""

from pathlib import Path
import time
from urllib.parse import quote

import gradio as gr

from rag.config import load_config
from rag.llm import cited_numbers
from rag.pipeline import RAG
from rag.transcribe import format_ts, load_transcript

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}

cfg = load_config()
rag = RAG(cfg)


# ----------------------------------------------------------------- utilidades
def list_videos() -> list[str]:
    return sorted(p.name for p in cfg.videos_dir.iterdir() if p.suffix.lower() in VIDEO_EXT)


def player_html(video: str | None, start: float = 0, autoplay: bool = False) -> str:
    if not video:
        return "<div class='vacio'>Aquí aparecerá el video</div>"
    path = (cfg.videos_dir / video).resolve()
    # #t=SEGUNDOS es un "media fragment": el navegador abre el video en ese instante
    src = f"/gradio_api/file={quote(str(path))}#t={int(start)}"
    auto = "autoplay" if autoplay else ""
    return (
        f"<video id='rag-video-player' src='{src}' controls playsinline {auto} "
        "preload='metadata' style='display:block;width:100%;aspect-ratio:16/9;"
        "object-fit:contain;background:#000;border-radius:8px'></video>"
    )


def hits_table(hits: list[dict]) -> list[list]:
    return [[n, round(h["score"], 3), f"{format_ts(h['start'])}–{format_ts(h['end'])}", h["text"]]
            for n, h in enumerate(hits, start=1)]


def citation_choices(hits: list[dict], answer: str) -> list[tuple[str, int]]:
    numbers = [n for n in cited_numbers(answer) if 1 <= n <= len(hits)] or list(range(1, len(hits) + 1))
    return [(f"[{n}] {format_ts(hits[n - 1]['start'])} · {hits[n - 1]['video']}", n) for n in numbers]


def config_md() -> str:
    return (
        f"**LLM:** `{cfg.llm_model}` @ `{cfg.llm_base_url}`  \n"
        f"**Embeddings:** `{cfg.embed_model}` @ `{cfg.embed_base_url}`  \n"
        f"**Almacén vectorial:** `{cfg.vector_store}` · **top-k:** {cfg.top_k}  \n"
        f"**Whisper:** `{cfg.transcriber}` / `{cfg.whisper_model}` · **chunk:** {cfg.chunk_max_chars} car."
    )


def generation_note(seconds: float, llm) -> str:
    tokens = getattr(llm, "last_completion_tokens", None)
    if tokens is None:
        return f"*⏱️ Generado en {seconds:.1f} s · el proveedor no informó tokens*"
    tokens_per_second = tokens / seconds if seconds > 0 else 0
    return (
        f"*⏱️ Generado en {seconds:.1f} s · {tokens} tokens · "
        f"{tokens_per_second:.1f} tokens/s promedio (incluye la espera inicial)*"
    )


# ----------------------------------------------------------------- acciones
def do_ingest(video: str, force: bool):
    if not video:
        yield "Elige un video de data/videos/", None
        return
    log: list[str] = []
    for msg in rag.ingest(cfg.videos_dir / video, force):
        log.append(msg)
        yield "\n".join(log), None
    data = load_transcript(cfg, cfg.videos_dir / video)
    rows = [[format_ts(s["start"]), s["text"]] for s in data["segments"]]
    yield "\n".join(log), rows


def do_ask(question: str, compare: bool):
    """Generador: primero muestra lo recuperado, luego la respuesta en streaming."""
    empty_radio = gr.Radio(choices=[], value=None)
    if not question.strip():
        yield "", "", [], [], empty_radio, gr.skip(), ""
        return

    retrieval_start = time.perf_counter()
    hits = rag.retrieve(question)
    retrieval_seconds = time.perf_counter() - retrieval_start
    retrieval_status = f"⏱️ Recuperación: {retrieval_seconds:.3f} s · {len(hits)} fragmentos"
    if not hits:
        yield "⚠️ No hay nada indexado todavía. Ve a la pestaña de ingesta.", "", [], [], empty_radio, gr.skip(), retrieval_status
        return

    # 1) Lo recuperado aparece primero: es el momento clave para entender el RAG
    yield "⏳ Generando…", "", hits_table(hits), hits, empty_radio, gr.skip(), retrieval_status

    answer = ""
    generation_start = time.perf_counter()
    for answer in rag.answer(question, hits):
        yield answer, "", hits_table(hits), hits, empty_radio, gr.skip(), retrieval_status
    answer_seconds = time.perf_counter() - generation_start
    answer_display = f"{answer}\n\n{generation_note(answer_seconds, rag.llm)}"

    choices = citation_choices(hits, answer)
    first = hits[choices[0][1] - 1]
    radio = gr.Radio(choices=choices, value=None)
    player = player_html(first["video"], first["start"])

    # 2) Opcional: la misma pregunta SIN contexto, para comparar (y verlo alucinar)
    no_rag = ""
    no_rag_display = ""
    if compare:
        no_rag_start = time.perf_counter()
        for no_rag in rag.answer(question, None):
            yield answer_display, no_rag, hits_table(hits), hits, radio, player, retrieval_status
        no_rag_seconds = time.perf_counter() - no_rag_start
        no_rag_display = f"{no_rag}\n\n{generation_note(no_rag_seconds, rag.llm)}"
    yield answer_display, no_rag_display, hits_table(hits), hits, radio, player, retrieval_status


def jump_to(n: int | None, hits: list[dict]):
    """Crea el reproductor en el minuto citado tras detener el anterior."""
    if not n or not hits:
        return gr.skip()
    hit = hits[n - 1]
    return player_html(hit["video"], hit["start"], autoplay=True)


STOP_PLAYER_JS = """(question, compare) => {
    const player = document.getElementById("rag-video-player");
    if (player) player.pause();
    return [question, compare];
}"""

SEEK_PLAYER_JS = """(n, hits) => {
    if (!n || !hits || !hits[n - 1]) return [n, hits];
    const player = document.getElementById("rag-video-player");
    if (player) {
        player.pause();
        player.removeAttribute("src");
        player.load();
    }
    return [n, hits];
}"""


# ----------------------------------------------------------------- interfaz
CSS = """
.vacio {display:flex;align-items:center;justify-content:center;height:220px;
        border:1px dashed var(--border-color-primary);border-radius:8px;color:var(--body-text-color-subdued)}
"""

with gr.Blocks(title="¿Cómo enseñarle a una IA a escuchar?") as demo:
    gr.Markdown("# 🎧 ¿Cómo enseñarle a una IA a escuchar?\nRAG sobre video: Whisper → chunks → embeddings → búsqueda → LLM")
    with gr.Accordion("Configuración activa", open=False):
        gr.Markdown(config_md())

    with gr.Tab("1 · Ingesta"):
        with gr.Row():
            video_dd = gr.Dropdown(choices=list_videos(), label="Video (data/videos/)", scale=3)
            refresh = gr.Button("🔄", scale=0, min_width=50)
            force = gr.Checkbox(label="Ignorar caché de transcripción", scale=1)
        ingest_btn = gr.Button("Transcribir e indexar", variant="primary")
        ingest_log = gr.Textbox(label="Etapas", lines=6, interactive=False)
        transcript = gr.Dataframe(headers=["Minuto", "Texto"], label="Transcripción", wrap=True,
                                  interactive=False, column_widths=["90px", "auto"])

    with gr.Tab("2 · Preguntar"):
        hits_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(label="Pregunta", placeholder="¿Qué se descubrió y cómo?")
                with gr.Row():
                    compare = gr.Checkbox(label="Comparar con respuesta SIN RAG")
                    ask_btn = gr.Button("Preguntar", variant="primary")
                answer_md = gr.Markdown(label="Respuesta con RAG")
                no_rag_md = gr.Markdown(label="Respuesta sin RAG")
            with gr.Column(scale=2):
                player = gr.HTML(player_html(None), js_on_load=None)
                cites = gr.Radio(choices=[], label="Ir al minuto citado")
        retrieval_time = gr.Markdown()
        retrieved = gr.Dataframe(headers=["#", "Similitud", "Minuto", "Fragmento recuperado"],
                                 label="Fragmentos recuperados (lo que realmente ve el LLM)", wrap=True,
                                 interactive=False, column_widths=["40px", "90px", "110px", "auto"])

    # eventos
    refresh.click(lambda: gr.Dropdown(choices=list_videos()), outputs=video_dd)
    ingest_btn.click(do_ingest, [video_dd, force], [ingest_log, transcript])
    ask_outputs = [answer_md, no_rag_md, retrieved, hits_state, cites, player, retrieval_time]
    ask_btn.click(do_ask, [question, compare], ask_outputs, js=STOP_PLAYER_JS)
    question.submit(do_ask, [question, compare], ask_outputs, js=STOP_PLAYER_JS)
    cites.change(jump_to, [cites, hits_state], player, js=SEEK_PLAYER_JS)


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(cfg.videos_dir.resolve())],
        theme=gr.themes.Soft(),
        css=CSS,
    )
