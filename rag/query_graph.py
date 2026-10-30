"""Flujo LangGraph de recuperación adaptativa y evaluación del contexto."""

import json
import re
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import build_messages


FALLBACK_ANSWER = "Eso no se dice en el video."


class QueryState(TypedDict, total=False):
    question: str
    query: str
    hits: list[dict]
    usable_context: bool
    retries: int
    answer: str
    suggested_query: str
    timings: dict[str, float]


def build_query_graph(rag):
    """Construye el grafo con dependencias RAG inyectadas para facilitar pruebas."""

    def retrieve(state: QueryState):
        started = time.perf_counter()
        hits = rag.retrieve(state["query"])
        retrieved_at = time.perf_counter()
        ranked = rag.reranker.rerank(state["query"], hits)
        timings = dict(state.get("timings", {}))
        timings["retrieval_seconds"] = timings.get("retrieval_seconds", 0) + retrieved_at - started
        timings["rerank_seconds"] = timings.get("rerank_seconds", 0) + time.perf_counter() - retrieved_at
        return {"hits": ranked, "timings": timings}

    def evaluate_context(state: QueryState):
        started = time.perf_counter()
        verdict = rag.llm.complete(
            _context_grade_messages(state["question"], state["query"], state["hits"]),
            temperature=0,
        )
        parsed = _parse_grade(verdict)
        timings = dict(state.get("timings", {}))
        timings["context_evaluation_seconds"] = (
            timings.get("context_evaluation_seconds", 0) + time.perf_counter() - started
        )
        return {
            "usable_context": bool(state["hits"]) and parsed["usable"],
            "suggested_query": parsed["rewritten_query"] or state["query"],
            "timings": timings,
        }

    def rewrite_query(state: QueryState):
        rewritten = state.get("suggested_query", "").strip().strip('"')
        return {
            "query": rewritten or state["query"],
            "retries": state["retries"] + 1,
        }

    def route_after_grade(state: QueryState):
        if state["usable_context"]:
            return "generate_answer"
        suggested = state.get("suggested_query", "").strip().casefold()
        current = state["query"].strip().casefold()
        if not suggested or suggested == current:
            return "finish_without_context"
        if state["retries"] >= rag.cfg.max_query_retries:
            return "finish_without_context"
        return "rewrite_query"

    def generate_answer(state: QueryState):
        started = time.perf_counter()
        answer = rag.llm.complete(build_messages(state["question"], state["hits"]))
        timings = dict(state.get("timings", {}))
        timings["generation_seconds"] = time.perf_counter() - started
        return {"answer": answer, "timings": timings}

    def finish_without_context(state: QueryState):
        return {"answer": FALLBACK_ANSWER, "hits": []}

    graph = StateGraph(QueryState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("evaluate_context", evaluate_context)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("finish_without_context", finish_without_context)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "evaluate_context")
    graph.add_conditional_edges(
        "evaluate_context",
        route_after_grade,
        {
            "generate_answer": "generate_answer",
            "finish_without_context": "finish_without_context",
            "rewrite_query": "rewrite_query",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate_answer", END)
    graph.add_edge("finish_without_context", END)
    return graph.compile()


def _context_grade_messages(question: str, current_query: str, hits: list[dict]) -> list[dict]:
    context = "\n\n".join(hit["text"] for hit in hits)
    return [
        {"role": "system", "content": (
            "Evalúa si el contexto contiene información suficiente para responder la pregunta. "
            'Devuelve únicamente JSON válido con este formato: '
            '{"usable": true, "rewritten_query": ""} o '
            '{"usable": false, "rewritten_query": "consulta reformulada"}. '
            "Si el contexto no basta, formula la consulta nueva en español y corrige errores "
            "ortográficos o palabras incompletas de la pregunta (por ejemplo, ‘jupite’ debe "
            "buscarse como ‘Júpiter’). Mantén la intención original, usa términos concretos "
            "que puedan aparecer en una transcripción y no repitas la consulta anterior. "
            "No uses conocimiento externo para decidir si el contexto basta."
        )},
        {"role": "user", "content": (
            f"Pregunta original: {question}\nConsulta usada: {current_query}\n\nContexto:\n{context}"
        )},
    ]


def _parse_grade(response: str) -> dict:
    """Acepta JSON directo o entre fences; ante una salida ambigua, falla cerrado."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)
    match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return {
                "usable": parsed.get("usable") is True,
                "rewritten_query": str(parsed.get("rewritten_query") or "").strip(),
            }
        except (json.JSONDecodeError, AttributeError):
            pass
    lowered = cleaned.lower()
    if lowered.startswith(("sí", "si", "yes")):
        return {"usable": True, "rewritten_query": ""}
    return {"usable": False, "rewritten_query": ""}
