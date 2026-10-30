"""Reordena localmente los fragmentos por relevancia respecto a la pregunta."""


class CrossEncoderReranker:
    """Cross-encoder local; carga el modelo solo al primer uso."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, question: str, hits: list[dict]) -> list[dict]:
        if len(hits) < 2:
            return hits
        scores = self._load().predict([(question, hit["text"]) for hit in hits])
        ranked = []
        for hit, score in zip(hits, scores):
            ranked.append({**hit, "rerank_score": float(score)})
        return sorted(ranked, key=lambda hit: hit["rerank_score"], reverse=True)
