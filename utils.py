#!/usr/bin/env python3
"""Shared utilities for embeddings and similarity.

Used by memory.py and tasks.py to avoid duplicating model initialization.
"""

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding  # lazy: only loaded when embeddings are needed
        _embedding_model = TextEmbedding()
    return _embedding_model


def embed_text(text: str) -> list[float]:
    """Generate a normalized embedding vector for the given text."""
    model = get_embedding_model()
    vec = list(model.embed([text]))[0]
    vec = [float(x) for x in vec]
    norm = sum(x ** 2 for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def embed_texts_batch(texts: list) -> list:
    """Generate normalized embedding vectors for a list of texts in one batch.

    Uses fastembed's native batch inference — significantly faster than calling
    embed_text() in a loop because the ONNX model processes the full batch at
    once.  Returns a list of float lists, one per input text, in input order.
    """
    if not texts:
        return []
    model = get_embedding_model()
    result = []
    for vec in model.embed(texts):
        vec = [float(x) for x in vec]
        norm = sum(x ** 2 for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        result.append(vec)
    return result


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two pre-normalized vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x ** 2 for x in a) ** 0.5
    nb = sum(x ** 2 for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_cross_encoder = None
_cross_encoder_tried = False


def get_cross_encoder():
    """Lazy-load mixedbread-ai/mxbai-rerank-xsmall-v1 for Phase 4 reranking.

    Returns None silently if sentence-transformers is not installed (~142MB
    model, CPU-only). Total memory with fastembed: ~252MB.
    """
    global _cross_encoder, _cross_encoder_tried
    if not _cross_encoder_tried:
        _cross_encoder_tried = True
        try:
            from sentence_transformers import CrossEncoder  # lazy import
            _cross_encoder = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")
        except Exception:
            pass
    return _cross_encoder
