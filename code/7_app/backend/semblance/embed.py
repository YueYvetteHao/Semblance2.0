"""BioLORD embedder: exact-key cache + lazy model load.

Standalone port of the embedding half of ``Semblance/mcp_server/engine.py`` (its ``_embed``,
lines 54-64) — no MCP, no ``config`` module. The embedder is *callable* (``name -> np.ndarray``)
so it drops straight into ``core.signatures.build_signatures`` as its ``embed_fn``.

Pathway names already in the cache (seeded from a precomputed ``.npz``) never touch the model;
unknown names lazy-load BioLORD (~420 MB on first use) and embed the prettified name text. This
is what lets the common case — a Reactome GSEA whose pathway names are all precomputed — serve
with no model load at all.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

DEFAULT_MODEL = "FremyCompany/BioLORD-2023"
EMBED_DIM = 768


def prettify(name: str) -> str:
    """Pathway id -> embedding text, matching Semblance's engine (`_prettify`)."""
    return name.replace("_", " ").lower()


class BioLordEmbedder:
    """Callable embedder with an exact-key cache and a lazily loaded SentenceTransformer."""

    def __init__(
        self,
        cache: Optional[dict[str, np.ndarray]] = None,
        model_id: str = DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        self._cache: dict[str, np.ndarray] = cache if cache is not None else {}
        self._model = None
        self._model_id = model_id
        self._device = device

    # --- model (lazy) -------------------------------------------------------
    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_id, device=self._device)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return np.asarray(
            model.encode(texts, normalize_embeddings=True, batch_size=64), dtype=np.float32
        )

    # --- lookup -------------------------------------------------------------
    def __call__(self, name: str) -> np.ndarray:
        """Embed one pathway name (cached by exact key; L2-normalized)."""
        vec = self._cache.get(name)
        if vec is None:
            vec = self._encode([prettify(name)])[0]
            self._cache[name] = vec
        return vec

    def embed_text(self, text: str) -> np.ndarray:
        """Embed arbitrary free text with the (warm) model — not cached."""
        return self._encode([text])[0]

    def warm(self, names: Iterable[str]) -> None:
        """Batch-embed a list of pathway names into the cache (offline-build speedup)."""
        todo = [n for n in dict.fromkeys(names) if n not in self._cache]
        if not todo:
            return
        vecs = self._encode([prettify(n) for n in todo])
        for n, v in zip(todo, vecs):
            self._cache[n] = v

    @property
    def cache(self) -> dict[str, np.ndarray]:
        return self._cache

    def loaded(self) -> bool:
        return self._model is not None
