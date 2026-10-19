"""FinBERT sentiment utilities.

Refactor highlights
-------------------
* **Lazy, thread-safe model loading** – importing this module no longer
  blocks on downloading/loading the transformer. The model is only built on
  the first inference, guarded by a lock (double-checked locking).
* **Batched inference** – :func:`estimate_sentiments_batch` scores many
  headlines in a single forward pass instead of one call per text.
* **TTL cache** – identical headline text is never re-inferred; scores are
  reused for ``SENTIMENT_TTL`` seconds.
* **Never raises** – inference failures degrade gracefully to a neutral
  score so a bad headline can never crash the trading loop.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import torch # pyright: ignore[reportMissingImports]
from transformers import AutoModelForSequenceClassification, AutoTokenizer # pyright: ignore[reportMissingImports]

from cache_utils import TTLCache

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
LABELS = ("positive", "negative", "neutral")

# How long a single headline's score stays cached (seconds).
SENTIMENT_TTL = 300.0

_NEUTRAL: Tuple[float, str] = (0.0, "neutral")

# Lazily-initialised, process-wide singleton (shared by all threads).
_model_lock = threading.Lock()
_tokenizer = None
_model = None

# Keyed by the exact headline text -> (score, label).
_score_cache = TTLCache(ttl=SENTIMENT_TTL, maxsize=4096)


def _ensure_model():
    """Build the tokenizer + model once, in a thread-safe way."""
    global _tokenizer, _model
    if _model is None:
        with _model_lock:
            if _model is None:  # double-checked locking
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
                model.eval()
                _tokenizer, _model = tokenizer, model
    return _tokenizer, _model


def warm_up() -> None:
    """Pre-load the model so the first real inference is fast. Never raises."""
    try:
        _ensure_model()
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("FinBERT warm-up failed: %s", exc)


@torch.no_grad()
def _infer(texts: Sequence[str]) -> List[Tuple[float, str]]:
    tokenizer, model = _ensure_model()
    inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    values, indices = torch.max(probs, dim=1)
    return [(float(v), LABELS[int(i)]) for v, i in zip(values, indices)]


def estimate_sentiments_batch(headlines: Sequence[str]) -> List[Tuple[float, str]]:
    """Score a batch of headlines, using the cache for already-seen text.

    Returns a list of ``(confidence, label)`` tuples in the same order as the
    input. Empty/blank entries and any failed inference map to neutral.
    """
    if not headlines:
        return []

    results: List[Optional[Tuple[float, str]]] = [None] * len(headlines)
    # Map unique uncached text -> list of positions it appears at, so the
    # same headline is inferred only once even if it repeats in the batch.
    misses: Dict[str, List[int]] = {}

    for i, raw in enumerate(headlines):
        text = (raw or "").strip()
        if not text:
            results[i] = _NEUTRAL
            continue
        cached = _score_cache.get(text)
        if cached is not None:
            results[i] = cached
        else:
            misses.setdefault(text, []).append(i)

    if misses:
        unique_texts = list(misses)
        try:
            inferred = _infer(unique_texts)
            for text, value in zip(unique_texts, inferred):
                _score_cache.set(text, value)
                for idx in misses[text]:
                    results[idx] = value
        except Exception as exc:
            logger.error("FinBERT inference failed: %s", exc)
            for indices in misses.values():
                for idx in indices:
                    results[idx] = _NEUTRAL

    return [r if r is not None else _NEUTRAL for r in results]


def estimate_sentiment(headlines) -> Tuple[float, str]:
    """Backwards-compatible single-item API.

    ``headlines`` is the (already joined) headline string for one symbol.
    """
    if not headlines:
        return _NEUTRAL
    return estimate_sentiments_batch([str(headlines)])[0]
