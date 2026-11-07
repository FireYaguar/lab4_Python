"""Corpus statistics in one streaming pass (lazy) and the deliberately eager version (Lab 1)."""

from __future__ import annotations

import time
import tracemalloc
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from findex.corpus import Document
from findex.tokenize import tokenize


@dataclass(slots=True)
class CorpusStats:
    documents: int
    tokens: int
    counts: Counter[str]

    @property
    def vocabulary(self) -> int:
        return len(self.counts)


def compute_stats(docs: Iterable[Document]) -> CorpusStats:
    """Lazy: one document in memory at a time; only the Counter (the sink) grows."""
    counts: Counter[str] = Counter()
    n_docs = 0
    for doc in docs:
        n_docs += 1
        counts.update(tokenize(doc.text))
    return CorpusStats(n_docs, counts.total(), counts)


def compute_stats_eager(docs: Iterable[Document]) -> CorpusStats:
    """Eager on purpose: all documents in a list, all tokens in lists of lists."""
    all_docs = list(docs)
    all_tokens = [list(tokenize(d.text)) for d in all_docs]
    counts: Counter[str] = Counter()
    for toks in all_tokens:
        counts.update(toks)
    return CorpusStats(len(all_docs), sum(len(t) for t in all_tokens), counts)


def measure[T](fn: Callable[[], T]) -> tuple[T, float, int]:
    """Run fn under tracemalloc; return (result, elapsed seconds, peak bytes)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, elapsed, peak
