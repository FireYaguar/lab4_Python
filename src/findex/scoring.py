"""Scorers. Any object with a matching __call__ satisfies the Scorer protocol —
no common base class (duck typing; Lab 4 checks it with pyright)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from findex.index import Index, Posting


class Scorer(Protocol):
    def __call__(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """tf · log(N / df): linear in tf, no length normalization."""

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return posting.tf * math.log(index.num_docs / index.df(term))


@dataclass(frozen=True, slots=True)
class BM25:
    """Okapi BM25. k1: how fast tf saturates. b: how much long documents are penalized
    (b = 0 -> no length normalization, b = 1 -> full)."""

    k1: float = 1.5
    b: float = 0.75

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        n, df = index.num_docs, index.df(term)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        norm = self.k1 * (
            1 - self.b + self.b * index.doc_length(posting.doc_id) / index.avg_doc_length
        )
        return idf * posting.tf * (self.k1 + 1) / (posting.tf + norm)


SCORERS: dict[str, Scorer] = {"bm25": BM25(), "tfidf": TfIdf()}
