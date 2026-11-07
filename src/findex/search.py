"""Ranked search.

    python -m findex.search index.bin 'python AND (async OR await) NOT java "event loop"'
        [--scorer bm25|tfidf] [-k 10] [--repeat 2] [-v]

1. parse the query into a tree and evaluate it -> matching doc ids (lru_cache per index)
2. score the matches with the query's positive terms (Scorer protocol)
3. top-k with heapq.nlargest, then a snippet with highlighted terms for each result
"""

from __future__ import annotations

import argparse
import heapq
import logging
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from findex.index import Index
from findex.query import QueryError, parse
from findex.scoring import BM25, SCORERS, Scorer
from findex.timing import timed
from findex.tokenize import iter_spans

log = logging.getLogger(__name__)
SNIPPET_RADIUS = 80


@dataclass(frozen=True, slots=True, order=True)
class SearchResult:
    score: float  # first field: sorted(results) orders by score
    doc_id: int
    title: str
    path: str
    snippet: str = ""


def match(index: Index, query: str) -> tuple[int, ...]:
    """Doc ids matching the Boolean query. Cached per index with lru_cache."""
    before = index.match.cache_info().hits
    ids = index.match(query)
    if index.match.cache_info().hits > before:
        log.debug("cache hit for %r", query)
    return ids


@timed
def search(
    index: Index, query: str, scorer: Scorer | None = None, k: int = 10
) -> list[SearchResult]:
    scorer = scorer or BM25()
    ids = match(index, query)
    if not ids:
        return []
    wanted = set(ids)
    terms = parse(query).terms()
    scores: defaultdict[int, float] = defaultdict(float)
    for d in ids:
        scores[d] = 0.0  # matched docs without positive terms (e.g. "NOT x") still appear
    for term in terms:
        for posting in index.postings.get(term, ()):
            if posting.doc_id in wanted:
                scores[posting.doc_id] += scorer(term, posting, index)
    top = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])  # O(n log k), not a full sort
    results: list[SearchResult] = []
    for doc_id, score in top:
        meta = index.doc_meta[doc_id]
        text = _read_text(meta.path)
        results.append(SearchResult(score, doc_id, meta.title, meta.path, snippet(text, terms)))
    return results


def _read_text(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""


def snippet(text: str, terms: set[str], radius: int = SNIPPET_RADIUS) -> str:
    """A 2·radius window (±radius around the best occurrence) with the most distinct query
    terms; query terms are highlighted as **term**. Sliding window: O(number of hits)."""
    if not text or not terms:
        return ""
    nfc = unicodedata.normalize("NFC", text)  # iter_spans offsets point into the NFC text
    hits = [(s, e, t) for t, s, e in iter_spans(text) if t in terms]
    if not hits:
        return ""
    window: Counter[str] = Counter()
    best_i, best_j, best_n, j = 0, 0, 0, 0
    for i, (s, _, _) in enumerate(hits):
        while j < len(hits) and hits[j][1] <= s + 2 * radius:
            window[hits[j][2]] += 1
            j += 1
        if len(window) > best_n:
            best_i, best_j, best_n = i, j, len(window)
        window[hits[i][2]] -= 1
        if not window[hits[i][2]]:
            del window[hits[i][2]]
    center = (hits[best_i][0] + hits[best_j - 1][1]) // 2
    lo, hi = max(0, center - radius), min(len(nfc), center + radius)
    out: list[str] = []
    cur = lo
    for s, e, _ in hits[best_i:best_j]:
        if s >= cur and e <= hi:
            out += [nfc[cur:s], "**", nfc[s:e], "**"]
            cur = e
    out.append(nfc[cur:hi])
    body = re.sub(r"\s+", " ", "".join(out)).strip()
    return ("…" if lo > 0 else "") + body + ("…" if hi < len(nfc) else "")


def main(argv: list[str] | None = None) -> None:
    from findex.stats import measure
    from findex.store import open_index

    ap = argparse.ArgumentParser(prog="python -m findex.search", description=__doc__)
    ap.add_argument("index", type=Path)
    ap.add_argument("query")
    ap.add_argument("--scorer", choices=sorted(SCORERS), default="bm25")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=1, help="run the query N times (shows cache)")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug log (@timed, cache)")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    t0 = time.perf_counter()
    with open_index(args.index) as index:
        print(f"load:   {time.perf_counter() - t0:.2f} s  {index!r}")
        try:
            results: list[SearchResult] = []
            for _ in range(args.repeat):
                results, t, peak = measure(
                    lambda: search(index, args.query, SCORERS[args.scorer], args.k)
                )
                print(f"search: {t * 1000:.2f} ms, peak memory {peak / 2**20:.1f} MiB")
        except QueryError as e:
            raise SystemExit(f"bad query: {e}") from None
        print(f"\n{len(match(index, args.query)):,} documents match; top {len(results)}:\n")
        for rank, r in enumerate(results, 1):
            print(f"{rank:>2}. {r.score:6.2f}  {r.title[:70]}  ({r.path})")
            if r.snippet:
                print(f"    {r.snippet}\n")


if __name__ == "__main__":
    main()
