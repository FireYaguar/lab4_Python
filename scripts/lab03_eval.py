"""Lab 3: ranking sanity checks + precision@5 (TF-IDF vs BM25) -> markdown for the README.

uv run python scripts/lab03_eval.py index.bin      # index built with --positions
"""

from __future__ import annotations

import sys
from pathlib import Path

from findex.index import Index, Posting
from findex.scoring import BM25, TfIdf
from findex.search import search
from findex.store import open_index

# Hand-labeled BEFORE looking at results: "these should be in the top 5".
# Paths relative to data/python-docs/.
QRELS: dict[str, set[str]] = {
    '"event loop"': {
        "library/asyncio-eventloop.rst",
        "library/asyncio-dev.rst",
        "howto/a-conceptual-overview-of-asyncio.rst",
        "library/asyncio-runner.rst",
        "library/asyncio-llapi-index.rst",
    },
    '"regular expression"': {"library/re.rst", "howto/regex.rst"},
    "unicode normalization": {"library/unicodedata.rst", "howto/unicode.rst"},
    '"garbage collector"': {
        "library/gc.rst",
        "c-api/gcsupport.rst",
        "library/weakref.rst",
        "reference/datamodel.rst",
    },
    '"context manager"': {
        "library/contextlib.rst",
        "reference/datamodel.rst",
        "reference/compound_stmts.rst",
    },
    "thread lock": {"library/threading.rst", "library/_thread.rst", "library/asyncio-sync.rst"},
    "json decode": {"library/json.rst"},
    "pickle unpickling security": {
        "library/pickle.rst",
        "library/pickletools.rst",
        "library/copyreg.rst",
        "library/shelve.rst",
    },
    "decorator wraps": {"library/functools.rst"},
    '"virtual environment"': {
        "library/venv.rst",
        "tutorial/venv.rst",
        "installing/index.rst",
        "library/ensurepip.rst",
    },
}


def rel(path: str) -> str:
    return path.split("python-docs/", 1)[-1]


def sanity(ix: Index) -> None:
    bm, tf = BM25(), TfIdf()
    # 1. rare term beats common term: docs that contain ONLY one of the two terms
    common, rare = "class", "winsound"
    res = search(ix, f"{common} OR {rare}", bm, k=ix.num_docs)
    print(
        f"### 1. Rare vs common: `{common} OR {rare}` "
        f"(df {ix.df(common)} vs {ix.df(rare)}), {len(res)} matches\n"
    )
    print("| rank | BM25 | document | contains |\n|---|---|---|---|")
    c_ids, r_ids = set(ix.doc_ids(common)), set(ix.doc_ids(rare))
    first_common_only = next(i for i, r in enumerate(res) if r.doc_id not in r_ids)
    for i, r in enumerate(res):
        if i < 4 or i == first_common_only:
            has = [t for t, ids in ((common, c_ids), (rare, r_ids)) if r.doc_id in ids]
            print(f"| {i + 1} | {r.score:.2f} | `{rel(r.path)}` | {', '.join(has)} |")
    # 2. tf saturation (synthetic posting in a doc of average length)
    d = min(ix.doc_lengths, key=lambda x: abs(ix.doc_lengths[x] - ix.avg_doc_length))
    term = "loop"
    print(
        f"\n### 2. Saturation: term `{term}` in a doc of avg length ({ix.doc_lengths[d]} tokens)\n"
    )
    print("| tf | TF-IDF | BM25 |\n|---|---|---|")
    for n in (1, 2, 5, 10, 19, 20):
        print(f"| {n} | {tf(term, Posting(d, n), ix):.2f} | {bm(term, Posting(d, n), ix):.3f} |")
    g_tf = tf(term, Posting(d, 20), ix) - tf(term, Posting(d, 19), ix)
    g_bm = bm(term, Posting(d, 20), ix) - bm(term, Posting(d, 19), ix)
    g1 = bm(term, Posting(d, 1), ix)
    print(
        f"\n20th occurrence adds: TF-IDF +{g_tf:.2f} (same as the 1st), "
        f"BM25 +{g_bm:.3f} ({g_bm / g1:.1%} of the 1st occurrence)"
    )
    # 3. short doc with one match vs very long doc with one match
    best = None
    for t, plist in ix.items():
        ones = [p for p in plist if p.tf == 1]
        if len(ones) >= 2:
            s = min(ones, key=lambda p: ix.doc_lengths[p.doc_id])
            lg = max(ones, key=lambda p: ix.doc_lengths[p.doc_id])
            ratio = ix.doc_lengths[lg.doc_id] / ix.doc_lengths[s.doc_id]
            if ix.df(t) <= 5 and (best is None or ratio > best[0]):
                best = (ratio, t, s, lg)
    assert best is not None
    _, t, s, lg = best
    print(f"\n### 3. Short vs long, both tf=1, term `{t}`\n")
    print("| document | length | TF-IDF | BM25 |\n|---|---|---|---|")
    for p in (s, lg):
        print(
            f"| `{rel(ix.doc_meta[p.doc_id].path)}` | {ix.doc_lengths[p.doc_id]:,} "
            f"| {tf(t, p, ix):.2f} | {bm(t, p, ix):.2f} |"
        )


def precision(ix: Index) -> None:
    print("\n### precision@5\n")
    print("| Query | relevant | max P@5 | TF-IDF | BM25 |\n|---|---|---|---|---|")
    sums = [0.0, 0.0, 0.0]
    for q, relevant in QRELS.items():
        row = []
        for scorer in (TfIdf(), BM25()):
            top = [rel(r.path) for r in search(ix, q, scorer, k=5)]
            row.append(sum(p in relevant for p in top) / 5)
        mx = min(len(relevant), 5) / 5
        sums = [sums[0] + mx, sums[1] + row[0], sums[2] + row[1]]
        print(f"| `{q}` | {len(relevant)} | {mx:.1f} | {row[0]:.1f} | {row[1]:.1f} |")
    n = len(QRELS)
    print(f"| **mean** | | **{sums[0] / n:.2f}** | **{sums[1] / n:.2f}** | **{sums[2] / n:.2f}** |")


if __name__ == "__main__":
    with open_index(Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin")) as ix:
        sanity(ix)
        precision(ix)
