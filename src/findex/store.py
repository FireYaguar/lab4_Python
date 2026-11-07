"""Save / load the index. Two formats, chosen by file suffix.

pickle (default, e.g. index.bin)
    Fast and exact: frozen slotted dataclasses round-trip as they are.
    SECURITY: pickle.load() can execute arbitrary code — a pickle stream may tell the
    loader to call any importable function (os.system, ...) while rebuilding objects.
    Only ever load index files YOU built. Never load a pickle downloaded or received
    over the network.

json (index.json)
    Safe (pure data, no code) and readable, but bigger and slower. JSON has no tuples
    and only string keys, so postings are stored column-wise per term:
    {"doc_ids": [...], "tfs": [...], "positions": [[...], ...]} and doc ids as str keys.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from findex.index import DocMeta, Index, Posting
from findex.timing import timed


def save(index: Index, path: Path) -> None:
    if path.suffix == ".json":
        _save_json(index, path)
    else:
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


@timed
def load(path: Path) -> Index:
    if path.suffix == ".json":
        return _load_json(path)
    with path.open("rb") as f:
        obj = pickle.load(f)  # trusted, self-built file only — see module docstring
    if not isinstance(obj, Index):
        raise TypeError(f"{path} does not contain an Index")
    return obj


def _save_json(index: Index, path: Path) -> None:
    postings: dict[str, dict[str, list[Any]]] = {}
    for term, plist in index.postings.items():
        col: dict[str, list[Any]] = {
            "doc_ids": [p.doc_id for p in plist],
            "tfs": [p.tf for p in plist],
        }
        if index.has_positions:
            col["positions"] = [list(p.positions) for p in plist]
        postings[term] = col
    data = {
        "has_positions": index.has_positions,
        "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
        "doc_meta": {str(m.doc_id): [m.path, m.title] for m in index.doc_meta.values()},
        "postings": postings,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def _load_json(path: Path) -> Index:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    postings: dict[str, list[Posting]] = {}
    for term, col in data["postings"].items():
        pos = col.get("positions") or [()] * len(col["doc_ids"])
        postings[term] = [
            Posting(d, tf, tuple(ps)) for d, tf, ps in zip(col["doc_ids"], col["tfs"], pos)
        ]
    return Index(
        postings,
        {int(k): v for k, v in data["doc_lengths"].items()},
        {int(k): DocMeta(int(k), p, t) for k, (p, t) in data["doc_meta"].items()},
        bool(data["has_positions"]),
    )


@contextmanager
def open_index(path: Path) -> Iterator[Index]:
    """with open_index(p) as ix: ...  — close() runs even if the body raises or returns."""
    index = load(path)
    try:
        yield index
    finally:
        index.close()
