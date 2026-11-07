"""Lazy corpus reader: yields one document at a time, never the whole corpus."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

TEXT_SUFFIXES = frozenset({".txt", ".rst", ".md"})


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: int
    path: str  # file path, or "file.jsonl#line" / URL for JSON-lines corpora
    title: str
    text: str


def _title_from(text: str, fallback: str) -> str:
    """First non-empty line that is not an rst directive/field or a decoration line."""
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line.startswith(("..", ":")):
            continue
        if line and not set(line) <= set("=-~*^#.:`'\"+_ "):
            return line[:120]
    return fallback


def _iter_jsonl(path: Path, start_id: int) -> Iterator[Document]:
    doc_id = start_id
    with path.open(encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):  # the file is an iterator of lines
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                text = str(rec["text"])
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                log.warning("skip %s:%d: %s", path, lineno, e)
                continue
            src = str(rec.get("url") or f"{path}#{lineno}")
            yield Document(doc_id, src, str(rec.get("title") or _title_from(text, src)), text)
            doc_id += 1


def count_documents(root: Path) -> int:
    """Cheap upper bound for a progress bar: text files + lines of .jsonl files."""
    paths = [root] if root.is_file() else root.rglob("*")
    n = 0
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("rb") as f:
                n += sum(1 for _ in f)
        elif suffix in TEXT_SUFFIXES and path.is_file():
            n += 1
    return n


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield documents from a directory tree (.txt/.rst/.md) and/or .jsonl files.

    Generator: opens one file at a time. Files that are not valid UTF-8 are read
    with errors="replace" and logged, so one bad file never stops a build.
    """
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    # sorted() materializes only the *paths* (small), so doc_ids are stable across runs.
    doc_id = 0
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            for doc in _iter_jsonl(path, doc_id):
                yield doc
                doc_id = doc.doc_id + 1
        elif suffix in TEXT_SUFFIXES and path.is_file():
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                log.warning("bad UTF-8 in %s, replacing undecodable bytes", path)
                text = raw.decode("utf-8", errors="replace")
            yield Document(doc_id, str(path), _title_from(text, path.stem), text)
            doc_id += 1
