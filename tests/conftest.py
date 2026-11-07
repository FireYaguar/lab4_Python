"""Shared fixtures: a tiny in-memory corpus and indexes built from it (no data/ needed)."""

from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import Index, build_index
from findex.store import save

TINY_DOCS = [
    Document(0, "a.txt", "A", "the event loop runs python coroutines"),
    Document(1, "b.txt", "B", "loop event the python"),
    Document(2, "c.txt", "C", "java threads and the java loop"),
    Document(3, "d.txt", "D", "async await python " + "filler " * 50),
]


@pytest.fixture
def ix() -> Index:
    return build_index(iter(TINY_DOCS), positions=True)


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    for d in TINY_DOCS:
        (root / d.path).write_text(f"{d.title}\n{d.text}", encoding="utf-8")
    return root


@pytest.fixture
def index_file(ix: Index, tmp_path: Path) -> Path:
    path = tmp_path / "index.bin"
    save(ix, path)
    return path
