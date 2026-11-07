"""Property-based tests (Hypothesis): invariants that must hold for ANY input."""

from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

from findex.boolean import ENGINES
from findex.corpus import Document
from findex.index import build_index
from findex.store import load, save
from findex.tokenize import tokenize

words = st.sampled_from(["python", "loop", "event", "кіт", "café", "a", "the", "java"])
texts = st.lists(words, max_size=30).map(" ".join)
corpora = st.lists(texts, min_size=1, max_size=8).map(
    lambda ts: [Document(i, f"d{i}", f"T{i}", t) for i, t in enumerate(ts)]
)
sorted_ids = st.lists(st.integers(0, 200), max_size=60, unique=True).map(sorted)


@given(st.text())
def test_tokens_are_never_empty_and_idempotent(s: str) -> None:
    toks = list(tokenize(s))
    assert all(toks)
    assert list(tokenize(" ".join(toks))) == toks


@given(corpora, st.booleans())
def test_postings_are_always_sorted(docs: list[Document], positions: bool) -> None:
    ix = build_index(iter(docs), positions=positions)
    for plist in ix.postings.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(set(ids))  # sorted and no duplicates
    assert sum(p.tf for pl in ix.postings.values() for p in pl) == sum(ix.doc_lengths.values())


@given(sorted_ids, sorted_ids)
def test_merge_agrees_with_set(a: list[int], b: list[int]) -> None:
    for merge_op, set_op in zip(ENGINES["merge"], ENGINES["set"]):
        assert merge_op(a, b) == set_op(a, b)


@settings(max_examples=30)
@given(corpora, st.booleans(), st.sampled_from(["index.bin", "index.json"]))
def test_load_save_round_trip(docs: list[Document], positions: bool, name: str) -> None:
    ix = build_index(iter(docs), positions=positions)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        save(ix, path)
        assert load(path) == ix
