from pathlib import Path

import pytest

from findex.index import DocMeta, Index, Posting
from findex.query import parse
from findex.store import load, open_index, save


def test_postings_sorted_with_tf_and_positions(ix: Index) -> None:
    assert ix["python"] == [Posting(0, 1, (4,)), Posting(1, 1, (3,)), Posting(3, 1, (2,))]
    assert ix["java"] == [Posting(2, 2, (0, 4))]
    for plist in ix.postings.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(ids)
    assert ix.doc_lengths[0] == 6 and ix.doc_meta[3] == DocMeta(3, "d.txt", "D")


def test_records_are_hashable_and_frozen() -> None:
    assert len({Posting(1, 2), Posting(1, 2), DocMeta(1, "p", "t")}) == 2
    with pytest.raises(AttributeError):
        Posting(1, 2).tf = 5  # type: ignore[misc]  # frozen on purpose: this must raise


def test_index_dunders(ix: Index) -> None:
    assert "python" in ix and "rust" not in ix
    with pytest.raises(KeyError):
        ix["rust"]
    assert "event" in list(ix) and len(ix) == len(ix.postings)
    assert repr(ix).startswith("Index(terms=")
    assert ix.num_docs == 4 and ix.df("python") == 3
    assert ix.avg_doc_length == sum(ix.doc_lengths.values()) / 4
    assert dict(ix.items()) == ix.postings  # Mapping mixin methods for free


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("python", [0, 1, 3]),
        ("python java", []),
        ("java OR async", [2, 3]),
        ("python NOT async", [0, 1]),
        ("NOT python", [2]),
        ("Python, LOOP!", [0, 1]),
        ("missing", []),
    ],
)
def test_boolean_queries(ix: Index, query: str, expected: list[int]) -> None:
    assert parse(query).evaluate(ix) == expected


@pytest.mark.parametrize("name", ["index.bin", "index.json"])
def test_save_load_round_trip(ix: Index, tmp_path: Path, name: str) -> None:
    save(ix, tmp_path / name)
    assert load(tmp_path / name) == ix


def test_open_index_closes_on_exception(index_file: Path) -> None:
    with pytest.raises(RuntimeError), open_index(index_file) as loaded:
        loaded.match("python")
        assert loaded.match.cache_info().currsize == 1
        raise RuntimeError("boom")
    assert loaded.match.cache_info().currsize == 0  # close() ran
