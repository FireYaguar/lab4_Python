from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import Index, Posting, build_index
from findex.query import And, Not, Or, Phrase, QueryError, Term, parse
from findex.scoring import BM25, TfIdf
from findex.search import SearchResult, search, snippet
from findex.store import open_index, save
from findex.timing import timed

DOCS = [
    Document(0, "a", "A", "the event loop runs python coroutines"),
    Document(1, "b", "B", "loop event the python"),
    Document(2, "c", "C", "java threads and the java loop"),
    Document(3, "d", "D", "async await python " + "filler " * 50),
]


@pytest.fixture
def ix() -> Index:
    return build_index(iter(DOCS), positions=True)


def test_index_dunders(ix: Index) -> None:
    assert len(ix) == len(set(" ".join(d.text for d in DOCS).split()))
    assert "python" in ix and "rust" not in ix
    assert ix["java"] == [Posting(2, 2, (0, 4))]
    with pytest.raises(KeyError):
        ix["rust"]
    assert "event" in list(ix)
    assert repr(ix).startswith("Index(terms=")
    assert ix.num_docs == 4 and ix.df("python") == 3
    assert ix.avg_doc_length == sum(ix.doc_lengths.values()) / 4
    assert dict(ix.items()) == ix.postings  # Mapping mixin methods for free


@pytest.mark.parametrize(
    ("text", "tree"),
    [
        ("a OR b c", Or(Term("a"), And(Term("b"), Term("c")))),
        ("a AND b", And(Term("a"), Term("b"))),
        ("a NOT b", And(Term("a"), Not(Term("b")))),
        ('"Event Loop"', Phrase(("event", "loop"))),
        ("(a OR b) c", And(Or(Term("a"), Term("b")), Term("c"))),
        ("well-known", And(Term("well"), Term("known"))),
        (
            'python AND (async OR await) NOT java "event loop"',
            And(
                And(And(Term("python"), Or(Term("async"), Term("await"))), Not(Term("java"))),
                Phrase(("event", "loop")),
            ),
        ),
    ],
)
def test_parse_trees(text: str, tree: object) -> None:
    assert parse(text) == tree


def test_operators_build_same_tree() -> None:
    assert parse("a OR b NOT c") == Term("a") | (Term("b") & ~Term("c"))


@pytest.mark.parametrize("bad", ["(a OR b", "a OR", '"open', "...", ")"])
def test_parse_errors(bad: str) -> None:
    with pytest.raises(QueryError):
        parse(bad)


def test_phrase_needs_adjacent_positions(ix: Index) -> None:
    assert parse('"event loop"').evaluate(ix) == [0]  # doc 1 has "loop event"
    assert parse("event loop").evaluate(ix) == [0, 1]
    assert parse("python AND (async OR await) NOT java").evaluate(ix) == [3]


def test_scorers_are_interchangeable(ix: Index) -> None:
    for scorer in (TfIdf(), BM25()):
        res = search(ix, "python OR java", scorer, k=2)
        assert len(res) == 2 and res[0].score >= res[1].score
        assert res[0].doc_id == 2  # java: rarer term and tf=2


def test_bm25_saturation_and_length() -> None:
    long_doc = Index({}, {0: 10, 1: 1000}, {}, False)
    long_doc.postings = {"t": [Posting(0, 1), Posting(1, 1)]}
    bm = BM25()
    assert bm("t", Posting(0, 1), long_doc) > bm("t", Posting(1, 1), long_doc)
    gain20 = bm("t", Posting(0, 20), long_doc) - bm("t", Posting(0, 19), long_doc)
    gain2 = bm("t", Posting(0, 2), long_doc) - bm("t", Posting(0, 1), long_doc)
    assert gain20 < gain2 / 20


def test_search_results_sorted_and_cached(ix: Index, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    r1 = search(ix, "python")
    r2 = search(ix, "python")
    assert r1 == r2 and sorted(r1, reverse=True) == r1
    assert ix.match.cache_info().hits == 1
    assert "cache hit" in caplog.text and "search took" in caplog.text
    assert SearchResult(2.0, 0, "", "") > SearchResult(1.0, 9, "", "")


def test_timed_keeps_metadata() -> None:
    @timed
    def f() -> int:
        """Doc."""
        return 1

    assert (f.__name__, f.__doc__, f()) == ("f", "Doc.", 1)


def test_snippet_highlights() -> None:
    text = "x " * 100 + "the Event loop is here" + " y" * 100
    s = snippet(text, {"event", "loop"})
    assert "**Event** **loop**" in s and s.startswith("…") and s.endswith("…")
    assert len(s) < 200


def test_open_index_closes_on_exception(ix: Index, tmp_path: Path) -> None:
    save(ix, tmp_path / "ix.bin")
    with pytest.raises(RuntimeError), open_index(tmp_path / "ix.bin") as loaded:
        loaded.match("python")
        assert loaded.match.cache_info().currsize == 1
        raise RuntimeError("boom")
    assert loaded.match.cache_info().currsize == 0  # close() ran
