import inspect

import pytest

from findex.corpus import iter_documents
from findex.tokenize import iter_spans, tokenize


def toks(s: str) -> list[str]:
    return list(tokenize(s))


def test_is_generator() -> None:
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python PYTHON python", ["python"] * 3),
        ("Straße STRASSE", ["strasse", "strasse"]),  # casefold, not lower
        ("Привіт, Світе! Їжак і ґанок", ["привіт", "світе", "їжак", "і", "ґанок"]),
        ("café", ["café"]),
        ("café", ["café"]),  # combining accent -> NFC
        ("Hello, world... well-known!", ["hello", "world", "well", "known"]),
        ("don't п'ять п’ять 'quoted'", ["don't", "п'ять", "п'ять", "quoted"]),
        ("Python 3.12 and 2026", ["python", "3", "12", "and", "2026"]),
        ("__init__ ______ snake_case", ["__init__", "snake_case"]),
        ("", []),
        ("   ...  —  ", []),
    ],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert toks(text) == expected


def test_spans_point_into_original_case() -> None:
    text = "Straße and Python"
    assert [(t, text[s:e]) for t, s, e in iter_spans(text)] == [
        ("strasse", "Straße"),
        ("and", "and"),
        ("python", "Python"),
    ]
