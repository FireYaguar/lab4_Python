"""findex — a small full-text search engine built step by step."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("findex")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"
