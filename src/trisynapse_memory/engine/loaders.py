"""Local document loaders for text, HTML, PDF, and source code."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


CODE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".java",
    ".js", ".jsx", ".kt", ".lua", ".php", ".py", ".rb", ".rs", ".scala",
    ".sh", ".sql", ".swift", ".ts", ".tsx", ".vue", ".zig",
}
TEXT_SUFFIXES = {".txt", ".md", ".mdx", ".rst", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".csv"}


@dataclass(frozen=True)
class LoadedDocument:
    text: str
    title: str
    media_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class _TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored += 1
        elif tag in {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "li", "br", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored:
            self._ignored -= 1
        elif tag in {"p", "div", "section", "article", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            self.parts.append(data)


def load_document(path: str | Path) -> LoadedDocument:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    suffix = file_path.suffix.lower()
    media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    metadata = {"path": str(file_path), "size_bytes": file_path.stat().st_size, "suffix": suffix}
    if suffix == ".pdf":
        return _load_pdf(file_path, metadata)
    if suffix in {".html", ".htm"}:
        parser = _TextHTMLParser()
        parser.feed(file_path.read_text(encoding="utf-8", errors="replace"))
        text = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
        return LoadedDocument(text=text, title=file_path.name, media_type="text/html", metadata=metadata)
    if suffix in CODE_SUFFIXES:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        numbered = "\n".join(f"{index:06d}: {line}" for index, line in enumerate(raw.splitlines(), 1))
        metadata["language"] = suffix.lstrip(".")
        return LoadedDocument(text=numbered, title=file_path.name, media_type=media_type, metadata=metadata)
    if suffix in TEXT_SUFFIXES or media_type.startswith("text/"):
        return LoadedDocument(
            text=file_path.read_text(encoding="utf-8", errors="replace"),
            title=file_path.name,
            media_type=media_type,
            metadata=metadata,
        )
    raise ValueError(f"unsupported file type: {suffix or media_type}; supported: PDF, HTML, text, and source code")


def _load_pdf(path: Path, metadata: dict[str, Any]) -> LoadedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF loading requires: pip install 'trisynapse-memory[files]'") from exc
    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append(f"\n[Page {index}]\n{text.strip()}")
    metadata["page_count"] = len(reader.pages)
    title = str((reader.metadata or {}).get("/Title") or path.name)
    return LoadedDocument(text="\n".join(pages).strip(), title=title, media_type="application/pdf", metadata=metadata)

