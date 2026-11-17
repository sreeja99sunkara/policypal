"""Structured sources and page-citation handling for the /query response.

The LLM context labels chunks with 0-indexed PyMuPDF page numbers ("[Page 4]"),
so generated answers cite 0-indexed pages ("(Page 4)"). That context is left
untouched (generation is unchanged); this module converts to viewer page
numbers (1-indexed) at the response layer only.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document

SNIPPET_CHARS = 300

# A page number in a citation: 1-3 digits, not part of an amount like "1,500" or
# "10,00,000", and not followed by a unit ("5 lakh", "30 days", "10%").
_NUM = (r"\d{1,3}(?!\d)(?!,\d{3}\b)(?!,\d{2},\d)"
        r"(?!\s*(?:%|/-|lakhs?\b|lacs?\b|crores?\b|days?\b|months?\b|years?\b|rs\b))")
_SEP = r"\s*(?:,|&|\band\b|-|–|to)\s*"
_CITATION_RE = re.compile(
    rf"\b(?P<word>pages?)\s+(?P<nums>{_NUM}(?:{_SEP}{_NUM})*)", re.IGNORECASE
)
_RANGE_SEP_RE = re.compile(r"\s*(?:-|–|to)\s*")


def _viewer_pages(doc: Document) -> List[int]:
    pages = str(doc.metadata.get("pages") or "")
    nums = [int(p) for p in pages.split(",") if p.strip().isdigit()]
    if not nums and doc.metadata.get("page") is not None:
        nums = [int(doc.metadata["page"])]
    return [p + 1 for p in nums]


def _table_title(doc: Document) -> Optional[str]:
    """Stored "table_title" metadata if present; otherwise read the title the
    table loader writes into table chunks (older indexes lack the metadata):
    header chunk "Table <title> gives ..."; row chunk "[<title> > <section>] ..."."""
    if doc.metadata.get("chunk_type") != "table":
        return None
    title = doc.metadata.get("table_title")
    if title:
        return title
    text = doc.page_content
    if doc.metadata.get("table_part") == "header":
        if text.startswith("Table ") and " gives " in text:
            return text[len("Table "):text.index(" gives ")].strip() or None
        return None
    m = re.match(r"^\[(.+?)\]\s", text)
    if m:
        return m.group(1).split(" > ")[0].strip() or None
    return None


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …"


def build_sources(docs: List[Document]) -> List[dict]:
    """Structured sources from retrieved docs, in retrieval order."""
    out = []
    for d in docs:
        pages = _viewer_pages(d)
        out.append({
            "page": int(d.metadata.get("page", pages[0] - 1 if pages else 0)) + 1,
            "pages": pages,
            "chunk_type": d.metadata.get("chunk_type") or "text",
            "title": _table_title(d),
            "snippet": _snippet(d.page_content),
            "text": d.page_content,
        })
    return out


def _expand(nums_text: str) -> Tuple[str, List[int]]:
    """Shift every number in a citation list by +1; return (new text, viewer pages)."""
    pages: List[int] = []
    parts = re.split(f"({_SEP})", nums_text)
    out = []
    for i, part in enumerate(parts):
        if re.fullmatch(r"\d{1,3}", part.strip() or "x"):
            out.append(str(int(part) + 1))
        else:
            out.append(part)
    shifted = "".join(out)
    # collect pages, expanding ranges "5-6" / "5 to 6"
    for item in re.split(r"\s*(?:,|&|\band\b)\s*", shifted):
        bounds = _RANGE_SEP_RE.split(item.strip())
        nums = [int(b) for b in bounds if b.isdigit()]
        if len(nums) == 2 and nums[0] <= nums[1]:
            pages.extend(range(nums[0], nums[1] + 1))
        else:
            pages.extend(nums)
    return shifted, pages


def rewrite_citations(answer: str) -> Tuple[str, List[int]]:
    """'(Page 4)' -> '(Page 5)', '(Pages 4-5)' -> '(Pages 5-6)', '(Page 4, 5)' -> '(Page 5, 6)'.
    Returns the rewritten answer and the sorted, de-duplicated viewer pages cited."""
    cited: set = set()

    def repl(m: re.Match) -> str:
        shifted, pages = _expand(m.group("nums"))
        cited.update(pages)
        return f"{m.group('word')} {shifted}"

    return _CITATION_RE.sub(repl, answer), sorted(cited)
