"""Table-aware PDF loading.

PyMuPDFLoader flattens tables to one cell per line, so values lose their column
headers. This loader:

1. Detects tables per page with PyMuPDF ``page.find_tables(strategy="lines_strict")``.
   "lines_strict" uses only stroked ruling lines; the default strategy also
   treats filled rectangles as edges, and the striped watermark logo in the
   Star Health PDF produced phantom columns that split cells mid-word.
2. Handles pages whose content is drawn rotated (the landscape premium
   illustration on page index 2) by rendering the page upright onto a
   temporary page, so rows read left-to-right and table + text coordinates
   share one frame.
3. Serializes every table ROW-WISE, repeating the (multi-level) column
   headers on each row, filling merged cells, and prefixing each row with the
   table title (the paragraph directly above the table + any title rows inside
   it) and in-table section labels ("Illustration 1"). Cell text is built from
   whole words so a word is never cut across cells. Each table also gets one
   "header" chunk: a sentence from one fixed template for every table,
   "Table <title> gives <row label> for each <column group>: <values>."
   Rows and header share metadata "table_key" (used by rag_chain to attach the
   header when a row is retrieved).
4. Rejects unreliable detections (grid cutting through running text, framed
   notes, paragraph-like headers); those regions stay in the text path.
5. Emits the remaining page text (minus anything inside a serialized table)
   through the usual RecursiveCharacterTextSplitter (500 / 80).

Every chunk has metadata: page (0-indexed, same as PyMuPDFLoader), pages
(comma-separated string; chunks with identical text on several pages are kept
once, e.g. "18,19"), source (the path passed in, same as before), filename,
chunk_type ("table" | "text").
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TABLE_STRATEGY = "lines_strict"
# Tables where this many words are cut by column edges are treated as mis-detected.
MAX_SPLIT_WORDS = 5
# Label for the first descriptive column when the header row is itself a list
# of column values (e.g. "Sum Insured (INR) | 5 lacs | 10 lacs ...").
ROW_LABEL_NAME = "Benefit"
# Paragraph-above titles are capped (start kept) so prefixed rows stay readable.
MAX_TITLE_CHARS = 250
_INDEX_COL_RE = re.compile(r"^(s|si|sl|sr)\.?\s*no\.?$", re.I)  # "S.No", "SI.No.", "Sr.No."

_AMOUNT_RE = re.compile(r"^(rs\.?\s*)?[\d.,/&\s-]+(lacs?|lakhs?|crores?|cr|l)?\.?$|^(\d+\s*)?(crore|lacs?|lakhs?)", re.I)


def _clean(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.replace(" ", " ")
    text = re.sub(r"-\n(?=[a-z])", "-", text)  # keep "co-\npayment" -> "co-payment"
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class _Cell:
    bbox: tuple
    text: str
    row: int  # physical row the cell starts in
    col: int  # physical column the cell starts in


@dataclass
class _TableOut:
    bbox: tuple
    rows: List[str] = field(default_factory=list)  # serialized rows, in order
    header_summary: str = ""


# --------------------------------------------------------------------------- #
# Page orientation
# --------------------------------------------------------------------------- #
def _text_rotation(page: fitz.Page) -> int:
    """Return 90/270 if most text on the page is drawn vertically, else 0."""
    dirs: Counter = Counter()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            n = sum(len(s["text"].strip()) for s in line["spans"])
            dx, dy = line["dir"]
            if abs(dx) >= abs(dy):
                dirs[0] += n
            else:
                dirs[90 if dy < 0 else 270] += n
    total = sum(dirs.values())
    if not total:
        return 0
    rot, n = dirs.most_common(1)[0]
    return rot if rot and n / total > 0.6 else 0


# --------------------------------------------------------------------------- #
# Table serialization
# --------------------------------------------------------------------------- #
def _short(label: str, limit: int = 60) -> str:
    """Drop parenthetical asides from long group headers used inside rows."""
    if len(label) <= limit:
        return label
    return re.sub(r"\s*\([^)]*\)", "", label).strip() or label


def _join_grouped(fields: List[tuple]) -> str:
    """fields: (parent, label, value|None). Writes each parent once:
    'Parent -> a: 1; b: 2 | Other: 3'. value None -> header-only listing."""
    pieces: List[str] = []
    k = 0
    while k < len(fields):
        parent = fields[k][0]
        m = k
        group = []
        while m < len(fields) and fields[m][0] == parent and (parent or m == k):
            _, label, value = fields[m]
            if value is None:
                group.append(label)
            else:
                group.append(f"{label}: {value}" if label else value)
            m += 1
        sep = " / " if fields[k][2] is None else "; "
        pieces.append(f"{parent} -> {sep.join(group)}" if parent else group[0])
        k = m
    return " | ".join(pieces)


def _covers(bbox: tuple, x: float, y: float, tol: float = 0.5) -> bool:
    return bbox[0] - tol <= x <= bbox[2] + tol and bbox[1] - tol <= y <= bbox[3] + tol


_LIST_START_RE = re.compile(r"^\s*(\d+\.|[a-zA-Z]\)|\(?[ivxIVX]+[.)]|[•·\-–])(\s|$)")


def _paragraph_above(page: fitz.Page, bbox: tuple, boilerplate: frozenset,
                     other_tables: List[fitz.Rect] = (),
                     max_gap: float = 40, max_lines: int = 8) -> str:
    """The paragraph immediately above a table, used as the table's title/context.

    Starts at the nearest text line above the table (same column, within max_gap)
    and walks upward through lines separated by normal line spacing. Stops at a
    paragraph break (larger vertical gap), after a line that opens a list item
    ("12. Health Checkup Assure: ..."), or after max_lines. A list marker set
    as a hanging indent ("1." left of the text line) also ends the paragraph.
    Running headers / footers (boilerplate) and text inside other tables are skipped.
    """
    tb = fitz.Rect(bbox)
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for ln in block.get("lines", []):
            text = _clean("".join(s["text"] for s in ln["spans"]))
            r = fitz.Rect(ln["bbox"])
            c = fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
            if text and text not in boilerplate and not any(c in t for t in other_tables):
                lines.append((r, text))

    def overlaps(a: fitz.Rect, b: fitz.Rect) -> bool:
        return min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5 * min(a.width, b.width)

    def hanging_marker(r0: fitz.Rect) -> Optional[str]:
        for r, t in lines:
            if abs(r.y0 - r0.y0) < 2 and r.x1 <= r0.x0 + 2 and r.x0 >= r0.x0 - 40 and _LIST_START_RE.match(t):
                return t
        return None

    above = [(r, t) for r, t in lines if -2 <= tb.y0 - r.y1 <= max_gap and overlaps(r, tb)]
    if not above:
        return ""
    cur = max(above, key=lambda rt: rt[0].y1)  # nearest line above the table
    picked = [cur]
    while len(picked) < max_lines and not _LIST_START_RE.match(cur[1]):
        r0, _ = cur
        marker = hanging_marker(r0)
        if marker:
            picked.append((r0, marker))
            break
        spacing = max(r0.height, 6)
        prev = [(r, t) for r, t in lines
                if r.y1 <= r0.y0 + 1 and r0.y0 - r.y1 <= 0.8 * spacing and overlaps(r, r0)]
        if not prev:
            break
        cur = max(prev, key=lambda rt: rt[0].y1)
        picked.append(cur)
    text = _clean(" ".join(t for _, t in reversed(picked)))
    if len(text) > MAX_TITLE_CHARS:  # keep the start: it names the clause
        text = text[:MAX_TITLE_CHARS].rsplit(" ", 1)[0] + " ..."
    return text


def _split_word_count(tab, words: list) -> int:
    """Words cut by a column edge that is not covered by a spanning cell.
    A non-zero count means the detected column grid is wrong (phantom columns)."""
    cells = [c for r in tab.rows for c in r.cells if c is not None]
    edges = sorted({round(c[0], 1) for c in cells} | {round(c[2], 1) for c in cells})
    tb = tab.bbox
    n = 0
    for x0, y0, x1, y1, w, *_ in words:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not (tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]) or len(w) < 2:
            continue
        for e in edges:
            if x0 < e - 1.5 and x1 > e + 1.5:
                # legit if the word sits in a cell that spans across this edge
                if not any(c[0] < e - 1 and c[2] > e + 1 and c[1] <= cy <= c[3] for c in cells):
                    n += 1
                break
    return n


def _words_in(bbox: tuple, words: list) -> str:
    """Text of the whole words whose centre lies in bbox. Unlike tab.extract() (which
    assigns individual characters) this never cuts a word across two cells."""
    x0, y0, x1, y1 = bbox
    hit = [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
    hit.sort(key=lambda w: (w[5], w[6], w[7]))  # block, line, word number
    return _clean(" ".join(w[4] for w in hit))


def _boilerplate_lines(pdf: fitz.Document) -> frozenset:
    """Lines repeated on at least half the pages (running headers/footers, document IDs)."""
    seen: Counter = Counter()
    for page in pdf:
        lines = {_clean(ln) for ln in page.get_text().splitlines()}
        seen.update(ln for ln in lines if ln)
    cutoff = max(3, pdf.page_count // 2)
    return frozenset(ln for ln, n in seen.items() if n >= cutoff)


def _header_sentence(title: str, col_parts: List[List[str]]) -> str:
    """One fixed template for every table:
        "Table <title> gives <row label> for each <column group>: <v1>, <v2>, ...; for each <group2>: ..."
    row label   = leftmost non-index column without a parent header
    column group = parent header of the remaining columns; columns without a
                   parent form the group "column"."""
    cols = []
    for p in col_parts:
        if p and (not cols or p != cols[-1]):  # collapse phantom sub-columns
            cols.append(p)
    cols = [p for p in cols if not _INDEX_COL_RE.match(p[-1])]
    row_idx = next((k for k, p in enumerate(cols) if len(p) == 1), None)
    row_label = cols[row_idx][0] if row_idx is not None else "rows"
    groups: List[tuple] = []  # (group name, [values])
    for k, p in enumerate(cols):
        if k == row_idx:
            continue
        name = " - ".join(p[:-1]) if len(p) > 1 else "column"
        if groups and groups[-1][0] == name:
            groups[-1][1].append(p[-1])
        else:
            groups.append((name, [p[-1]]))
    body = "; ".join(f"for each {name}: {', '.join(vals)}" for name, vals in groups)
    head = f"Table {title}" if title else "Untitled table"
    return f"{head} gives {row_label} {body}." if body else f"{head} gives {row_label}."


def _serialize_table(tab, words: list, paragraph_title: str = "",
                     ignore_captions: frozenset = frozenset()) -> Optional[_TableOut]:
    n_rows, n_cols = tab.row_count, tab.col_count
    if n_rows < 2 or n_cols < 2:
        return None

    # Physical column x-ranges and row y-ranges.
    col_x0 = [None] * n_cols
    col_x1 = [None] * n_cols
    for r in tab.rows:
        for j, c in enumerate(r.cells):
            if c is None:
                continue
            col_x0[j] = c[0] if col_x0[j] is None else min(col_x0[j], c[0])
    for j in range(n_cols):
        nxt = next((col_x0[k] for k in range(j + 1, n_cols) if col_x0[k] is not None), tab.bbox[2])
        col_x1[j] = nxt
    if any(x is None for x in col_x0):
        return None
    col_cx = [(a + b) / 2 for a, b in zip(col_x0, col_x1)]
    # Row bands from cell top edges (row.bbox stretches over vertically merged cells).
    row_y0 = [min((c[1] for c in r.cells if c is not None), default=r.bbox[1]) for r in tab.rows]
    row_y1 = row_y0[1:] + [tab.bbox[3]]
    row_cy = [(a + b) / 2 for a, b in zip(row_y0, row_y1)]

    cells: List[_Cell] = []
    grid: List[List[Optional[_Cell]]] = [[None] * n_cols for _ in range(n_rows)]
    for i, r in enumerate(tab.rows):
        for j, c in enumerate(r.cells):
            if c is not None:
                cell = _Cell(bbox=tuple(c), text=_words_in(c, words), row=i, col=j)
                cells.append(cell)
                grid[i][j] = cell
    # Resolve merged cells: a missing (i, j) is covered by a neighbouring cell's bbox.
    for i in range(n_rows):
        for j in range(n_cols):
            if grid[i][j] is None:
                grid[i][j] = next((c for c in cells if _covers(c.bbox, col_cx[j], row_cy[i])), None)

    def row_units(i: int) -> List[_Cell]:
        seen, out = set(), []
        for j in range(n_cols):
            c = grid[i][j]
            if c is not None and id(c) not in seen:
                seen.add(id(c))
                out.append(c)
        return out

    def is_banner(i: int) -> bool:
        units = [c for c in row_units(i) if c.text]
        if len(units) != 1:
            return False
        c = units[0]
        spanned = sum(1 for j in range(n_cols) if grid[i][j] is c)
        return spanned >= max(2, n_cols // 2)

    # ---- caption rows (full-width title rows above the header) ----
    captions: List[str] = []
    i = 0
    while i < n_rows and is_banner(i):
        text = next(c.text for c in row_units(i) if c.text)
        if text not in ignore_captions:
            captions.append(text)
        i += 1
    if i >= n_rows:
        return None

    # ---- header block: first multi-cell row, extended while it has horizontal
    # spans that are split into sub-columns by the next row ----
    header_rows = [i]
    while header_rows[-1] + 1 < n_rows:
        h, nxt = header_rows[-1], header_rows[-1] + 1
        if is_banner(nxt):
            break
        split = False
        for c in row_units(h):
            span = [j for j in range(n_cols) if grid[h][j] is c]
            if len(span) > 1:
                below = {id(grid[nxt][j]) for j in span if grid[nxt][j] is not None and grid[nxt][j] is not c and grid[nxt][j].text}
                if len(below) > 1:
                    split = True
                    break
        if not split:
            break
        header_rows.append(nxt)
    first_data = header_rows[-1] + 1

    # Column labels: header texts covering each column, top to bottom.
    col_parts: List[List[str]] = []
    for j in range(n_cols):
        parts: List[str] = []
        for h in header_rows:
            c = grid[h][j]
            if c is not None and c.text and c.text not in parts:
                parts.append(c.text)
        col_parts.append(parts)

    # Corner heuristic: header row like "S.No | Sum Insured (INR) | 5 lacs | 10 lacs ...":
    # the descriptive column's header names the other columns' values.
    leaf = [p[-1] if p else "" for p in col_parts]
    for j in range(n_cols - 2):
        rest = [leaf[k] for k in range(j + 1, n_cols) if leaf[k]]
        if leaf[j] and len(rest) >= 2 and all(_AMOUNT_RE.match(x) for x in rest) and not _AMOUNT_RE.match(leaf[j]):
            corner = leaf[j]
            col_parts[j] = col_parts[j][:-1] + [ROW_LABEL_NAME]
            for k in range(j + 1, n_cols):
                if col_parts[k]:
                    # the corner text becomes the parent of the value columns
                    col_parts[k] = col_parts[k][:-1] + [corner, col_parts[k][-1]]
            break

    if sum(1 for p in col_parts if p) < 2:
        return None  # no usable header; leave this region as plain text
    # Paragraph-like "headers" mean this is a framed text box / address list, not a
    # real table (e.g. the Ombudsman list). Leave it to the plain-text path.
    if any(len(p[-1].split()) > 15 for p in col_parts if p):
        return None
    first_label = next((p[-1] for p in col_parts if p), "")
    if re.match(r"^note\b", first_label, re.I):
        return None  # a framed note, not a table

    out = _TableOut(bbox=tuple(tab.bbox))
    # Title = paragraph directly above the table + any title rows inside it.
    caption = " / ".join([paragraph_title] * bool(paragraph_title) + captions)
    out.header_summary = _header_sentence(caption, col_parts)

    # ---- data rows ----
    section = ""
    pending_notes: List[str] = []
    for i in range(first_data, n_rows):
        units = row_units(i)
        if not any(c.text for c in units):
            continue
        row_text = " ".join(c.text for c in units if c.text)
        if re.match(r"^note\b", row_text, re.I):
            # note rows inside the table frame: keep as plain text, cells re-joined
            out.rows.append((f"[{caption}] " if caption else "") + row_text)
            continue
        if is_banner(i):
            text = next(c.text for c in units if c.text)
            # A banner directly followed by data is a section label; otherwise a note.
            section = text
            pending_notes.append(text)
            continue
        pending_notes = []

        # Build (label, value) fields; group columns by parent header so that
        # identical adjacent values under the same parent are written once.
        fields: List[tuple] = []  # (parent, leaf_label, value)
        seen = set()
        for j in range(n_cols):
            c = grid[i][j]
            if c is None or id(c) in seen:
                continue
            seen.add(id(c))
            span = [k for k in range(n_cols) if grid[i][k] is c]
            if not c.text:
                continue
            if len(span) == 1 or all(col_parts[k] == col_parts[j] for k in span):
                parts = col_parts[j]
                parent = _short(" - ".join(parts[:-1])) if len(parts) > 1 else ""
                fields.append((parent, parts[-1] if parts else "", c.text))
            else:
                # Cell spanning several data columns (e.g. "Total Premium for all members
                # ... is Rs.57,460/-"): such cells describe themselves; keep unlabeled.
                fields.append(("", "", c.text))

        # collapse adjacent sibling columns with identical values
        # ("5 lacs / 10 lacs / 15 lacs: Available")
        merged: List[tuple] = []
        for parent, label, value in fields:
            if merged and label and merged[-1][1] and merged[-1][0] == parent and merged[-1][2] == value:
                merged[-1] = (parent, f"{merged[-1][1]} / {label}", value)
            elif merged and label and merged[-1][0] == parent and merged[-1][1] == label:
                # phantom sub-columns under one spanning header: same label -> one value
                merged[-1] = (parent, label, f"{merged[-1][2]} {value}")
            else:
                merged.append((parent, label, value))
        ctx = [x for x in (caption, section) if x]
        prefix = f"[{' > '.join(ctx)}] " if ctx else ""
        out.rows.append(prefix + _join_grouped(merged))

    # Trailing banners (notes like "Note: Premium rates ... exclusive of taxes")
    for note in pending_notes:
        out.rows.append((f"[{caption}] " if caption else "") + note)
    return out if out.rows else None


# --------------------------------------------------------------------------- #
# Page -> chunks
# --------------------------------------------------------------------------- #
def _residual_text(page: fitz.Page, table_rects: List[fitz.Rect]) -> str:
    """Page text with lines inside serialized tables removed (PyMuPDF's 'text' order)."""
    blocks_out: List[str] = []
    for block in page.get_text("dict")["blocks"]:
        lines_out = []
        for line in block.get("lines", []):
            r = fitz.Rect(line["bbox"])
            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            if any(t.x0 <= cx <= t.x1 and t.y0 <= cy <= t.y1 for t in table_rects):
                continue
            lines_out.append("".join(s["text"] for s in line["spans"]))
        if lines_out:
            blocks_out.append("\n".join(lines_out))
    return "\n".join(blocks_out)


def _pack_rows(rows: List[str], size: int) -> List[str]:
    """Group consecutive serialized rows into chunks <= size. Never split a row."""
    chunks, cur = [], ""
    for row in rows:
        if cur and len(cur) + 1 + len(row) > size:
            chunks.append(cur)
            cur = row
        else:
            cur = f"{cur}\n{row}" if cur else row
    if cur:
        chunks.append(cur)
    return chunks


def load_and_chunk_tables(pdf_path: str, chunk_size: int = CHUNK_SIZE,
                          chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    filename = os.path.basename(pdf_path)
    docs: List[Document] = []

    with fitz.open(pdf_path) as pdf:
        boilerplate = _boilerplate_lines(pdf)
        for pno, page in enumerate(pdf):
            base_meta = {"source": pdf_path, "filename": filename, "page": pno,
                         "total_pages": pdf.page_count}

            # Landscape content drawn rotated on a portrait page: render it upright onto a
            # temporary page so table detection and text extraction share one frame.
            tmp = None
            rot = _text_rotation(page)
            if rot:
                tmp = fitz.open()
                r = page.rect
                work = tmp.new_page(width=r.height, height=r.width)
                work.show_pdf_page(work.rect, pdf, pno, rotate=-rot)
            else:
                work = page
            try:
                try:
                    found = work.find_tables(strategy=TABLE_STRATEGY).tables
                except Exception:
                    logger.exception(f"find_tables failed on page {pno}; using plain text")
                    found = []
                words = work.get_text("words")
                tables: List[_TableOut] = []
                page_headers: set = set()
                for t_idx, tab in enumerate(found):
                    split = _split_word_count(tab, words)
                    if split >= MAX_SPLIT_WORDS:
                        # grid cuts through running text (table fused with an adjacent
                        # text column) -> unreliable; keep the region as plain text
                        logger.info(f"Skipping table p{pno} t{t_idx}: {split} words cut by column edges")
                        continue
                    try:
                        others = [fitz.Rect(t.bbox) for t in found if t is not tab]
                        ser = _serialize_table(tab, words,
                                               _paragraph_above(work, tab.bbox, boilerplate, others),
                                               boilerplate)
                    except Exception:
                        logger.exception(f"Table serialization failed (page {pno}, table {t_idx})")
                        ser = None
                    if ser is None:
                        continue  # region stays in the plain-text path
                    tables.append(ser)
                    # table_key links rows to their header chunk (rag_chain attaches the header
                    # when a row is retrieved). Keyed on the header text so a table repeated on
                    # two pages (Schedule of Benefits) shares one key and one header chunk.
                    table_key = hashlib.sha1(f"{filename}|{ser.header_summary}".encode()).hexdigest()[:16]
                    meta = {**base_meta, "chunk_type": "table", "table_index": t_idx,
                            "table_key": table_key}
                    # one table continued across two text columns -> same header twice
                    if ser.header_summary not in page_headers:
                        page_headers.add(ser.header_summary)
                        docs.append(Document(page_content=ser.header_summary,
                                             metadata={**meta, "table_part": "header"}))
                    for chunk in _pack_rows(ser.rows, chunk_size):
                        docs.append(Document(page_content=chunk, metadata={**meta, "table_part": "rows"}))

                text = _residual_text(work, [fitz.Rect(t.bbox) for t in tables])
            finally:
                if tmp is not None:
                    tmp.close()

            if text.strip():
                for chunk in splitter.split_text(text):
                    docs.append(Document(page_content=chunk,
                                         metadata={**base_meta, "chunk_type": "text"}))
    return _dedupe_across_pages(docs)


def _dedupe_across_pages(docs: List[Document]) -> List[Document]:
    """Keep one copy of chunks whose text is identical on several pages (e.g. the
    Schedule of Benefits header on pages 18 and 19). "page" stays the first page;
    "pages" lists all of them as a comma-separated string (Chroma metadata must be
    scalar). Every chunk gets "pages" so consumers can rely on it."""
    first: dict = {}
    out: List[Document] = []
    for d in docs:
        kept = first.get(d.page_content)
        if kept is None:
            d.metadata["pages"] = str(d.metadata["page"])
            first[d.page_content] = d
            out.append(d)
        else:
            pages = kept.metadata["pages"].split(",")
            if str(d.metadata["page"]) not in pages:
                kept.metadata["pages"] = ",".join(pages + [str(d.metadata["page"])])
    return out
