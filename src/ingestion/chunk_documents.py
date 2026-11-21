"""Section-aware chunking with fiscal-period metadata.

Changes vs. your version (each one targets a retrieval-quality problem):

  1. fiscal_year / fiscal_period metadata. NVIDIA's fiscal year ends in late January,
     so the 10-K filed in Feb 2025 is FY2025. Without this, "fiscal 2025" questions
     cannot be matched to the right filing, and all filings look near-identical.
  2. Section labels come from short heading lines only (not from any "Item 7." that
     appears inside prose such as "see Item 1A. Risk Factors"), and a new chunk
     always starts when the section changes.
  3. Real overlap between consecutive text chunks (your packing path had none).
  4. Big tables are split by rows with the header row repeated, instead of being
     kept as one giant chunk that the embedding model silently truncates.
  5. A short caption line just above a table ("Revenue by Reportable Segments")
     is attached to that table so the table chunk carries its topic.
  6. Every chunk gets `search_text` = "NVIDIA 10-K FY2025 | Item 7 ... \\n text".
     BM25 and the vector store index this, so the filing/year/section are searchable.
"""

import json
import re

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, NVIDIA_PROCESSED_DIR, NVIDIA_RAW_DIR

HEADING_RE = re.compile(
    r"^(?:PART\s+(?P<part>I{1,3}|IV)\b[\s.\-–—,:]*)?"
    r"Item\s+(?P<num>\d{1,2}[A-C]?)\s*[.:\-–—]?\s*(?P<title>[A-Za-z][^|]{2,100})?$",
    re.IGNORECASE,
)
PART_RE = re.compile(r"^PART\s+(I{1,3}|IV)\s*$", re.IGNORECASE)
MAX_HEADING_WORDS = 18
MAX_CAPTION_WORDS = 30


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
def load_filing_index():
    """accession (no dashes) -> {report_date, form} from data/raw/nvidia/filings.json."""
    path = NVIDIA_RAW_DIR / "filings.json"
    if not path.exists():
        return {}
    filings = json.loads(path.read_text(encoding="utf-8"))
    return {
        f["accession_number"].replace("-", ""): {
            "report_date": f.get("report_date"),
            "form": f.get("form"),
        }
        for f in filings
    }


def fiscal_info(report_date, form):
    """NVIDIA fiscal calendar: year ends last Sunday of January.

    report 2025-01-26 (10-K) -> FY2025 / 'FY'
    report 2025-04-27 (10-Q) -> FY2026 / 'Q1'   (Jul -> Q2, Oct -> Q3)
    """
    if not report_date:
        return None, None
    year, month = int(report_date[:4]), int(report_date[5:7])
    fiscal_year = year + 1 if month >= 2 else year
    if form == "10-K":
        return fiscal_year, "FY"
    quarter = {4: "Q1", 5: "Q1", 7: "Q2", 8: "Q2", 10: "Q3", 11: "Q3"}.get(month)
    return fiscal_year, quarter


def extract_metadata(filename, filing_index=None):
    match = re.match(r"(\d{4}-\d{2}-\d{2})_(10-K|10-Q)_(\d+)", filename)
    if not match:
        return {"filing_date": None, "filing_type": None, "report_date": None,
                "fiscal_year": None, "fiscal_period": None}

    filing_date, form, accession = match.groups()
    report_date = (filing_index or {}).get(accession, {}).get("report_date")
    fiscal_year, fiscal_period = fiscal_info(report_date, form)
    return {
        "filing_date": filing_date,
        "filing_type": form,
        "report_date": report_date,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
    }


# --------------------------------------------------------------------------- #
# block splitting
# --------------------------------------------------------------------------- #
def split_into_blocks(text):
    parts = re.split(r"(\[TABLE\].*?\[/TABLE\])", text, flags=re.DOTALL)
    blocks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("[TABLE]"):
            blocks.append({"text": part, "type": "table"})
            continue
        for paragraph in re.split(r"\n+", part):
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if paragraph:
                blocks.append({"text": paragraph, "type": "text"})
    return blocks


def parse_heading(block_text):
    """Return (part, section_label) if this short text block is an Item heading."""
    if len(block_text.split()) > MAX_HEADING_WORDS:
        return None
    m = HEADING_RE.match(block_text.strip())
    if not m:
        return None
    title = (m.group("title") or "").strip(" .")
    label = f"Item {m.group('num').upper()}" + (f". {title}" if title else "")
    return (m.group("part") or "").upper() or None, label


def split_table(text, max_words):
    """Split a [TABLE] block into row groups, repeating the header row."""
    lines = [l for l in text.split("\n") if l not in ("[TABLE]", "[/TABLE]")]
    if sum(len(l.split()) for l in lines) <= max_words or len(lines) <= 2:
        return [text]

    header = lines[0]
    pieces, current, words = [], [], len(header.split())
    for line in lines[1:]:
        w = len(line.split())
        if current and words + w > max_words:
            pieces.append(current)
            current, words = [], len(header.split())
        current.append(line)
        words += w
    if current:
        pieces.append(current)
    return ["[TABLE]\n" + "\n".join([header] + rows) + "\n[/TABLE]" for rows in pieces]


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
def chunk_blocks(blocks, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Yield dicts {text, chunk_type, section, part}."""
    chunks = []
    part, section = None, None
    buf = []          # list of text-block strings
    buf_words = 0

    def flush(carry_overlap=True):
        nonlocal buf, buf_words
        if not buf:
            return
        chunks.append({"text": "\n".join(buf), "chunk_type": "text",
                       "section": section, "part": part})
        tail, words = [], 0
        if carry_overlap and overlap > 0:
            for b in reversed(buf):
                w = len(b.split())
                if words + w > overlap and tail:
                    break
                tail.insert(0, b)
                words += w
                if words >= overlap:
                    break
        buf = tail if carry_overlap else []
        buf_words = sum(len(b.split()) for b in buf)

    for block in blocks:
        text, kind = block["text"], block["type"]

        if kind == "text":
            if PART_RE.match(text):
                flush(carry_overlap=False)
                part = text.upper().replace("PART", "").strip()
                continue
            heading = parse_heading(text)
            if heading:
                flush(carry_overlap=False)
                new_part, section = heading
                part = new_part or part
                buf, buf_words = [text], len(text.split())
                continue

        if kind == "table":
            caption = None
            if buf and len(buf[-1].split()) <= MAX_CAPTION_WORDS and buf_words > 0:
                caption = buf.pop()
                buf_words -= len(caption.split())
            flush(carry_overlap=False)
            for piece in split_table(text, chunk_size):
                body = f"{caption}\n{piece}" if caption else piece
                chunks.append({"text": body, "chunk_type": "table",
                               "section": section, "part": part})
            continue

        words = len(text.split())
        if words > chunk_size:                      # very long paragraph
            flush(carry_overlap=False)
            tokens = text.split()
            step = max(chunk_size - overlap, 1)
            for start in range(0, len(tokens), step):
                piece = tokens[start:start + chunk_size]
                if piece:
                    chunks.append({"text": " ".join(piece), "chunk_type": "text",
                                   "section": section, "part": part})
            continue

        if buf_words + words > chunk_size:
            flush()
        buf.append(text)
        buf_words += words

    flush(carry_overlap=False)
    return chunks


def make_header(meta, section, part):
    bits = ["NVIDIA", meta.get("filing_type") or "filing"]
    if meta.get("fiscal_year"):
        bits.append(f"FY{meta['fiscal_year']}")
    if meta.get("fiscal_period") and meta["fiscal_period"] != "FY":
        bits.append(meta["fiscal_period"])
    header = " ".join(bits)
    if meta.get("report_date"):
        header += f" (period ended {meta['report_date']})"
    where = " ".join(x for x in [f"Part {part}" if part else None, section] if x)
    return f"{header} | {where}" if where else header


def build_chunks():
    filing_index = load_filing_index()
    if not filing_index:
        print("WARNING: data/raw/nvidia/filings.json not found - fiscal_year will be None.")

    all_chunks = []
    for path in sorted(NVIDIA_PROCESSED_DIR.glob("*.txt")):
        meta = extract_metadata(path.name, filing_index)
        blocks = split_into_blocks(path.read_text(encoding="utf-8"))

        for i, chunk in enumerate(chunk_blocks(blocks)):
            header = make_header(meta, chunk["section"], chunk["part"])
            all_chunks.append({
                "chunk_id": f"{path.stem}_{i}",
                "source_file": path.name,
                "company": "NVIDIA",
                "ticker": "NVDA",
                **meta,
                "part": chunk["part"],
                "section": chunk["section"],
                "chunk_type": chunk["chunk_type"],
                "word_count": len(chunk["text"].split()),
                "text": chunk["text"],
                "search_text": f"{header}\n{chunk['text']}",
            })

    out = NVIDIA_PROCESSED_DIR / "chunks.json"
    out.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"Created {len(all_chunks)} chunks -> {out}")
    print("REBUILD THE VECTOR INDEX NOW (the index is tied to chunks.json).")
    return all_chunks


if __name__ == "__main__":
    build_chunks()
