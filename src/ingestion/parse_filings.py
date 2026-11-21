"""Parse SEC filing HTML into clean text with [TABLE] blocks.

Changes vs. your version:
  * HTML parser ("lxml") instead of "xml": the XML parser is strict and can silently
    drop content on filings that are not perfectly well-formed.
  * Removes the inline-XBRL hidden header (<ix:header>) and display:none elements.
    That block is a huge pile of machine tags that otherwise lands at the top of
    every filing and pollutes BM25 and embeddings.
  * Re-assembles SEC's split table cells: "$" | "130,497" | ")" -> "$130,497" and
    "(1,234" + ")" -> "(1,234)" so numbers keep their sign and currency.
"""

import re

from bs4 import BeautifulSoup

from src.config import NVIDIA_PROCESSED_DIR, NVIDIA_RAW_DIR

_SYMBOL_ONLY = re.compile(r"^[\$€£]+$")
_CLOSERS = {")", "%", ")%", "%)"}


def _merge_cells(cells):
    """Clean one table row's cell values."""
    merged = []
    pending_currency = ""
    for value in cells:
        value = value.strip()
        if not value:
            continue
        if _SYMBOL_ONLY.match(value):
            pending_currency = value
            continue
        if value in _CLOSERS and merged:
            merged[-1] += value
            continue
        if pending_currency:
            value = pending_currency + value
            pending_currency = ""
        merged.append(value)
    return merged


def clean_text(html):
    soup = BeautifulSoup(html, "lxml")

    for element in soup(["script", "style", "noscript", "ix:header"]):
        element.decompose()

    for element in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        element.decompose()

    for table in soup.find_all("table"):
        rows = []

        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            cells = _merge_cells(cells)
            if cells:
                rows.append(" | ".join(cells))

        if rows:
            table.replace_with("\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n")
        else:
            table.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_filing(path):
    return clean_text(path.read_text(encoding="utf-8"))


def main():
    NVIDIA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for path in sorted(NVIDIA_RAW_DIR.glob("*.html")):
        text = parse_filing(path)
        (NVIDIA_PROCESSED_DIR / f"{path.stem}.txt").write_text(text, encoding="utf-8")
        print(f"Parsed: {path.name} ({len(text.split()):,} words)")


if __name__ == "__main__":
    main()
