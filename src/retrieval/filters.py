"""Pull fiscal-year / filing-type hints out of a question.

NVIDIA's filings are near-duplicates year to year, so for "fiscal 2025 revenue"
a chunk from the FY2024 10-K can score just as high as the FY2025 one. These hints
let the retriever boost chunks from the filing the question is actually about.
"""

import re

_FY_PATTERNS = [
    r"fiscal(?:\s+year)?\s+(20\d\d)",
    r"\bFY\s?'?(20\d\d)\b",
    r"\bFY\s?'?(\d\d)\b",
    r"(20\d\d)\s+(?:annual\s+report|10-?K|10-?Q|filing|form\s+10-?K)",
]


def parse_query_hints(question):
    years = []
    for pattern in _FY_PATTERNS:
        for m in re.finditer(pattern, question, flags=re.IGNORECASE):
            y = int(m.group(1))
            if y < 100:
                y += 2000
            if 2000 <= y <= 2100 and y not in years:
                years.append(y)

    q = question.lower()
    form = None
    if re.search(r"\b10-?k\b|annual report", q):
        form = "10-K"
    elif re.search(r"\b10-?q\b|quarter", q):
        form = "10-Q"

    return {"fiscal_years": years, "filing_type": form}
