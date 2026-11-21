"""Finance-aware tokenizer for BM25.

`text.lower().split()` leaves punctuation glued to words ("margin," != "margin")
and breaks numbers ("$130,497" never matches "130,497"). This keeps numbers intact
(130,497 / 75.0 / 12%) and drops filler words that would dominate BM25 scores.
"""

import re

STOPWORDS = frozenset("""
a an the and or of to in on for with by at from as is are was were be been being it its this that these those
which what why how did does do has have had will would could should can may might than then there their
into over under about between during also not no per our we us
""".split())

_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?|[a-z][a-z\-']*")


def tokenize(text):
    tokens = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tok = tok.rstrip(",")
        if tok in STOPWORDS or len(tok) < 2 and not tok.isdigit():
            continue
        tokens.append(tok)
        # also index the number without thousands separators: 130,497 -> 130497
        if tok[0].isdigit() and "," in tok:
            tokens.append(tok.replace(",", ""))
    return tokens
