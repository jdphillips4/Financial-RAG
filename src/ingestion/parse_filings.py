import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from src.config import (
    NVIDIA_RAW_DIR,
    NVIDIA_PROCESSED_DIR,
)


def clean_text(html):
    soup = BeautifulSoup(html, "lxml")

    # Remove elements that don't contribute useful filing text.
    for element in soup([
        "script",
        "style",
        "noscript",
    ]):
        element.decompose()

    text = soup.get_text(
        separator="\n",
        strip=True,
    )

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def parse_filing(path):
    html = path.read_text(encoding="utf-8")

    return clean_text(html)


def main():
    NVIDIA_PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in NVIDIA_RAW_DIR.glob("*.html"):
        text = parse_filing(path)

        output_path = (
            NVIDIA_PROCESSED_DIR
            / f"{path.stem}.txt"
        )

        output_path.write_text(
            text,
            encoding="utf-8",
        )

        print(f"Parsed: {path.name}")


if __name__ == "__main__":
    main()