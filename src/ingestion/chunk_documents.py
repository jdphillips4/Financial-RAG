import json
from pathlib import Path

from src.config import (
    NVIDIA_PROCESSED_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = min(
            start + chunk_size,
            len(words),
        )

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def build_chunks():
    all_chunks = []

    for path in NVIDIA_PROCESSED_DIR.glob("*.txt"):
        text = path.read_text(
            encoding="utf-8"
        )

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{path.stem}_{i}",
                "source_file": path.name,
                "company": "NVIDIA",
                "ticker": "NVDA",
                "text": chunk,
            })

    output_path = (
        NVIDIA_PROCESSED_DIR
        / "chunks.json"
    )

    output_path.write_text(
        json.dumps(
            all_chunks,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Created {len(all_chunks)} chunks.")

    return all_chunks


if __name__ == "__main__":
    build_chunks()