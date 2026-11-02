
import json
import re

from src.config import (
    NVIDIA_PROCESSED_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def extract_metadata(filename):
    match = re.match(
        r"(\d{4}-\d{2}-\d{2})_(10-K|10-Q)_",
        filename,
    )

    if not match:
        return {
            "filing_date": None,
            "filing_type": None,
        }

    return {
        "filing_date": match.group(1),
        "filing_type": match.group(2),
    }


def extract_section(text):
    """
    Extract a section heading when one is present.
    """
    match = re.search(
        r"(Item\s+\d+[A-Z]?\.\s+[^\n]+)",
        text,
        re.IGNORECASE,
    )

    return match.group(1).strip() if match else None


def split_into_blocks(text):
    """
    Keep tables intact as separate blocks.
    Split remaining text into paragraphs.
    """
    pattern = r"(\[TABLE\].*?\[/TABLE\])"
    parts = re.split(pattern, text, flags=re.DOTALL)

    blocks = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        if part.startswith("[TABLE]"):
            blocks.append({
                "text": part,
                "type": "table",
            })
        else:
            paragraphs = re.split(r"\n+", part)

            for paragraph in paragraphs:
                paragraph = re.sub(
                    r"\s+",
                    " ",
                    paragraph,
                ).strip()

                if paragraph:
                    blocks.append({
                        "text": paragraph,
                        "type": "text",
                    })

    return blocks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    blocks = split_into_blocks(text)

    chunks = []
    current_blocks = []
    current_words = 0
    current_type = None

    for block in blocks:
        block_text = block["text"]
        block_type = block["type"]
        word_count = len(block_text.split())

        # Keep tables together, even if they exceed chunk_size.
        if block_type == "table":
            if current_blocks:
                chunks.append({
                    "text": "\n".join(current_blocks),
                    "chunk_type": current_type,
                })
                current_blocks = []
                current_words = 0

            chunks.append({
                "text": block_text,
                "chunk_type": "table",
            })
            continue

        # Split unusually long text blocks.
        if word_count > chunk_size:
            if current_blocks:
                chunks.append({
                    "text": "\n".join(current_blocks),
                    "chunk_type": current_type,
                })
                current_blocks = []
                current_words = 0

            words = block_text.split()

            for start in range(0, len(words), chunk_size - overlap):
                piece = words[start:start + chunk_size]

                if not piece:
                    break

                chunks.append({
                    "text": " ".join(piece),
                    "chunk_type": "text",
                })

            continue

        # Start a new chunk when the size limit is reached.
        if current_words + word_count > chunk_size:
            chunks.append({
                "text": "\n".join(current_blocks),
                "chunk_type": current_type,
            })

            current_blocks = []
            current_words = 0

        current_blocks.append(block_text)
        current_words += word_count
        current_type = block_type

    if current_blocks:
        chunks.append({
            "text": "\n".join(current_blocks),
            "chunk_type": current_type,
        })

    return chunks


def build_chunks():
    all_chunks = []

    for path in sorted(NVIDIA_PROCESSED_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")

        metadata = extract_metadata(path.name)
        chunks = chunk_text(text)

        current_section = None

        for i, chunk in enumerate(chunks):
            section = extract_section(chunk["text"])

            if section:
                current_section = section

            all_chunks.append({
                "chunk_id": f"{path.stem}_{i}",
                "source_file": path.name,
                "company": "NVIDIA",
                "ticker": "NVDA",
                "filing_date": metadata["filing_date"],
                "filing_type": metadata["filing_type"],
                "section": current_section,
                "chunk_type": chunk["chunk_type"],
                "text": chunk["text"],
            })

    output_path = NVIDIA_PROCESSED_DIR / "chunks.json"

    output_path.write_text(
        json.dumps(all_chunks, indent=2),
        encoding="utf-8",
    )

    print(f"Created {len(all_chunks)} chunks.")

    return all_chunks


if __name__ == "__main__":
    build_chunks()