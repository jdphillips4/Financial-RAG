import re
from bs4 import BeautifulSoup
from src.config import NVIDIA_RAW_DIR, NVIDIA_PROCESSED_DIR


def clean_text(html):
    soup = BeautifulSoup(html, "xml")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    for table in soup.find_all("table"):
        rows = []

        for tr in table.find_all("tr", recursive=True):
            cells = []

            for cell in tr.find_all(["th", "td"], recursive=False):
                value = cell.get_text(" ", strip=True)
                cells.append(value)

            if cells and any(cells):
                rows.append(" | ".join(cell for cell in cells if cell.strip())) #removes empty cells while preserving the order of nonempty values

        if rows:
            table.replace_with(
                "\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n"
            )
        else:
            table.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_filing(path):
    html = path.read_text(encoding="utf-8")
    return clean_text(html)


def main():
    NVIDIA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for path in NVIDIA_RAW_DIR.glob("*.html"):
        text = parse_filing(path)
        output_path = NVIDIA_PROCESSED_DIR / f"{path.stem}.txt"
        output_path.write_text(text, encoding="utf-8")
        print(f"Parsed: {path.name}")


if __name__ == "__main__":
    main()