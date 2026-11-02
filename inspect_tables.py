from bs4 import BeautifulSoup
from pathlib import Path

path = Path("data/raw/nvidia/2024-05-29_10-Q_000104581024000124.html")

soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")

for i, table in enumerate(soup.find_all("table")[:5]):
    print(f"\n{'=' * 60}")
    print(f"TABLE {i}")
    print(f"{'=' * 60}")
    print(table.get_text(" ", strip=True)[:2000])