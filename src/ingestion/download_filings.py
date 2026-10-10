import json
import requests
from pathlib import Path

from src.config import (
    NVIDIA_CIK,
    NVIDIA_RAW_DIR,
    SEC_HEADERS,
)


SUBMISSIONS_URL = (
    f"https://data.sec.gov/submissions/CIK{NVIDIA_CIK}.json"
)


def get_submissions():
    response = requests.get(
        SUBMISSIONS_URL,
        headers=SEC_HEADERS,
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def get_filings(forms=("10-K", "10-Q")):
    data = get_submissions()

    recent = data["filings"]["recent"]

    filings = []

    for i, form in enumerate(recent["form"]):
        if form not in forms:
            continue

        filings.append({
            "form": form,
            "filing_date": recent["filingDate"][i],
            "report_date": recent["reportDate"][i],
            "accession_number": recent["accessionNumber"][i],
            "primary_document": recent["primaryDocument"][i],
        })

    return filings


def build_filing_url(filing):
    accession = filing["accession_number"].replace("-", "")

    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(NVIDIA_CIK)}/"
        f"{accession}/"
        f"{filing['primary_document']}"
    )


def download_filing(filing):
    url = build_filing_url(filing)

    response = requests.get(
        url,
        headers=SEC_HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    filename = (
        f"{filing['filing_date']}_"
        f"{filing['form']}_"
        f"{filing['accession_number'].replace('-', '')}.html"
    )

    output_path = NVIDIA_RAW_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        response.text,
        encoding="utf-8",
    )

    print(f"Downloaded: {filename}")

    return output_path


def save_metadata(filings):
    metadata_path = NVIDIA_RAW_DIR / "filings.json"

    metadata_path.write_text(
        json.dumps(filings, indent=2),
        encoding="utf-8",
    )


def main():
    filings = get_filings()

    # Start small while developing.
    filings = filings[:10]

    save_metadata(filings)

    for filing in filings:
        try:
            download_filing(filing)
        except requests.RequestException as e:
            print(
                f"Failed to download "
                f"{filing['accession_number']}: {e}"
            )


if __name__ == "__main__":
    main()