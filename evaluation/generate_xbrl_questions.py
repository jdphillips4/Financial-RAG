"""Generate numeric gold questions from SEC XBRL "Company Facts" data.

Why: ground truth that comes from the SEC's structured data is independent of both the
retriever and the LLM - exactly what a reviewer means by "independently validated".

    python -m evaluation.generate_xbrl_questions --years 4 --out evaluation/questions_xbrl.json
    python -m evaluation.generate_xbrl_questions --years 4 --merge        # append into questions.json
    python -m evaluation.generate_xbrl_questions --from-file companyfacts.json   # offline

Produces two categories:
  reported_fact  - "What was NVIDIA's <metric> in fiscal <year>?"
  calculation    - YoY growth %, gross margin %, operating margin %

Each question carries the formatted figure as a gold evidence snippet (e.g. "130,497"),
and acceptable_sources = the 10-Ks that actually contain that fiscal year's number.
"""

import argparse
import json
from datetime import date

from evaluation.common import QUESTIONS_PATH
from src.ingestion.chunk_documents import fiscal_info

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# key, label used in the question, candidate us-gaap concepts (first that exists wins), unit
METRICS = [
    ("revenue", "total revenue", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"], "USD"),
    ("cost_of_revenue", "cost of revenue", ["CostOfRevenue", "CostOfGoodsAndServicesSold"], "USD"),
    ("gross_profit", "gross profit", ["GrossProfit"], "USD"),
    ("operating_income", "operating income", ["OperatingIncomeLoss"], "USD"),
    ("net_income", "net income", ["NetIncomeLoss"], "USD"),
    ("rd_expense", "research and development expense", ["ResearchAndDevelopmentExpense"], "USD"),
    ("diluted_eps", "diluted earnings per share", ["EarningsPerShareDiluted"], "USD/shares"),
]


def fetch_companyfacts():
    import requests

    from src.config import NVIDIA_CIK, SEC_HEADERS

    resp = requests.get(COMPANYFACTS_URL.format(cik=NVIDIA_CIK), headers=SEC_HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.json()


def annual_values(facts, concepts, unit):
    """{fiscal_year: value} from 10-K full-year facts (latest filing wins on restatement)."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        entries = gaap.get(concept, {}).get("units", {}).get(unit)
        if not entries:
            continue
        best = {}
        for e in entries:
            if e.get("form") != "10-K" or "start" not in e:
                continue
            days = (date.fromisoformat(e["end"]) - date.fromisoformat(e["start"])).days
            if not 350 <= days <= 380:
                continue
            fy, _ = fiscal_info(e["end"], "10-K")
            if fy not in best or e["filed"] > best[fy]["filed"]:
                best[fy] = e
        if best:
            return {fy: e["val"] for fy, e in best.items()}, concept
    return {}, None


def fmt(key, value):
    return f"{value:.2f}" if key == "diluted_eps" else f"{value / 1e6:,.0f}"


def sources_for(fy, years=3):
    """10-Ks that contain fiscal year `fy` (an income statement spans three years)."""
    return [{"filing_type": "10-K", "fiscal_year": fy + i} for i in range(years)]


def build_questions(facts, n_years):
    values = {}
    for key, label, concepts, unit in METRICS:
        vals, concept = annual_values(facts, concepts, unit)
        if vals:
            values[key] = (label, vals, concept, unit)

    if not values:
        raise SystemExit("No annual 10-K facts found - check the companyfacts payload.")

    all_years = sorted({fy for _, vals, _, _ in values.values() for fy in vals})
    years = all_years[-n_years:]
    questions, n = [], 0

    def add(**kw):
        nonlocal n
        n += 1
        questions.append({
            "id": f"xq{n:03d}", "difficulty": "easy", "should_abstain": False,
            "gold_source": "sec_xbrl_companyfacts", "verified": True, "annotator": "xbrl",
            "gold_chunk_ids": [], **kw,
        })

    for key, (label, vals, concept, unit) in values.items():
        for fy in years:
            if fy not in vals:
                continue
            v = vals[fy]
            is_eps = unit == "USD/shares"
            add(
                question=f"What was NVIDIA's {label} in fiscal {fy}?",
                category="reported_fact",
                gold_answer=(f"${v:.2f} per diluted share" if is_eps else f"${v / 1e6:,.0f} million"),
                gold_numeric={"value": v if is_eps else v / 1e6,
                              "unit": "usd" if is_eps else "usd_millions",
                              "tolerance_rel": 0.005 if not is_eps else None,
                              "tolerance_abs": 0.011 if is_eps else None},
                gold_evidence_snippets=[fmt(key, v)],
                acceptable_sources=sources_for(fy),
                scoring_notes=f"XBRL concept {concept}",
            )

    def growth(key, label):
        if key not in values:
            return
        vals = values[key][1]
        for fy in years:
            if fy in vals and fy - 1 in vals and vals[fy - 1]:
                pct = (vals[fy] / vals[fy - 1] - 1) * 100
                add(
                    question=f"By what percentage did NVIDIA's {label} change between fiscal {fy - 1} and fiscal {fy}?",
                    category="calculation", difficulty="medium",
                    gold_answer=f"{pct:.1f}% ({'increase' if pct >= 0 else 'decrease'})",
                    gold_numeric={"value": round(pct, 2), "unit": "percent", "tolerance_abs": 0.3},
                    gold_evidence_snippets=[fmt(key, vals[fy]), fmt(key, vals[fy - 1])],
                    acceptable_sources=sources_for(fy - 1, 3),
                    scoring_notes="Needs both years' figures; percent change computed from XBRL values.",
                )

    def ratio(num_key, den_key, label):
        if num_key not in values or den_key not in values:
            return
        num, den = values[num_key][1], values[den_key][1]
        for fy in years:
            if fy in num and fy in den and den[fy]:
                pct = num[fy] / den[fy] * 100
                add(
                    question=f"What was NVIDIA's {label} in fiscal {fy}, as a percentage of revenue?",
                    category="calculation", difficulty="medium",
                    gold_answer=f"{pct:.1f}%",
                    gold_numeric={"value": round(pct, 2), "unit": "percent", "tolerance_abs": 0.3},
                    gold_evidence_snippets=[fmt(num_key, num[fy]), fmt(den_key, den[fy])],
                    acceptable_sources=sources_for(fy),
                    scoring_notes="Ratio of two XBRL figures; the filing may also state the margin directly.",
                )

    growth("revenue", "total revenue")
    growth("net_income", "net income")
    ratio("gross_profit", "revenue", "gross margin")
    ratio("operating_income", "revenue", "operating margin")
    return questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=4)
    ap.add_argument("--from-file", default=None)
    ap.add_argument("--out", default="evaluation/questions_xbrl.json")
    ap.add_argument("--merge", action="store_true", help="append into evaluation/questions.json")
    args = ap.parse_args()

    facts = json.load(open(args.from_file, encoding="utf-8")) if args.from_file else fetch_companyfacts()
    new = build_questions(facts, args.years)

    if args.merge:
        data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        existing = {q["question"] for q in data["questions"]}
        added = [q for q in new if q["question"] not in existing]
        data["questions"].extend(added)
        QUESTIONS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Merged {len(added)} XBRL questions into {QUESTIONS_PATH} ({len(data['questions'])} total).")
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"questions": new}, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(new)} questions -> {args.out}")


if __name__ == "__main__":
    main()
