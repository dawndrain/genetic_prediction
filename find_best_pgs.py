"""Search the PGS Catalog API for the best polygenic scores for a set of traits.

For each trait, queries the API for all associated PGS scores, fetches their
performance evaluations, and ranks them by R² (continuous) or AUROC (binary).

Usage:
  python find_best_pgs.py [--ancestry EUR] [--top 5] [--download]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

BASE_URL = "https://www.pgscatalog.org/rest"

# Traits of interest: name -> EFO search term
TRAITS = {
    "height": "body height",
    "heart_disease": "coronary artery disease",
    "type2_diabetes": "type 2 diabetes mellitus",
    "alzheimers": "Alzheimer disease",
    "schizophrenia": "schizophrenia",
    "depression": "major depressive disorder",
    "cognitive_ability": "cognitive function",
    "atrial_fibrillation": "atrial fibrillation",
    "breast_cancer": "breast carcinoma",
    "prostate_cancer": "prostate carcinoma",
    "stroke": "ischemic stroke",
    "colorectal_cancer": "colorectal cancer",
    "bipolar_disorder": "bipolar disorder",
    "chronic_kidney_disease": "chronic kidney disease",
    "asthma": "asthma",
    "inflammatory_bowel_disease": "inflammatory bowel disease",
    "adhd": "attention deficit hyperactivity disorder",
    "type1_diabetes": "type 1 diabetes mellitus",
    "osteoporosis": "osteoporosis",
    "anxiety_disorders": "anxiety disorder",
    "bmi": "body mass index",
    "income": "household income",
    "longevity": "longevity",
    "subjective_wellbeing": "subjective well-being",
}


def api_get(endpoint: str, params: dict | None = None) -> dict:
    """GET from PGS Catalog REST API with rate-limit handling."""
    url = f"{BASE_URL}/{endpoint}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query}"

    req = Request(url, headers={"Accept": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            if e.code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Failed after retries: {url}")


def api_get_all(endpoint: str, params: dict | None = None) -> list:
    """GET all pages from a paginated endpoint."""
    params = dict(params or {})
    params.setdefault("limit", "100")
    results = []
    page = api_get(endpoint, params)
    results.extend(page.get("results", []))

    while page.get("next"):
        # next is a full URL, fetch it directly
        req = Request(page["next"], headers={"Accept": "application/json"})
        with urlopen(req, timeout=30) as resp:
            page = json.loads(resp.read())
        results.extend(page.get("results", []))
    return results


def search_trait(term: str) -> list[dict]:
    """Search for traits matching a term. Returns list of trait objects."""
    return api_get("trait/search", {"term": term.replace(" ", "+")}).get("results", [])


def get_score(pgs_id: str) -> dict:
    """Fetch metadata for a single PGS score."""
    return api_get(f"score/{pgs_id}")


def get_performance(pgs_id: str) -> list[dict]:
    """Fetch all performance evaluations for a PGS score."""
    return api_get_all("performance/search", {"pgs_id": pgs_id})


def extract_best_metric(evaluations: list[dict], ancestry_filter: str | None) -> dict | None:
    """Extract the best R² or AUROC from a list of performance evaluations.

    Returns dict with metric_type, value, sample_size, cohort, or None.
    """
    best = None

    for ev in evaluations:
        # sampleset can be a dict or list
        sampleset = ev.get("sampleset", {})
        if isinstance(sampleset, dict):
            samples = sampleset.get("samples", [])
        elif isinstance(sampleset, list):
            samples = []
            for ss in sampleset:
                samples.extend(ss.get("samples", []))
        else:
            samples = []

        # Check ancestry filter
        if ancestry_filter:
            ancestries = [s.get("ancestry_broad", "") for s in samples]
            if not any(ancestry_filter.lower() in a.lower() for a in ancestries):
                continue

        # Metrics live under performance_metrics
        perf = ev.get("performance_metrics", {})
        all_metrics = (
            perf.get("effect_sizes", []) +
            perf.get("class_acc", []) +
            perf.get("othermetrics", [])
        )

        for metric in all_metrics:
            name = metric.get("name_short", "")
            value = metric.get("estimate")
            if value is None:
                continue

            # We care about R² (continuous) and AUROC (binary)
            if name in ("R²", "R2"):
                metric_type = "R²"
            elif name in ("AUROC", "C-index", "C-statistic"):
                metric_type = name
            else:
                continue

            value = float(value)

            # Get sample size and cohort
            n = sum(s.get("sample_number", 0) for s in samples)
            cohort_name = ""
            for s in samples:
                if s.get("cohorts"):
                    cohort_name = s["cohorts"][0].get("name_short", "")
                    break

            candidate = {
                "metric_type": metric_type,
                "value": value,
                "sample_size": n,
                "cohort": cohort_name,
            }

            if best is None or value > best["value"]:
                best = candidate

    return best


def find_best_pgs_for_trait(trait_name: str, search_term: str, ancestry: str | None, top_n: int) -> list[dict]:
    """Find and rank the best PGS scores for a trait."""
    print(f"\n{'='*60}")
    print(f"Trait: {trait_name} (searching: '{search_term}')")
    print(f"{'='*60}")

    traits = search_trait(search_term)
    if not traits:
        print(f"  No traits found for '{search_term}'")
        return []

    # Collect all PGS IDs from matching traits
    all_pgs_ids = set()
    for t in traits:
        label = t.get("label", "")
        pgs_ids = t.get("associated_pgs_ids", [])
        print(f"  Trait: {label} ({t.get('id', '?')}) — {len(pgs_ids)} scores")
        all_pgs_ids.update(pgs_ids)

    # Also include child trait associations
    for t in traits:
        child_ids = t.get("child_associated_pgs_ids", [])
        if child_ids:
            all_pgs_ids.update(child_ids)

    print(f"  Total unique PGS scores to evaluate: {len(all_pgs_ids)}")

    # Fetch score metadata + performance for each
    scored = []
    for i, pgs_id in enumerate(sorted(all_pgs_ids)):
        if (i + 1) % 20 == 0:
            print(f"  ... fetching {i+1}/{len(all_pgs_ids)}")

        try:
            score_meta = get_score(pgs_id)
            evals = get_performance(pgs_id)
        except Exception as e:
            print(f"  Warning: failed to fetch {pgs_id}: {e}")
            continue

        best = extract_best_metric(evals, ancestry)

        pub = score_meta.get("publication", {})
        entry = {
            "pgs_id": pgs_id,
            "name": score_meta.get("name", ""),
            "variants_number": score_meta.get("variants_number", 0),
            "method": score_meta.get("method_name", ""),
            "publication": f"{pub.get('firstauthor', '?')} ({pub.get('date_publication', '?')[:4]})",
            "doi": pub.get("doi", ""),
            "ftp_scoring_file": score_meta.get("ftp_scoring_file", ""),
            "best_metric": best,
        }
        scored.append(entry)

        # Be nice to the API
        time.sleep(0.2)

    # Rank by best metric value (descending)
    with_metrics = [s for s in scored if s["best_metric"] is not None]
    without_metrics = [s for s in scored if s["best_metric"] is None]

    with_metrics.sort(key=lambda s: s["best_metric"]["value"], reverse=True)
    ranked = with_metrics[:top_n]

    print(f"\n  Top {min(top_n, len(ranked))} scores (of {len(with_metrics)} with metrics, {len(without_metrics)} without):\n")
    print(f"  {'Rank':<5} {'PGS ID':<12} {'Metric':<12} {'Value':<8} {'Variants':<10} {'Method':<30} {'Publication'}")
    print(f"  {'-'*5} {'-'*12} {'-'*12} {'-'*8} {'-'*10} {'-'*30} {'-'*30}")

    for i, s in enumerate(ranked, 1):
        m = s["best_metric"]
        method = (s["method"] or "")[:28]
        print(f"  {i:<5} {s['pgs_id']:<12} {m['metric_type']:<12} {m['value']:<8.4f} {s['variants_number']:<10,} {method:<30} {s['publication']}")

    return ranked


def main():
    parser = argparse.ArgumentParser(description="Find best PGS scores from the PGS Catalog")
    parser.add_argument("--ancestry", default=None,
                        help="Filter evaluations by ancestry (e.g. EUR, EAS, AFR)")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top scores to show per trait (default: 5)")
    parser.add_argument("--traits", nargs="*", default=None,
                        help=f"Subset of traits to search. Options: {', '.join(TRAITS.keys())}")
    parser.add_argument("--download", action="store_true",
                        help="Download the scoring file for the #1 ranked score per trait")
    parser.add_argument("--output", type=Path, default=Path("data/pgs_search_results.json"),
                        help="Save full results to JSON")
    args = parser.parse_args()

    traits_to_search = {k: v for k, v in TRAITS.items()
                        if args.traits is None or k in args.traits}

    if not traits_to_search:
        print(f"No matching traits. Options: {', '.join(TRAITS.keys())}")
        sys.exit(1)

    all_results = {}
    for trait_name, search_term in traits_to_search.items():
        ranked = find_best_pgs_for_trait(trait_name, search_term, args.ancestry, args.top)
        all_results[trait_name] = ranked

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {args.output}")

    # Download top scoring files if requested
    if args.download:
        download_dir = Path("data/pgs_scoring_files")
        download_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading top scoring files to {download_dir}/")

        for trait_name, ranked in all_results.items():
            if not ranked:
                continue
            best = ranked[0]
            url = best["ftp_scoring_file"]
            if not url:
                print(f"  {trait_name}: no download URL for {best['pgs_id']}")
                continue

            filename = url.split("/")[-1]
            dest = download_dir / filename
            if dest.exists():
                print(f"  {trait_name}: {filename} already exists, skipping")
                continue

            print(f"  {trait_name}: downloading {best['pgs_id']} ({filename})...")
            try:
                req = Request(url)
                with urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
                    out.write(resp.read())
                print(f"    saved to {dest}")
            except Exception as e:
                print(f"    failed: {e}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Best PGS per trait")
    print(f"{'='*60}")
    print(f"{'Trait':<20} {'PGS ID':<12} {'Metric':<8} {'Value':<8} {'Variants':<10}")
    print(f"{'-'*20} {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
    for trait_name, ranked in all_results.items():
        if ranked:
            best = ranked[0]
            m = best["best_metric"]
            print(f"{trait_name:<20} {best['pgs_id']:<12} {m['metric_type']:<8} {m['value']:<8.4f} {best['variants_number']:<10,}")
        else:
            print(f"{trait_name:<20} {'—':<12}")


if __name__ == "__main__":
    main()
