"""Side-by-side imputation accuracy on the chr22 5-sample holdout set:
Michigan-HRC vs Beagle-1KG vs naive mean-imputation, all evaluated
against the same 1,348 masked sites in holdout_truth.tsv.
"""

import gzip
import subprocess
import sys
from pathlib import Path

BATCH = Path("data/michigan_batch_test")
TRUTH = BATCH / "holdout_truth.tsv"
INPUT_VCF = BATCH / "chr22.vcf.gz"
KG_REF = next(Path("data/1kg").glob("ALL.chr22.*.vcf.gz"))
JAVA = "tools/jdk-21.0.5+11-jre/bin/java"
BEAGLE = "tools/beagle.jar"


def load_truth():
    truth = {}
    with open(TRUTH) as f:
        hdr = f.readline().rstrip().split("\t")
        samples = hdr[4:]
        for line in f:
            r = line.rstrip().split("\t")
            truth[int(r[1])] = (r[2], r[3], [(int(g[0]) + int(g[2])) for g in r[4:]])
    return truth, samples


def concordance(vcf_path, truth, samples):
    n_match = n_total = 0
    with gzip.open(vcf_path, "rt") as f:
        cols = []
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                h = line.rstrip().split("\t")
                cols = [h.index(s) for s in samples]
                continue
            r = line.rstrip().split("\t")
            pos = int(r[1])
            if pos not in truth:
                continue
            tref, talt, tdos = truth[pos]
            if (r[3], r[4]) != (tref, talt):
                continue
            gt_i = r[8].split(":").index("GT")
            for c, td in zip(cols, tdos):
                g = r[c].split(":")[gt_i]
                if len(g) < 3 or g[0] not in "01" or g[2] not in "01":
                    continue
                d = int(g[0]) + int(g[2])
                n_total += 1
                if d == td:
                    n_match += 1
    return n_match, n_total


def mean_impute_concordance(truth):
    """For each held-out site, predict dosage = round(2·alt_AF) using the
    1KG EUR allele frequency, and compare to truth."""
    af = {}
    with gzip.open(KG_REF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            r = line.split("\t", 9)
            pos = int(r[1])
            if pos not in truth:
                continue
            for kv in r[7].split(";"):
                if kv.startswith("EUR_AF="):
                    af[pos] = float(kv[7:].split(",")[0])
                    break
    n_match = n_total = 0
    for pos, (_, _, tdos) in truth.items():
        p = af.get(pos)
        if p is None:
            continue
        pred = round(2 * p)
        for td in tdos:
            n_total += 1
            if pred == td:
                n_match += 1
    return n_match, n_total


def main():
    truth, samples = load_truth()
    print(f"{len(truth)} held-out sites × {len(samples)} samples\n")

    print("running Beagle on the masked chr22 input...", file=sys.stderr)
    out = BATCH / "chr22.beagle"
    subprocess.run(
        [
            JAVA,
            "-Xmx8g",
            "-jar",
            BEAGLE,
            f"gt={INPUT_VCF}",
            f"ref={KG_REF}",
            f"out={out}",
            "chrom=22",
            "nthreads=16",
        ],
        check=True,
        capture_output=True,
    )
    beagle_vcf = Path(f"{out}.vcf.gz")

    rows = []
    m, n = concordance(BATCH / "chr22.dose.vcf.gz", truth, samples)
    rows.append(("Michigan (HRC, eagle+minimac4)", m, n))
    m, n = concordance(beagle_vcf, truth, samples)
    rows.append(("Beagle (1KG Phase 3)", m, n))
    m, n = mean_impute_concordance(truth)
    rows.append(("Mean imputation (round 2·EUR_AF)", m, n))

    print(f"{'method':<36} {'matched':>9} {'total':>7} {'concordance':>12}")
    print("-" * 70)
    for name, m, n in rows:
        print(f"{name:<36} {m:>9,} {n:>7,} {m / max(n, 1):>11.2%}")


if __name__ == "__main__":
    main()
