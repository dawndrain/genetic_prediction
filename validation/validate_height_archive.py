"""Score every 23andMe genotype file in the openSNP IA datadump zip on the
Yengo height PGS, in parallel, without extracting the archive to disk.

Each worker opens its own handle on the zip, reads one entry, parses it
with load_genotypes-equivalent logic, and returns the height-PGS dot
product. Weights are loaded once in the parent and inherited via fork.
"""

import gzip
import multiprocessing as mp
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_height_cm(s: str) -> float | None:
    """Coerce free-text height entries to centimetres."""
    if not isinstance(s, str):
        return None
    s = s.strip().lower().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    v = float(m.group(1))
    if "'" in s or "ft" in s or "feet" in s:
        m2 = re.search(r"(\d+)\s*['f].*?(\d+)", s)
        if m2:
            return int(m2.group(1)) * 30.48 + int(m2.group(2)) * 2.54
        return v * 30.48
    if "cm" in s or 130 <= v <= 215:
        return v
    if "m" in s or 1.3 <= v <= 2.15:
        return v * 100
    if 50 <= v <= 85:  # bare inches
        return v * 2.54
    return None


ARCHIVE = Path("data/opensnp_archives/opensnp_datadump.2017-12-08.zip")
WEIGHT_FILES = {
    "height": Path("data/pgs_scoring_files/PGS002804_hmPOS_GRCh38.txt.gz"),
    "ea": Path("data/pgs_scoring_files/COGNITION_mtag_ldpredinf_hmPOS_GRCh38.txt.gz"),
}
for _k, _fn in [
    ("ea_v2", "COGNITION_mtag_sbayesrc_hmPOS_GRCh38.txt.gz"),
    ("ea_v3", "COGNITION_geneticg_sbayesrc_hmPOS_GRCh38.txt.gz"),
    ("ea_v4", "COGNITION_ea4_sbayesrc_hmPOS_GRCh38.txt.gz"),
    ("ea_v5", "COGNITION_savageiq_sbayesrc_hmPOS_GRCh38.txt.gz"),
]:
    _p = Path("data/pgs_scoring_files") / _fn
    if _p.exists():
        WEIGHT_FILES[_k] = _p

# Populated in main() before Pool fork; children inherit via COW.
# rsid -> (ea, oa, eaf, {trait: weight})
_WEIGHTS: dict[str, tuple[str, str, float | None, dict[str, float]]] = {}


def _load_snpinfo_af():
    """rsID -> (A1, A2, freq_of_A1) from the PRS-CS HapMap3 reference."""
    out = {}
    p = Path("data/ld_reference/ldblk_1kg_eur/snpinfo_1kg_hm3")
    with open(p) as f:
        f.readline()
        for line in f:
            r = line.split()
            out[r[1]] = (r[3].upper(), r[4].upper(), float(r[5]))
    return out


def _load_weights_multi(paths: dict[str, Path]):
    af_ref = _load_snpinfo_af()
    merged: dict[str, tuple[str, str, float | None, dict[str, float]]] = {}
    for trait, path in paths.items():
        with gzip.open(path, "rt") as f:
            cols = None
            for line in f:
                if line.startswith("#"):
                    continue
                r = line.rstrip("\n").split("\t")
                if cols is None:
                    cols = {c: i for i, c in enumerate(r)}
                    i_rs = cols.get("hm_rsID", cols.get("rsID", 0))
                    i_ea = cols["effect_allele"]
                    i_oa = cols.get("other_allele", cols.get("hm_inferOtherAllele"))
                    i_w = cols["effect_weight"]
                    i_af = cols.get("allelefrequency_effect")
                    continue
                rsid = r[i_rs]
                if not rsid.startswith("rs"):
                    continue
                ea = r[i_ea].upper()
                oa = r[i_oa].upper() if i_oa is not None and i_oa < len(r) else ""
                af = None
                if i_af is not None and i_af < len(r) and r[i_af]:
                    try:
                        af = float(r[i_af])
                    except ValueError:
                        pass
                w = float(r[i_w])
                if af is None and rsid in af_ref:
                    a1, a2, maf = af_ref[rsid]
                    if (ea, oa) == (a1, a2):
                        af = maf
                    elif (ea, oa) == (a2, a1):
                        af = 1.0 - maf
                if rsid in merged:
                    e0, o0, af0, ws = merged[rsid]
                    if (ea, oa) == (e0, o0):
                        ws[trait] = w
                    elif (ea, oa) == (o0, e0):
                        ws[trait] = -w
                else:
                    merged[rsid] = (ea, oa, af, {trait: w})
    return merged


_FNAME_RE = re.compile(r"user(\d+)_file\d+_yearofbirth_(\w+)_sex_(\w+)\.([\w-]+)\.txt$")


def _score_entry(name: str):
    m = _FNAME_RE.search(name)
    if not m:
        return None
    uid, yob, sex, platform = m.groups()
    if platform != "23andme":
        return None
    traits = list(WEIGHT_FILES)
    raw = {t: 0.0 for t in traits}
    exp = {t: 0.0 for t in traits}
    n_matched = {t: 0 for t in traits}
    n_imputed = {t: 0 for t in traits}
    n_total = y_called = 0
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(ARCHIVE) as z, z.open(name) as fh:
            for bline in fh:
                line = bline.decode("utf-8", "replace").rstrip("\r\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                rsid, chrom, gt = parts[0], parts[1], parts[3]
                if chrom in ("Y", "24", "chrY"):
                    if len(gt) >= 1 and gt[0] in "ACGT":
                        y_called += 1
                    continue
                if len(gt) != 2 or gt[0] not in "ACGT" or gt[1] not in "ACGT":
                    continue
                n_total += 1
                w = _WEIGHTS.get(rsid)
                if w is None:
                    continue
                ea, oa, af, wts = w
                seen.add(rsid)
                if gt[0] in (ea, oa) and gt[1] in (ea, oa):
                    d = (gt[0] == ea) + (gt[1] == ea)
                    for t, wt in wts.items():
                        raw[t] += wt * d
                        n_matched[t] += 1
                        if af is not None:
                            exp[t] += wt * 2 * af
        # mean-impute every weight-file SNP this genotype didn't observe,
        # using effect-allele frequency where available
        for rsid, (ea, oa, af, wts) in _WEIGHTS.items():
            if rsid in seen or af is None:
                continue
            d = 2 * af
            for t, wt in wts.items():
                raw[t] += wt * d
                exp[t] += wt * d
                n_imputed[t] += 1
        out = dict(
            uid=int(uid),
            sex_label=sex,
            yob=yob,
            file=name,
            n_total=n_total,
            y_called=y_called,
        )
        for t in traits:
            out[f"raw_{t}"] = raw[t]
            out[f"exp_{t}"] = exp[t]
            out[f"matched_{t}"] = n_matched[t]
            out[f"imputed_{t}"] = n_imputed[t]
        return out
    except Exception as e:
        return dict(
            uid=int(uid),
            sex_label=sex,
            yob=yob,
            file=name,
            n_total=0,
            y_called=0,
            error=str(e),
        )


def _load_phenotypes():
    """Height (cm), edu years, SAT M+V, self-reported IQ from the 2017
    openSNP CSV (temporally matched to the 2017 genotype dump)."""
    ph = pd.read_csv(
        "data/opensnp/phenotypes_2017.csv",
        sep=";",
        engine="python",
        on_bad_lines="skip",
        dtype=str,
    )
    cols = {c.strip().lower(): c for c in ph.columns}
    uid_col = next(c for k, c in cols.items() if "user" in k and "id" in k)

    def col(*needles):
        for k, c in cols.items():
            if all(n in k for n in needles):
                return c
        return None

    c_height = col("height")
    c_iq = cols.get("iq")
    c_edu = col("academic", "degree") or col("highest", "education")
    c_sat_m, c_sat_v = col("sat", "math"), col("sat", "verbal")
    print(
        f"  cols: height={c_height!r} iq={c_iq!r} edu={c_edu!r} "
        f"sat={c_sat_m!r}/{c_sat_v!r}",
        file=sys.stderr,
    )

    def num(s, lo, hi):
        try:
            v = float(re.search(r"-?\d+(?:\.\d+)?", str(s)).group())
            return v if lo <= v <= hi else None
        except (AttributeError, ValueError):
            return None

    # Substring keys (matched anywhere in the lowercased string)
    EDU_SUBSTR = {
        "phd": 21,
        "doctor": 21,
        "d.phil": 21,
        "jd": 20,
        "professional": 20,
        "master": 18,
        "msc": 18,
        "mba": 18,
        "postgrad": 18,
        "m.a": 18,
        "m.s": 18,
        "meng": 18,
        "bachelor": 16,
        "bsc": 16,
        "b.a": 16,
        "b.s": 16,
        "undergrad": 16,
        "university": 16,
        "licenc": 16,
        "diplom": 16,
        "laurea": 16,
        "engineer": 16,
        "cum laude": 16,
        "associate": 14,
        "some college": 14,
        "still in": 14,
        "currently": 14,
        "vocational": 13,
        "trade school": 13,
        "technical": 13,
        "certificat": 13,
        "apprentice": 13,
        "a-level": 13,
        "a level": 13,
        "abitur": 13,
        "baccalaur": 13,
        "high school": 12,
        "highschool": 12,
        "ged": 12,
        "secondary": 12,
        "gymnasium": 12,
        "realschule": 10,
        "some high": 10,
        "multiple": 18,
    }
    # Word-boundary keys (matched as whole tokens — avoids 'ba' inside 'global')
    EDU_WORD = {
        "ma": 18,
        "ms": 18,
        "mphil": 18,
        "med": 18,
        "ba": 16,
        "bs": 16,
        "ab": 16,
        "sb": 16,
        "beng": 16,
        "llb": 16,
        "college": 16,
        "degree": 16,
        "aa": 14,
        "as": 14,
        "hs": 12,
        "dd": 21,
        "md": 21,
        "do": 21,
    }

    def edu_years(s):
        s = str(s).lower().strip()
        if not s or s in ("-", "nan", "n/a", "none"):
            return None
        m = re.search(r"\b(\d{1,2})\s*(?:year|yr)", s)
        if m and 0 <= int(m.group(1)) <= 28:
            return float(m.group(1))
        m = re.fullmatch(r"(\d{1,2})", s)
        if m and 6 <= int(m.group(1)) <= 28:
            return float(m.group(1))
        for k in sorted(EDU_SUBSTR, key=len, reverse=True):
            if k in s:
                return float(EDU_SUBSTR[k])
        toks = set(re.findall(r"[a-z]+", s))
        for k, yrs in EDU_WORD.items():
            if k in toks:
                return float(yrs)
        return None

    out = {}
    for _, r in ph.iterrows():
        try:
            u = int(r[uid_col])
        except (ValueError, TypeError):
            continue
        d = out.setdefault(u, {})
        if c_height:
            v = parse_height_cm(str(r.get(c_height, "")))
            if v and 130 <= v <= 215:
                d["height_cm"] = v
        if c_iq:
            v = num(r.get(c_iq, ""), 70, 180)
            if v:
                d["iq"] = v
        if c_edu:
            v = edu_years(r.get(c_edu, ""))
            if v:
                d["edu_years"] = v
        if c_sat_m and c_sat_v:
            m, v = num(r.get(c_sat_m, ""), 200, 800), num(r.get(c_sat_v, ""), 200, 800)
            if m and v:
                d["sat"] = m + v
    n = {
        k: sum(1 for x in out.values() if k in x)
        for k in ("height_cm", "iq", "edu_years", "sat")
    }
    print(f"  parsed phenotypes: {n}", file=sys.stderr)
    return out


def main():
    global _WEIGHTS
    _WEIGHTS = _load_weights_multi(WEIGHT_FILES)
    n_per = {t: sum(1 for v in _WEIGHTS.values() if t in v[3]) for t in WEIGHT_FILES}
    n_af = sum(1 for v in _WEIGHTS.values() if v[2] is not None)
    print(
        f"  {len(_WEIGHTS):,} rsIDs ({n_per}); {n_af:,} with AF for imputation",
        file=sys.stderr,
    )

    with zipfile.ZipFile(ARCHIVE) as z:
        entries = [
            n for n in z.namelist() if n.endswith(".23andme.txt") and "exome" not in n
        ]
    print(f"  {len(entries):,} 23andMe entries to score", file=sys.stderr)

    n_proc = int(os.environ.get("COO_CPUS") or os.cpu_count() or 8)
    print(f"  {n_proc} workers", file=sys.stderr)

    rows = []
    with mp.get_context("fork").Pool(n_proc) as pool:
        for i, r in enumerate(pool.imap_unordered(_score_entry, entries), 1):
            if r:
                rows.append(r)
            if i % 200 == 0 or i == len(entries):
                print(f"  [{i}/{len(entries)}] scored", file=sys.stderr)

    df = pd.DataFrame(rows)
    df = df[df.get("matched_height", 0) > 50_000].copy()

    # Y-chromosome sex inference: 23andMe arrays have ~2k chrY probes; males
    # get calls on most, females get near-zero (no-calls). Threshold by gap.
    yc = df["y_called"].to_numpy()
    thresh = 200
    df["sex_inferred"] = np.where(yc >= thresh, "M", np.where(yc < 30, "F", "?"))
    lab = df["sex_label"].str.upper()
    df["sex"] = np.where(
        lab == "XY", "M", np.where(lab == "XX", "F", df["sex_inferred"])
    )
    agree = df[lab.isin(["XY", "XX"])]
    if len(agree):
        conc = (
            (agree.sex_label.str.upper() == "XY") == (agree.sex_inferred == "M")
        ).mean()
        print(
            f"  Y-call sex concordance with label: {conc:.1%} "
            f"(n={len(agree)}, threshold={thresh})",
            file=sys.stderr,
        )
    print(f"  sex (label+inferred): {df.sex.value_counts().to_dict()}", file=sys.stderr)

    df.to_csv("data/opensnp_archive_pgs.tsv", sep="\t", index=False)
    print(f"wrote {len(df)} usable scores", file=sys.stderr)

    pheno = _load_phenotypes()
    for col in ("height_cm", "iq", "edu_years", "sat"):
        df[col] = df.uid.map(lambda u, _c=col: pheno.get(u, {}).get(_c))

    def rep(label, sub, x, y):
        sub = sub.dropna(subset=[x, y])
        if len(sub) < 5:
            print(f"{label:46s} n={len(sub)} (too few)")
            return
        r = float(np.corrcoef(sub[x], sub[y])[0, 1])
        se = (1 - r**2) / np.sqrt(max(len(sub) - 2, 1))
        print(f"{label:46s} r={r:+.3f}  R²={r**2:.3f}  n={len(sub):>4}  SE(r)≈{se:.3f}")

    uniq = df.sort_values("matched_height", ascending=False).drop_duplicates("uid")

    pairs = [
        ("HEIGHT-PGS → height", "raw_height", "height_cm"),
        ("COG-PGS v1 (LDpred-inf) → IQ", "raw_ea", "iq"),
        ("COG-PGS v1 (LDpred-inf) → edu_years", "raw_ea", "edu_years"),
        ("COG-PGS v1 (LDpred-inf) → SAT (M+V)", "raw_ea", "sat"),
    ]
    for vk, vn in [
        ("v2", "MTAG→SBayesRC"),
        ("v3", "g→SBayesRC"),
        ("v4", "EA4→SBayesRC"),
        ("v5", "SavageIQ→SBayesRC"),
    ]:
        col = f"raw_ea_{vk}"
        if col in uniq.columns:
            pairs += [
                (f"COG-PGS {vk} ({vn}) → IQ", col, "iq"),
                (f"COG-PGS {vk} ({vn}) → edu_years", col, "edu_years"),
                (f"COG-PGS {vk} ({vn}) → SAT (M+V)", col, "sat"),
            ]
    for trait, x, y in pairs:
        print(f"\n=== {trait} ===")
        rep("all (no sex adj)", uniq, x, y)
        for s in ("M", "F"):
            rep(f"  sex={s}", uniq[uniq.sex == s], x, y)
        known = uniq[uniq.sex.isin(["M", "F"])].dropna(subset=[x, y]).copy()
        if len(known) > 10:
            known["male"] = (known.sex == "M").astype(int)
            b = np.polyfit(known.male, known[y], 1)
            known["_resid"] = known[y] - np.polyval(b, known.male)
            rep("  sex-adjusted", known, x, "_resid")
        if y == "height_cm":
            for s in ("M", "F"):
                rep(
                    f"  sex={s}, >500k matched",
                    uniq[(uniq.sex == s) & (uniq.matched_height > 500_000)],
                    x,
                    y,
                )

    print(
        "\n=== sanity: cor(height-PGS, cog-PGS) — expect small positive (rg≈0.15) ==="
    )
    print(
        f"  v1: {np.corrcoef(uniq.raw_height, uniq.raw_ea)[0, 1]:+.3f} (n={len(uniq)})"
    )
    cog_cols = [
        c
        for c in ("raw_ea", "raw_ea_v2", "raw_ea_v3", "raw_ea_v4", "raw_ea_v5")
        if c in uniq.columns
    ]
    for c in cog_cols[1:]:
        v = c.split("_")[-1]
        print(f"  {v}: {np.corrcoef(uniq.raw_height, uniq[c])[0, 1]:+.3f}")
    print("\nCognition score sd & cross-correlations:")
    for c in cog_cols:
        print(f"  sd({c}) = {uniq[c].std():.4e}")
    for i, a in enumerate(cog_cols):
        for b in cog_cols[i + 1 :]:
            print(f"  cor({a}, {b}) = {np.corrcoef(uniq[a], uniq[b])[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
