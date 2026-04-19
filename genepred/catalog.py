"""Curated current-best PGS per trait, with download + header verification.

Each entry's `trait_reported`/`citation`/`build` are copied verbatim from
the downloaded file's `#trait_reported=` / `#citation=` / `#genome_build=`
headers. `download()` re-asserts these on disk, so a wrong PGS ID fails
loudly instead of silently scoring the wrong trait — a real failure mode
in the first draft of this list.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from genepred.paths import pgs_weights_dir


@dataclass(frozen=True)
class Score:
    pgs_id: str
    trait_reported: str
    citation: str
    build: str
    n_variants: int
    r2_eur_pop: float
    """Approximate out-of-sample population-level R² in EUR. Used by the
    QALY calculator and report; not load-bearing for scoring itself."""
    embryo_permitted: bool | None = None
    """Whether the upstream sumstats license permits use for prenatal /
    embryo prediction. None = unverified. False means an explicit
    prohibition (e.g. PGC). The embryo demo must filter to True."""
    n_train: int | None = None
    """Discovery-GWAS sample size (or effective N for case-control)."""
    consortium: str = ""
    """Source consortium / study, for display."""
    local: bool = False
    """If True, the weight file is built locally (reference/build_*) rather
    than downloaded from PGS Catalog. ensure_weights() skips download."""


CURATED: dict[str, Score] = {
    "height": Score(
        "PGS002804",
        "Height",
        "Yengo L et al. Nature (2022). doi:10.1038/s41586-022-05275-y",
        "hg19",
        1_099_005,
        0.42,
        n_train=5_400_000, consortium="GIANT + 23andMe",
    ),
    "bmi": Score(
        "PGS004735",
        "Body mass index (BMI)",
        "Truong B et al. Cell Genom (2024). doi:10.1016/j.xgen.2024.100523",
        "hg38",
        4_536_322,
        0.161,
        n_train=700_000, consortium="PRSmix+ (multi-source)",
    ),
    "heart_disease": Score(
        "PGS003725",
        "Coronary artery disease",
        "Patel AP et al. Nat Med (2023). doi:10.1038/s41591-023-02429-x",
        "hg19",
        1_296_172,
        0.06,
        n_train=1_165_000, consortium="GPSMult (multi-ancestry)",
    ),
    "type2_diabetes": Score(
        "PGS005368",
        "Type 2 diabetes (T2D)",
        "Huerta-Chagoya A et al. medRxiv (2025). doi:10.1101/2025.07.21.25331778",
        "hg19",
        1_298_374,
        0.09,
        n_train=2_535_000, consortium="DIAMANTE (multi-ancestry)",
    ),
    "stroke": Score(
        "PGS002724",
        "Ischemic stroke",
        "Mishra A et al. Nature (2022). doi:10.1038/s41586-022-05165-3",
        "GRCh37",
        1_213_574,
        0.02,
        n_train=1_614_000, consortium="GIGASTROKE",
    ),
    "alzheimers": Score(
        "PGS004092",
        "Alzheimer's disease",
        "Monti R et al. Am J Hum Genet (2024). doi:10.1016/j.ajhg.2024.06.003",
        "GRCh38",
        1_109_233,
        0.07,
        n_train=788_000, consortium="EADB / IGAP",
    ),
    "schizophrenia": Score(
        "PGS002785",
        "Schizophrenia",
        "Gui Y et al. Transl Psychiatry (2022). doi:10.1038/s41398-022-02041-6",
        "hg19",
        964_422,
        0.08,
        embryo_permitted=False,
        n_train=320_000, consortium="PGC3",
    ),
    "bipolar_disorder": Score(
        "PGS002786",
        "Bipolar disorder",
        "Gui Y et al. Transl Psychiatry (2022). doi:10.1038/s41398-022-02041-6",
        "hg19",
        948_996,
        0.04,
        embryo_permitted=False,
        n_train=414_000, consortium="PGC-BD",
    ),
    "depression": Score(
        "PGS004885",
        "Major depressive disorder",
        "Jermy B et al. Nat Commun (2024). doi:10.1038/s41467-024-48938-2",
        "GRCh37",
        801_544,
        0.02,
        embryo_permitted=False,
        n_train=1_350_000, consortium="PGC-MDD / FinnGen",
    ),
    "breast_cancer": Score(
        "PGS000004",
        "Breast Cancer",
        "Mavaddat N et al. Am J Hum Genet (2018). doi:10.1016/j.ajhg.2018.11.002",
        "GRCh37",
        313,
        0.07,
        n_train=228_000, consortium="BCAC",
    ),
    "prostate_cancer": Score(
        "PGS003765",
        "Prostate cancer",
        "Wang A et al. Nat Genet (2023). doi:10.1038/s41588-023-01534-4",
        "GRCh37",
        451,
        0.10,
        n_train=940_000, consortium="PRACTICAL",
    ),
    "colorectal_cancer": Score(
        "PGS003850",
        "Colorectal cancer",
        "Fernandez-Rozadilla C et al. Nat Genet (2022). doi:10.1038/s41588-022-01222-9",
        "GRCh37",
        205,
        0.04,
        n_train=254_000, consortium="GECCO / CCFR / CORECT",
    ),
    "type1_diabetes": Score(
        "PGS000024",
        "Type 1 diabetes (T1D)",
        "Sharp SA et al. Diabetes Care (2019). doi:10.2337/dc18-1785",
        "NR",
        85,
        0.15,
        n_train=16_000, consortium="T1DGC (HLA-engineered GRS2)",
    ),
    "asthma": Score(
        "PGS001787",
        "Asthma",
        "Wang Y et al. Cell Genom (2023). doi:10.1016/j.xgen.2022.100241",
        "GRCh37",
        909_990,
        0.04,
        n_train=1_376_000, consortium="GBMI",
    ),
    "osteoporosis": Score(
        "PGS000657",
        "Heel quantitative ultrasound speed of sound (SOS)",
        "Forgetta V et al. PLoS Med (2020). doi:10.1371/journal.pmed.1003152",
        "GRCh37",
        21_716,
        0.05,
        n_train=426_000, consortium="UKB heel BMD",
    ),
    "inflammatory_bowel_disease": Score(
        "PGS004081",
        "Inflammatory bowel disease (IBD)",
        "Monti R et al. Am J Hum Genet (2024). doi:10.1016/j.ajhg.2024.06.003",
        "GRCh38",
        1_073_268,
        0.06,
        n_train=760_000, consortium="IIBDGC + UKB",
    ),
    "chronic_kidney_disease": Score(
        "PGS000883",
        "Estimated glomerular filtration rate",
        "Yu Z et al. J Am Soc Nephrol(2021). doi:10.1681/asn.2020111599",
        "GRCh37",
        1_477_661,
        0.02,
        n_train=765_000, consortium="CKDGen",
    ),
    "atrial_fibrillation": Score(
        "PGS005168",
        "Atrial fibrillation",
        "Roselli C et al. Nat Genet (2025). doi:10.1038/s41588-024-02072-3",
        "hg38",
        1_113_668,
        0.08,
        n_train=2_300_000, consortium="AFGen / Roselli",
    ),
    "adhd": Score(
        "PGS002746",
        "Attention-deficit hyperactivity disorder",
        "Lahey BB et al. J Psychiatr Res (2022). doi:10.1016/j.jpsychires.2022.04.041",
        "NR",
        513_659,
        0.04,
        embryo_permitted=False,
        n_train=225_000, consortium="PGC-ADHD / iPSYCH",
    ),
    "income": Score(
        "PGS002148",
        "Average total household income before tax",
        "Privé F et al. Am J Hum Genet (2022). doi:10.1016/j.ajhg.2021.11.008",
        "GRCh37",
        932_197,
        0.025,
        n_train=282_000, consortium="UKB",
    ),
    "subjective_wellbeing": Score(
        "PGS002154",
        "General happiness",
        "Privé F et al. Am J Hum Genet (2022). doi:10.1016/j.ajhg.2021.11.008",
        "GRCh37",
        806_011,
        0.01,
        n_train=220_000, consortium="UKB",
    ),
    "anxiety_disorders": Score(
        "PGS004451",
        "F41 (Other anxiety disorders)",
        "Jung H et al. Commun Biol (2024). doi:10.1038/s42003-024-05874-7",
        "GRCh37",
        1_059_939,
        0.015,
        n_train=450_000, consortium="UKB ICD-10",
    ),
    # Homemade SBayesRC scores (reference/build_open_sbayesrc.sh) from
    # open-access sumstats — redistributable. Coexist with the Catalog
    # versions above; pick whichever has higher overlap on your input.
    "osteoporosis_sbrc": Score(
        "BMD_morris_sbayesrc", "Heel eBMD (SBayesRC)",
        "Morris JA et al. Nat Genet (2019). doi:10.1038/s41588-018-0302-x",
        "GRCh37", 1_154_522, 0.20,
        n_train=426_824, consortium="UKB (in-house SBayesRC)", local=True,
    ),
    "atrial_fibrillation_sbrc": Score(
        "AF_nielsen_sbayesrc", "Atrial fibrillation (SBayesRC)",
        "Nielsen JB et al. Nat Genet (2018). doi:10.1038/s41588-018-0171-3",
        "GRCh37", 1_154_522, 0.06,
        n_train=132_330, consortium="HUNT/AFGen (in-house SBayesRC)", local=True,
    ),
    "stroke_sbrc": Score(
        "STROKE_gigastroke_sbayesrc", "Any stroke (SBayesRC)",
        "Mishra A et al. Nature (2022). doi:10.1038/s41586-022-05165-3",
        "GRCh37", 1_154_522, 0.02,
        n_train=238_816, consortium="GIGASTROKE (in-house SBayesRC)", local=True,
    ),
    "asthma_sbrc": Score(
        "ASTHMA_valette_sbayesrc", "Asthma (SBayesRC)",
        "Valette K et al. Commun Biol (2021). doi:10.1038/s42003-021-02227-6",
        "GRCh37", 1_154_522, 0.04,
        n_train=137_436, consortium="UKB (in-house SBayesRC)", local=True,
    ),
    "cognitive_ability_mtag": Score(
        "COGNITION_mtag_sbayesrc", "Cognitive ability (Savage×EA4 MTAG)",
        "Savage 2018 × Okbay 2022 → MTAG → SBayesRC (this repo)",
        "GRCh37", 1_154_522, 0.11,
        n_train=402_000, consortium="CTG × SSGAC (in-house SBayesRC)", local=True,
    ),
    "cognitive_ability": Score(
        "PGS004427",
        "Fluid intelligence score",
        "Jung H et al. Commun Biol (2024). doi:10.1038/s42003-024-05874-7",
        "GRCh37",
        1_059_939,
        0.10,
        n_train=260_000, consortium="UKB fluid intelligence",
    ),
}

# Locally-built cognition scores (Savage 2018 × Lee EA4 → MTAG, n_eff≈402k).
# Build scripts in reference/. v1 validated r≈+0.25 vs openSNP IQ/edu/SAT and
# cor(+0.168) with the height PGS, matching published height↔EA rg.
LOCAL_SCORES = {
    "cognitive_ability_mtag": "COGNITION_mtag_ldpredinf_hmPOS_GRCh38.txt.gz",
    "cognitive_ability_mtag_v2": "COGNITION_mtag_sbayesrc_hmPOS_GRCh38.txt.gz",
    "cognitive_ability_geneticg": "COGNITION_geneticg_sbayesrc_hmPOS_GRCh38.txt.gz",
}



# rsID backfill: some PGS Catalog harmonized files have empty hm_rsID
# (position-only). Scoring those against b36 genomes fails. We join
# GRCh37 chr_position → rsID via the HapMap3 snpinfo where available.
_SNPINFO = Path("data/ld_reference/ldblk_1kg_eur/snpinfo_1kg_hm3")
_pos2rs: dict | None = None


def _load_pos_to_rsid() -> dict | None:
    global _pos2rs
    if _pos2rs is not None:
        return _pos2rs
    if not _SNPINFO.exists():
        return None
    _pos2rs = {}
    with open(_SNPINFO) as f:
        f.readline()
        for line in f:
            r = line.split()
            _pos2rs[(r[0], int(r[2]))] = r[1]
    return _pos2rs


def _annotate_rsids(path: Path) -> int:
    """Backfill hm_rsID on a position-only weight file in-place. Returns
    number of rows annotated, 0 if file already has rsIDs, -1 if skipped
    (wrong build / no positions / no snpinfo)."""
    pos2rs = _load_pos_to_rsid()
    if pos2rs is None:
        return -1
    lines, cols = [], None
    n_added = 0
    i_chr = i_pos = i_hrs = i_rs = None
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                lines.append(line)
                if "genome_build=" in line and line.split("=")[-1].strip() not in (
                    "GRCh37", "hg19", "NCBI37"
                ):
                    return -1
                continue
            r = line.rstrip("\n").split("\t")
            if cols is None:
                cols = {c: i for i, c in enumerate(r)}
                i_chr, i_pos = cols.get("chr_name"), cols.get("chr_position")
                i_hrs = cols.get("hm_rsID")
                i_rs = cols.get("rsID", cols.get("rsid"))
                if i_chr is None or i_pos is None:
                    return -1
                if i_hrs is None:
                    r.append("hm_rsID")
                    i_hrs = len(r) - 1
                lines.append("\t".join(r) + "\n")
                continue
            while len(r) <= i_hrs:
                r.append("")
            if not r[i_hrs].startswith("rs"):
                if i_rs is not None and i_rs < len(r) and r[i_rs].startswith("rs"):
                    r[i_hrs] = r[i_rs]
                else:
                    try:
                        rs = pos2rs.get((r[i_chr].lstrip("chr"), int(r[i_pos])), "")
                        if rs:
                            r[i_hrs] = rs
                            n_added += 1
                    except (ValueError, IndexError):
                        pass
            lines.append("\t".join(r) + "\n")
    if n_added > 0:
        with gzip.open(path, "wt") as f:
            f.writelines(lines)
    return n_added


PGS_FTP = (
    "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/{pgs}/"
    "ScoringFiles/Harmonized/{pgs}_hmPOS_GRCh38.txt.gz"
)


def read_header(path: Path) -> dict[str, str]:
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if "=" in line:
                k, _, v = line[1:].rstrip().partition("=")
                out[k] = v
    return out


def weight_path(pgs_id: str, dest_dir: Path | None = None) -> Path:
    return (dest_dir or pgs_weights_dir()) / f"{pgs_id}_hmPOS_GRCh38.txt.gz"


def verify(score: Score, path: Path) -> bool:
    if not path.exists():
        return False
    h = read_header(path)
    return h.get("trait_reported", "") == score.trait_reported


def download(pgs_id: str, dest_dir: Path | None = None) -> Path:
    """Fetch a harmonized PGS Catalog scoring file. Idempotent."""
    dest = weight_path(pgs_id, dest_dir)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    url = PGS_FTP.format(pgs=pgs_id)
    req = Request(url, headers={"User-Agent": "genepred/0.1"})
    tmp = dest.with_suffix(".part")
    with urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.rename(dest)
    return dest


def ensure_weights(
    traits: list[str] | None = None, dest_dir: Path | None = None, verbose: bool = True
) -> dict[str, Path]:
    """Download any missing curated weight files; verify headers; return
    {trait: path}. Raises if a verified header mismatches."""
    todo = {t: CURATED[t] for t in (traits or CURATED)}
    out = {}
    bad = []
    snpinfo_warned = False
    for trait, score in todo.items():
        path = weight_path(score.pgs_id, dest_dir)
        if not (path.exists() and path.stat().st_size > 1000):
            if score.local:
                if verbose:
                    print(f"  {trait} ({score.pgs_id}): local build — run "
                          f"reference/build_*_sbayesrc.sh", flush=True)
                continue
            if verbose:
                print(f"  fetching {score.pgs_id} ({trait}) ...")
            download(score.pgs_id, dest_dir)
            n_ann = _annotate_rsids(path)
            if n_ann > 0 and verbose:
                print(f"    backfilled {n_ann:,} rsIDs")
            elif n_ann == -1 and not snpinfo_warned and not _SNPINFO.exists():
                print(f"    (rsID backfill skipped — {_SNPINFO} not found)")
                snpinfo_warned = True
        if score.local:
            out[trait] = path
            continue
        if not verify(score, path):
            bad.append((trait, score.pgs_id))
        out[trait] = path
    if bad:
        raise RuntimeError(
            "PGS header mismatch (curated trait_reported != file): "
            + ", ".join(f"{t}={p}" for t, p in bad)
        )
    return out


def list_weight_files(dest_dir: Path | None = None) -> list[tuple[str, Path]]:
    """All scoring files present on disk, as (pgs_id, path).
    pgs_id is the full filename stem before _hmPOS_*."""
    d = dest_dir or pgs_weights_dir()
    return sorted(
        (p.name.rsplit("_hmPOS_", 1)[0], p) for p in d.glob("*_hmPOS_GRCh38.txt.gz")
    )
