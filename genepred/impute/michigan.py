"""Michigan Imputation Server (Minimac4) — submit / status / fetch.

This backend cannot be fully automated: the server emails a one-time
password to decrypt the results. So the flow is split:

  1. submit(genotypes, out_dir)
       → conforms VCFs, uploads via REST, writes <out_dir>/job.json,
         exits. The job typically runs 30–120 min server-side.
  2. status(out_dir)         — poll and print state.
  3. fetch(out_dir, password)
       → downloads encrypted zips, decrypts with the emailed password,
         leaves chr*.dose.vcf.gz alongside job.json.

State lives entirely in job.json so the three calls can run in
different processes / sessions.

Server quirks worth knowing:
  - Minimum of 5 samples per upload — verified via API ("At least 5
    samples must be uploaded."). We pad by duplicating columns when
    fewer are supplied.
  - All samples in one submission must pass a cross-sample concordance
    check, so mixing array vendors (e.g., 23andMe + AncestryDNA) in
    one batch usually fails — submit per-vendor batches instead.
  - Results are AES-zipped; system `unzip` can't open them, so the
    fetch step shells out to `7z`.

Get an API token at https://imputationserver.sph.umich.edu →
Account → Profile → API Token, and set MICHIGAN_API_TOKEN.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from genepred.io import bgzf_compress, load_genotype_by_chrom, parse_chroms
from genepred.paths import find_tool, kg_dir

API = "https://imputationserver.sph.umich.edu/api/v2"


# ----------------------------------------------------------------- VCF prep


def _merge_chrom(chrom, per_sample, kg_vcf, out_path, sample_names, holdout_frac, rng):
    """Multi-sample VCF for one chromosome at the intersection of
    per-sample sites, conformed to 1KG REF/ALT. Optional random holdout
    for imputation-accuracy checking."""
    common = set.intersection(*(set(s.get(chrom, {})) for s in per_sample))
    held = {}
    n_ok = 0
    with gzip.open(kg_vcf, "rt") as ref, open(out_path, "w") as out:
        out.write("##fileformat=VCFv4.2\n")
        out.write("##source=genepred.impute.michigan (REF/ALT from 1KG Phase3)\n")
        out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        out.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(sample_names)
            + "\n"
        )
        for line in ref:
            if line.startswith("#"):
                continue
            t1 = line.find("\t")
            t2 = line.find("\t", t1 + 1)
            try:
                pos = int(line[t1 + 1 : t2])
            except ValueError:
                continue
            if pos not in common:
                continue
            row = line.split("\t", 8)
            r, a = row[3], row[4]
            if len(r) != 1 or len(a) != 1 or "," in a:
                continue
            gts, ok = [], True
            for s in per_sample:
                _, a1, a2 = s[chrom][pos]
                if {a1, a2} <= {r, a}:
                    gts.append(f"{int(a1 == a)}/{int(a2 == a)}")
                else:
                    ok = False
                    break
            if not ok:
                continue
            if holdout_frac > 0 and rng.random() < holdout_frac:
                held[pos] = (r, a, list(gts))
                gts = ["./."] * len(gts)
            out.write(
                f"{chrom}\t{pos}\t{row[2]}\t{r}\t{a}\t.\t.\t.\tGT\t"
                + "\t".join(gts)
                + "\n"
            )
            n_ok += 1
    return n_ok, held


def _bgzip_tabix(path: Path) -> Path:
    """bgzip + tabix the VCF. Falls back to a pure-Python BGZF writer
    when the htslib binaries aren't available; in that case the .tbi
    index is skipped (Michigan doesn't require it on upload)."""
    out = Path(f"{path}.gz")
    try:
        bgzip = find_tool("bgzip")
        subprocess.run([bgzip, "-f", str(path)], check=True)
    except FileNotFoundError:
        bgzf_compress(path.read_bytes(), out)
        path.unlink()
    try:
        tabix = find_tool("tabix")
        subprocess.run([tabix, "-f", "-p", "vcf", str(out)], check=True)
    except FileNotFoundError:
        pass
    return out


def prepare(
    genotype_files: list, out_dir: Path, chroms: str = "1-22", holdout_frac: float = 0.0
) -> list[Path]:
    """Conform genotype files to multi-sample bgzipped VCFs ready for
    upload. Pads to ≥3 samples (Michigan's minimum) by duplicating columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(f) for f in genotype_files]
    sample_names = [p.stem.split(".")[0] for p in paths]
    print(f"[michigan] loading {len(paths)} genotype file(s)...", file=sys.stderr)
    per_sample = [load_genotype_by_chrom(f) for f in paths]

    if len(per_sample) < 5:
        i = 0
        while len(per_sample) < 5:
            per_sample.append(per_sample[i % len(paths)])
            sample_names.append(f"{sample_names[i % len(paths)]}_dup{len(per_sample)}")
            i += 1

    rng = random.Random(0)
    held_all = {}
    out_files = []
    for chrom in parse_chroms(chroms):
        kg = next(kg_dir().glob(f"ALL.chr{chrom}.*.vcf.gz"), None)
        if kg is None:
            print(f"  chr{chrom:>2}: no 1KG panel, skip", file=sys.stderr)
            continue
        out_path = out_dir / f"chr{chrom}.vcf"
        n, held = _merge_chrom(
            chrom, per_sample, kg, out_path, sample_names, holdout_frac, rng
        )
        held_all.update({(chrom, p): v for p, v in held.items()})
        gz = _bgzip_tabix(out_path)
        out_files.append(gz)
        print(f"  chr{chrom:>2}: {n:,} sites → {gz.name}", file=sys.stderr)

    if held_all:
        with open(out_dir / "holdout_truth.tsv", "w") as f:
            f.write("chrom\tpos\tref\talt\t" + "\t".join(sample_names) + "\n")
            for (c, p), (r, a, gts) in sorted(held_all.items()):
                f.write(f"{c}\t{p}\t{r}\t{a}\t" + "\t".join(gts) + "\n")
    return out_files


# --------------------------------------------------------------- REST flow


def _token(explicit: str | None = None) -> str:
    tok = explicit or os.environ.get("MICHIGAN_API_TOKEN")
    if not tok:
        tf = Path.home() / ".michigan_token"
        if tf.exists():
            tok = tf.read_text().strip()
    if not tok:
        raise RuntimeError(
            "Michigan API token required. Set MICHIGAN_API_TOKEN or write "
            "~/.michigan_token. Get one at "
            "https://imputationserver.sph.umich.edu (Account → API Token)."
        )
    return tok


def _api_post(files, token, refpanel, population, job_name) -> dict:
    boundary = "----genepred-michigan"
    fields = {
        "refpanel": refpanel,
        "population": population,
        "mode": "imputation",
        "phasing": "eagle",
        "build": "hg19",
        "r2Filter": "0",
        "aesEncryption": "yes",
        "meta": "no",
    }
    if job_name:
        fields["job-name"] = job_name
    body = b""
    for k, v in fields.items():
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
        ).encode()
    for fp in files:
        body += (
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="files"; filename="{fp.name}"\r\n'
            f"Content-Type: application/gzip\r\n\r\n"
        ).encode()
        body += fp.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{API}/jobs/submit/imputationserver2",
        data=body,
        headers={
            "X-Auth-Token": token,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def _api_get(path: str, token: str) -> dict:
    req = urllib.request.Request(f"{API}/{path}", headers={"X-Auth-Token": token})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _state_path(out_dir: Path) -> Path:
    return out_dir / "job.json"


def _load_state(out_dir: Path) -> dict:
    p = _state_path(out_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"No Michigan job state at {p}. Run `genepred impute michigan "
            f"submit ...` first."
        )
    return json.loads(p.read_text())


def _save_state(out_dir: Path, state: dict):
    _state_path(out_dir).write_text(json.dumps(state, indent=2))


def submit(
    genotype_files: list,
    out_dir: Path,
    *,
    refpanel: str = "apps@hrc-r1.1",
    population: str = "mixed",
    job_name: str | None = None,
    holdout_frac: float = 0.0,
    token: str | None = None,
) -> str:
    """Prepare VCFs, upload, and persist job state. Returns job_id."""
    out_dir = Path(out_dir)
    tok = _token(token)
    files = prepare(genotype_files, out_dir, holdout_frac=holdout_frac)
    print(f"[michigan] uploading {len(files)} files to {API} ...", file=sys.stderr)
    resp = _api_post(files, tok, refpanel, population, job_name or out_dir.name)
    if not resp.get("success"):
        raise RuntimeError(f"submission failed: {resp}")
    job_id = resp["id"]
    _save_state(
        out_dir,
        {
            "job_id": job_id,
            "refpanel": refpanel,
            "population": population,
            "submitted_at": time.time(),
            "uploaded": [str(f) for f in files],
            "samples": [Path(f).stem.split(".")[0] for f in genotype_files],
        },
    )
    print(f"\n[michigan] submitted job {job_id}", file=sys.stderr)
    print(f"  state file: {_state_path(out_dir)}", file=sys.stderr)
    print(
        f"  follow progress: genepred impute michigan status {out_dir}", file=sys.stderr
    )
    print("  when done, the password arrives by EMAIL. Then run:", file=sys.stderr)
    print(
        f"    genepred impute michigan fetch {out_dir} --password <PWD>",
        file=sys.stderr,
    )
    return job_id


# Cloudgene job state codes (subset).
_STATE_NAMES = {
    1: "queued",
    2: "running",
    3: "exporting",
    4: "done",
    5: "failed",
    6: "cancelled",
}


def status(out_dir: Path, *, token: str | None = None) -> dict:
    out_dir = Path(out_dir)
    state = _load_state(out_dir)
    info = _api_get(f"jobs/{state['job_id']}/status", _token(token))
    code = info.get("state", -1)
    name = _STATE_NAMES.get(code, str(code))
    state["last_status"] = {
        "state": code,
        "name": name,
        "checked_at": time.time(),
        "raw": info,
    }
    _save_state(out_dir, state)
    print(f"[michigan] job {state['job_id']}: {name}", file=sys.stderr)
    return info


def fetch(out_dir: Path, password: str, *, token: str | None = None) -> list[Path]:
    """Download result zips and decrypt with the emailed password.
    Decryption uses 7z (Michigan uses ZipCrypto AES which Python's
    zipfile cannot handle)."""
    out_dir = Path(out_dir)
    state = _load_state(out_dir)
    tok = _token(token)
    job_id = state["job_id"]
    detail = _api_get(f"jobs/{job_id}", tok)
    outputs = detail.get("outputParams", [])
    dl_dir = out_dir / "results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    fetched: list[Path] = []
    for out in outputs:
        for f in out.get("files", []):
            name = f.get("name", "")
            url = f"{API}/{f.get('path', '').lstrip('/')}"
            if not name.endswith(".zip"):
                continue
            dest = dl_dir / name
            if not dest.exists():
                print(f"  fetching {name} ...", file=sys.stderr)
                req = urllib.request.Request(url, headers={"X-Auth-Token": tok})
                with (
                    urllib.request.urlopen(req, timeout=600) as r,
                    open(dest, "wb") as w,
                ):
                    while chunk := r.read(1 << 20):
                        w.write(chunk)
            fetched.append(dest)
    if not fetched:
        raise RuntimeError(
            f"Job {job_id} has no downloadable outputs yet. "
            f"Check `genepred impute michigan status {out_dir}`."
        )
    sevenz = find_tool("7z", "7za")
    for z in fetched:
        print(f"  decrypting {z.name} ...", file=sys.stderr)
        rc = subprocess.run(
            [sevenz, "x", "-y", f"-p{password}", f"-o{dl_dir}", str(z)],
            capture_output=True,
            text=True,
        )
        if rc.returncode != 0:
            raise RuntimeError(
                f"7z failed on {z.name}: {rc.stderr.strip()}. "
                f"Check the emailed password."
            )
    state["fetched_at"] = time.time()
    _save_state(out_dir, state)
    print(f"[michigan] results in {dl_dir}/", file=sys.stderr)
    return sorted(dl_dir.glob("chr*.dose.vcf.gz"))
