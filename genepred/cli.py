"""Command-line interface for genepred.

genepred score <genotype>           # PGS + QALY/risk report (--basic for compact table)
genepred fetch-weights              # download curated PGS files
genepred impute beagle <genotype>   # local imputation (default)
genepred impute michigan submit ... # higher-quality, manual decrypt
genepred impute michigan status ...
genepred impute michigan fetch ... --password X
genepred qaly ...                   # QALY calculator / selection sim
genepred traits                     # list trait parameters
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from genepred import qaly as _qaly
from genepred.catalog import CURATED, ensure_weights
from genepred.impute import beagle as _beagle
from genepred.impute import michigan as _mich
from genepred.scoring import format_report, format_results, score_genome


@click.group()
@click.version_option()
def main():
    """Polygenic score pipeline + embryo-selection QALY model."""


# ----------------------------------------------------------------- scoring


@main.command("score")
@click.argument("genotype", type=click.Path(exists=True))
@click.option("--build", type=click.Choice(["GRCh37", "GRCh38"]), default="GRCh37")
@click.option(
    "--ref-pop",
    type=click.Choice(["EUR", "AFR", "AMR", "EAS", "SAS", "ALL"]),
    default=None,
    help="1KG super-population for normalization (default: inferred from PCs).",
)
@click.option(
    "--no-pc-adjust",
    is_flag=True,
    help="Skip ancestry-PC residualization; use ref-pop only.",
)
@click.option("--weights-dir", type=click.Path(), default=None)
@click.option(
    "--basic",
    is_flag=True,
    help="Compact one-row-per-PGS table (skips QALY/risk annotation).",
)
@click.option("--json", "as_json", is_flag=True)
def cli_score(genotype, build, ref_pop, no_pc_adjust, weights_dir, basic, as_json):
    """Score one genome on every available PGS weight file."""
    results, meta = score_genome(
        genotype,
        build=build,
        pc_adjust=not no_pc_adjust,
        ref_pop=ref_pop,
        weights_dir=Path(weights_dir) if weights_dir else None,
        verbose=not as_json,
    )
    if as_json:
        click.echo(
            json.dumps(
                {
                    "meta": {
                        "super_pop": meta["super_pop"],
                        "n_snps": meta["n_snps"],
                        "pcs": meta["pcs"],
                    },
                    "results": [r.__dict__ for r in results],
                },
                indent=2,
            )
        )
    elif basic:
        click.echo(format_results(results, meta))
    else:
        click.echo(format_report(results, meta, source=genotype))


@main.command("fetch-weights")
@click.option(
    "--trait",
    "traits",
    multiple=True,
    help="Limit to specific curated traits (default: all).",
)
@click.option("--dest-dir", type=click.Path(), default=None)
def cli_fetch_weights(traits, dest_dir):
    """Download the curated PGS Catalog weight files (~300 MB total)."""
    paths = ensure_weights(
        list(traits) or None,
        dest_dir=Path(dest_dir) if dest_dir else None,
    )
    click.echo(f"\n{len(paths)} weight file(s) ready.")
    for trait, p in sorted(paths.items()):
        click.echo(f"  {trait:<28} {CURATED[trait].pgs_id}  {p}")


# ----------------------------------------------------------------- impute


@main.group("impute")
def cli_impute():
    """Genotype imputation (Beagle local, or Michigan submit/fetch)."""


@cli_impute.command("beagle")
@click.argument("genotype", type=click.Path(exists=True))
@click.option("--name", default=None, help="Output dir name (default: from filename).")
@click.option("--chroms", default="1-22")
@click.option(
    "--parallel", default=8, type=int, help="Number of chromosomes to run concurrently."
)
@click.option("--threads-per-chrom", default=None, type=int)
@click.option("--heap-gb", default=6, type=int)
def cli_beagle(genotype, name, chroms, parallel, threads_per_chrom, heap_gb):
    """Impute locally with Beagle 5 (no account, ~10 min on 64 cores)."""
    out = _beagle.impute(
        genotype,
        name=name,
        chroms=chroms,
        parallel=parallel,
        threads_per_chrom=threads_per_chrom,
        heap_gb=heap_gb,
    )
    click.echo(
        f"\nDone. Imputed VCFs in {out}/. Score with:\n"
        f"  genepred score {out}/all.vcf.gz --build GRCh37\n"
        f"(or per-chromosome chr*.vcf.gz)"
    )


@cli_impute.group("michigan")
def cli_michigan():
    """Michigan Imputation Server — split into submit/status/fetch
    because results require an emailed decryption password."""


@cli_michigan.command("submit")
@click.argument("genotypes", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--out-dir", type=click.Path(), required=True)
@click.option(
    "--refpanel",
    default="apps@hrc-r1.1",
    help="HRC r1.1 (default), apps@1000g-phase-3-v5, "
    "or apps@topmed-r3 (separate server).",
)
@click.option(
    "--population",
    default="mixed",
    type=click.Choice(["eur", "afr", "asn", "amr", "sas", "eas", "mixed", "AA"]),
)
@click.option(
    "--holdout-frac",
    default=0.0,
    type=float,
    help="Random fraction of sites to mask for accuracy check.",
)
def cli_mich_submit(genotypes, out_dir, refpanel, population, holdout_frac):
    """Conform, upload, and persist job state."""
    _mich.submit(
        list(genotypes),
        Path(out_dir),
        refpanel=refpanel,
        population=population,
        holdout_frac=holdout_frac,
    )


@cli_michigan.command("status")
@click.argument("out_dir", type=click.Path(exists=True))
def cli_mich_status(out_dir):
    """Poll job state for a previously-submitted Michigan job."""
    _mich.status(Path(out_dir))


@cli_michigan.command("fetch")
@click.argument("out_dir", type=click.Path(exists=True))
@click.option(
    "--password", required=True, help="The decryption password emailed to you."
)
def cli_mich_fetch(out_dir, password):
    """Download and decrypt results once the job is done."""
    files = _mich.fetch(Path(out_dir), password)
    click.echo(
        f"\n{len(files)} per-chromosome dose VCFs ready. Score with:\n"
        f"  genepred score {files[0].parent}/chr1.dose.vcf.gz "
        f"--build GRCh37"
    )


# ------------------------------------------------------------------- qaly


@main.command("qaly")
@click.option(
    "--scores",
    multiple=True,
    metavar="TRAIT=Z",
    help="e.g. --scores heart_disease=-0.5 --scores height=1.2",
)
@click.option(
    "--json-in", type=click.Path(exists=True), help="JSON file with {trait: z}."
)
@click.option(
    "--embryos",
    type=int,
    default=None,
    help="Simulate selecting best of N sibling embryos by QALY.",
)
@click.option("--no-correlations", is_flag=True)
@click.option("--only", multiple=True)
@click.option("--exclude", multiple=True)
@click.option(
    "--rate",
    default=0.0,
    type=float,
    help="Pure annual time-discount rate (default 0).",
)
@click.option("--survival/--no-survival", default=True)
@click.option(
    "--ancestry",
    default="EUR",
    type=click.Choice(list(_qaly.ANCESTRY_R2_RATIO)),
    help="Scale R² by the cross-ancestry attenuation ratio.",
)
@click.option("--json", "as_json", is_flag=True)
def cli_qaly(
    scores,
    json_in,
    embryos,
    no_correlations,
    only,
    exclude,
    rate,
    survival,
    ancestry,
    as_json,
):
    """QALY calculator: per-trait impact, or best-of-N selection gain."""
    if embryos:
        restricted = sorted(
            k for k, s in CURATED.items() if s.embryo_permitted is False
        )
        if restricted:
            click.echo(
                f"Excluding traits whose source data prohibits prenatal "
                f"prediction: {', '.join(restricted)}\n",
                err=True,
            )
        exclude = list(exclude) + restricted
        ratio = _qaly.ANCESTRY_R2_RATIO[ancestry]
        if ancestry != "EUR":
            click.echo(
                f"Ancestry={ancestry}: scaling all R² by {ratio:.2f} "
                f"(EUR-trained scores; cross-ancestry mean attenuation).\n",
                err=True,
            )
        r = _qaly.simulate_selection(
            n_embryos=embryos,
            use_correlations=not no_correlations,
            only=list(only) or None,
            exclude=exclude or None,
            rate=rate,
            use_survival=survival,
            ancestry_ratio=ratio,
        )
        click.echo(
            json.dumps(r, indent=2) if as_json else _qaly.format_selection_results(r)
        )
        return
    if json_in:
        with open(json_in) as f:
            zd = json.load(f)
    elif scores:
        zd = {s.split("=", 1)[0].strip(): float(s.split("=", 1)[1]) for s in scores}
    else:
        click.echo("Provide --scores, --json-in, or --embryos.", err=True)
        sys.exit(1)
    ratio = _qaly.ANCESTRY_R2_RATIO[ancestry]
    if ancestry != "EUR":
        click.echo(f"Ancestry={ancestry}: scaling all R² by {ratio:.2f}.\n", err=True)
    r = _qaly.compute_all(zd, rate=rate, use_survival=survival, ancestry_ratio=ratio)
    click.echo(json.dumps(r, indent=2) if as_json else _qaly.format_qaly_results(r))


@main.command("traits")
def cli_traits():
    """List all traits the QALY model knows about, with parameters."""
    click.echo("Disease traits:")
    click.echo(
        f"  {'name':<26} {'prevalence':>10} {'QALY loss':>10} "
        f"{'cost':>12} {'pop R²':>8}"
    )
    for t in _qaly.DISEASE_TRAITS.values():
        click.echo(
            f"  {t.name:<26} {t.prevalence:>9.1%} "
            f"{t.qaly_loss_if_affected:>10.1f} "
            f"${t.lifetime_cost_if_affected:>10,} {t.pgs_r2_population:>8.3f}"
        )
    click.echo("\nContinuous traits:")
    click.echo(f"  {'name':<26} {'QALY/SD':>10} {'$/SD':>12} {'pop R²':>8}")
    for t in _qaly.CONTINUOUS_TRAITS.values():
        click.echo(
            f"  {t.name:<26} {t.qaly_per_sd:>+10.3f} "
            f"${t.savings_per_sd:>+10,} {t.pgs_r2_population:>8.3f}"
        )
    click.echo("\nCurated PGS:")
    click.echo(f"  {'trait':<26} {'PGS ID':<11} {'variants':>10} {'EUR R²':>8}")
    for k, s in CURATED.items():
        click.echo(f"  {k:<26} {s.pgs_id:<11} {s.n_variants:>10,} {s.r2_eur_pop:>8.3f}")


if __name__ == "__main__":
    main()
