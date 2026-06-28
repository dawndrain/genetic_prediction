# Embryo imputation under realistic parental phasing

`genepred embryo-demo` originally assumed the parents' haplotypes
are perfectly phased. They aren't. This note documents how badly that
assumption breaks the imputation, what to do about it, and how to
reproduce the numbers.

## TL;DR

- At a realistic switch-error rate of 1 % per consecutive het, the
  original 4-state HMM's PGS recovery drops from r² ≈ 1.0 to **r² ≈
  0.38** at 0.05× biopsy coverage — embryo selection becomes barely
  better than random.
- Pooling all embryos in a single 2^E-state HMM (`joint_recover`)
  recovers **r² ≈ 0.92** at 0.05× with 5 embryos.
- Adding **one genotyped relative** (a born sibling, or trio-phasing
  the parents from grandparents) takes any method back to r² ≈ 0.99.
- 0.005× coverage is below the useful floor for *any* method unless
  you have a relative.

## The problem

Statistical phasers (Beagle, Eagle, SHAPEIT) produce a switch error —
the hap0/hap1 labels flip — at roughly 0.3–2 % of consecutive
heterozygous pairs depending on panel size and ancestry. A parent
with ~28 k hets on chr22 at SER = 1 % has ~280 switches, versus ~1
meiotic crossover. The 4-state HMM's recombination prior (~10⁻⁸/bp)
is two orders of magnitude too tight to follow that many flips, so it
mis-tracks the inheritance path over long stretches and reconstructs
the wrong embryo genotype.

## Benchmark results

chr22, 1KG-CEU parents, 5 embryos, sequencing error 1 %, 8 replicates,
het-site genotype concordance / mean PGS r²(true, imputed) over 8
scores. Reproduce with:

```bash
python validation/embryo_phasing_bench.py --chroms 22 \
    --ser 0,0.005,0.01,0.02 --cov 0.005,0.05 --n-embryos 5 --reps 8
```

| method @ SER=1% | 0.005× cov | 0.05× cov |
|---|---|---|
| oracle (perfect phase) | 97.97 % / 0.93 | 99.86 % / 1.00 |
| **naive 4-state HMM** | 49.5 % / 0.23 | 62.8 % / **0.38** |
| switch-aware single | 52.8 % / 0.28 | 71.0 % / 0.46 |
| **joint 2^E HMM** | 64.4 % / 0.46 | **91.9 % / 0.92** |
| joint + 1 born sibling | **97.7 % / 0.93** | **99.7 % / 0.99** |

The sibling row: `--n-sibs 1 --methods oracle,naive,joint`.

Scaling with embryo count (joint method, SER=1%, het-conc):

| n_embryos | 0.005× | 0.02× | 0.05× | 0.1× |
|---|---|---|---|---|
| 3 | 60.9 % | 76.9 % | 87.5 % | 93.1 % |
| 5 | 64.7 % | 82.6 % | 91.8 % | 95.6 % |
| 8 | ~68 % | ~87 % | ~94 % | ~97 % |

```bash
python validation/embryo_phasing_bench.py --chroms 22 --ser 0.01 \
    --cov 0.005,0.02,0.05,0.1 --n-embryos 3,5,8 --reps 3
```

**Rule of thumb.** Define R = (coverage × n_embryos) / SER, the pooled
embryo reads per parental switch interval. R ≳ 50 → near-oracle;
R ≈ 25 → good (selection works); R ≲ 5 → unreliable.

## Amplification realism: allele dropout + coverage dispersion

The tables above use an idealised read model (Poisson coverage,
unbiased alleles). Real WGA biopsies add **allelic dropout** (one
allele of a het fails to amplify and all reads come from the survivor
— ~1–5 % per site for multi-cell TE biopsies with modern kits, 10–25 %
for single-cell MDA) and **amplification bias** (per-site coverage is
over-dispersed; CV ≈ 0.5–1 for MDA). Both are modelled in
`simulate_biopsy(ado=…, cov_dispersion=…)` and exposed as `--ado` /
`--cov-dispersion`. PGS r² at SER = 1 %, 0.05×, 8 reps:

| ado / cov-CV | oracle, 3 emb | joint, 3 emb | oracle, 5 emb | joint, 5 emb |
|---|---|---|---|---|
| 0 / 0 (idealised) | 1.00 | 0.85 | 1.00 | 0.90 |
| 0.05 / 0.7 (good multi-cell kit) | 1.00 | 0.76 | 1.00 | 0.89 |
| 0.10 / 0.7 | 1.00 | 0.80 | 1.00 | 0.88 |
| 0.20 / 1.0 (single-cell-grade MDA) | 0.98 | 0.74 | 0.98 | 0.87 |
| 0.20 / 1.0 **+ 1 born sibling** | 0.98 | **0.97** | — | — |

```bash
python validation/embryo_phasing_bench.py --chroms 22 --ser 0.01 \
    --cov 0.05 --n-embryos 3,5 --methods oracle,joint --reps 8 \
    --ado 0.2 --cov-dispersion 1.0 [--n-sibs 1]
```

Reading the grid:

- With **oracle phase, amplification artifacts are nearly free**
  (r² ≥ 0.98 even at single-cell-grade noise) — the HMM pools
  thousands of sites, and ADO mostly recalibrates evidence rather
  than flipping it. Hard-call het concordance for `joint` barely
  moves with ado (~87–88 % throughout); the damage shows up in the
  soft dosages.
- At **5 embryos** the joint HMM degrades gracefully: 0.90 → 0.87
  worst-case.
- At **3 embryos** amplification noise is the binding constraint:
  0.85 → ~0.74–0.80 (rep noise ±0.04), i.e. realistic artifacts cost
  another ~5–10 % of selection gain on top of the few-embryo penalty.
  Treat the rule-of-thumb R as derated ~2× for few embryos with
  single-cell-grade kits.
- **One born sibling fully rescues the worst case** (0.97 at 3
  embryos with ado 20 %, CV 1.0): phasing error, not biopsy noise,
  remains the binding constraint, and a relative fixes phasing
  regardless of amplification quality.

## Recommended fixes, in order of impact

### 1. Get a relative (best)

Any of these drives the effective SER to ≈0 and makes the simple
per-embryo HMM sufficient:

- **Grandparents on a side**: trio-phase that parent with SHAPEIT5
  `--pedigree`, listing the *parent* as the child row and the
  grandparents as father/mother (`genepred/impute/shapeit.py` wraps
  this; ~5 min/chr22 with the 1KG panel as reference). One
  grandparent works too (use `NA` for the other). SHAPEIT5's
  pedigree mode only scaffolds the *child* row — it does **not**
  back-propagate to correct founders — so this is the right tool
  iff you have the parent's parents. The bare Mendelian rule
  (`genepred.embryo.trio_phase`) resolves ~79 % of hets perfectly but the
  unfilled ~21 % cap het-conc at ~88 %; use SHAPEIT, not the bare
  rule, in production.
- **An existing child of the couple**: SHAPEIT5 will *not* use a
  child to correct the parents. Instead, append the child to the
  joint HMM as a 30× "embryo" (`validation/embryo_phasing_bench.py --n-sibs 1`).
  Three lines of code, takes het-conc from 92 % → 99.7 % at 0.05×
  and from 60 % → 98 % at 0.005×. **This is the single biggest
  lever** and needs no external binaries. The published alternative
  is duoHMM (O'Connell 2014, https://github.com/jaredo/duohmm), a
  post-processor that corrects *all* members of an arbitrary
  pedigree jointly — so it also covers the grandparent case and
  any mix of relatives, at the cost of building against Boost.
- **A sibling of a parent (aunt/uncle)**: duoHMM handles this via
  the pedigree; SHAPEIT5's pedigree mode does not.

The 1KG phase-3 main release we ship is unrelated-only by design, so
the default test parents (NA06984/NA06985) have no relatives in it.
Real trio members (e.g. NA12878's parents NA12891/NA12892) are in the
separate `related_samples` VCFs — see `genepred/impute/shapeit.py` for
the download path.

### 2. Long-read or Strand-seq the parents

PacBio HiFi / Nanopore at ~30× gives read-backed phase blocks of
5–50 Mb (SER ≪ 0.1 %), at which point R ≫ 50 even for a single
embryo at 0.01×. WhatsHap (`--ped` for hybrid trio+read phasing) is the
standard tool. This is what Orchid does.

### 3. Bigger reference panel (purely statistical)

If no relatives and no long reads, phase the parents against the
largest panel you can:

| panel | size | EUR SER | how |
|---|---|---|---|
| 1KG | ~2.5 k | ~1 % | already wired (`genepred/impute/beagle.py`) |
| HRC | ~32 k | ~0.3 % | Michigan Imputation Server (already wired) |
| TOPMed | ~97 k | ~0.1 % | `genepred impute michigan submit --server topmed` (needs a BioData Catalyst token) |
| UK Biobank | ~500 k | <0.1 % | approved-application only |

At TOPMed-class SER (~0.1 %), 5 embryos at 0.05× gives R ≈ 250 and
even the naive HMM is fine; non-EUR ancestry sees less benefit
because panel coverage is thinner.

### 4. Algorithmic — `joint_recover`

If you're stuck with statistical phasing and no relatives, run
`genepred.embryo.joint_recover` over all embryos instead of `hmm_recover` per
embryo. It models each parent's switch track as shared across embryos
(rate `switch_rate`) and each embryo's recombination as independent
(rate `recomb_per_bp`), runs forward-backward over the 2^E joint
state, and returns posterior dosage with per-site variance. State
space is 2^E so it's practical to ~10–12 embryos; the function will
warn at E>8 and refuse at E>12. The demo's `--method joint` flag
selects it.

## Real-data check

Running SHAPEIT5 `--pedigree` on the PUR trio (HG00731 × HG00732 →
HG00733; child from the 1KG `related_samples` VCF) and comparing the
child's transmitted haplotype to the parents' 1KG-published phase
gives a direct measurement of that phase's switch-error rate:

| parent | het sites (chr22) | transitions | implied SER |
|---|---|---|---|
| HG00731 | 31,181 | 594 | ~1.9 % |
| HG00732 | 33,448 | 529 | ~1.6 % |

About half of these are point flips (median inter-switch gap = 2
hets) rather than long-range switches; the long-range component is
roughly 0.5–1 %. PUR is admixed, so this is on the high side; EUR
samples with the same panel land closer to 0.5 %. Either way, the
1KG phase that the original demo treated as ground truth is squarely
in the regime where the naive HMM fails.

Reproduce (~5 min on chr22; downloads the related-samples VCF and
runs SHAPEIT5 — needs `tools/shapeit5/phase_common_static`):

```bash
python validation/embryo_phasing_validate.py --trio PUR --chrom 22
```

Other trios: `--trio YRI` (NA19240), `KHV`, `CHS`, `MXL`.

## Caveats on the simulation

The error model is simplified: switch errors are i.i.d. per het pair
(the real-data check above shows ~half are actually single-site point
flips, and the rest cluster in low-LD regions) and parents are
perfectly genotyped. Whole-genome-amplification artefacts — allelic
dropout and coverage over-dispersion — are now modelled (`--ado`,
`--cov-dispersion` on the bench/demo); at MDA-like settings
(ADO = 0.15, CV = 0.7) they move joint-recovery het-conc by < 1 pp at
0.05× because they're per-site noise the HMM smooths over, unlike
the long-range phasing switches that dominate the error budget.

## References

- Handyside et al. 2010, *J Med Genet* — Karyomapping
- Zamani Esteki et al. 2015, *AJHG* — Haplarithmisis
- Kumar et al. 2015, *Genome Med* — embryo WGS reconstruction with relatives
- Backenroth et al. 2019, *Genet Med* — Haploseek (per-embryo HMM)
- Masset et al. 2022, *NAR* — Hopla (multi-embryo linkage)
- O'Connell et al. 2014, *PLoS Genet* — duoHMM
- Hofmeister et al. 2023, *Nat Genet* — SHAPEIT5
