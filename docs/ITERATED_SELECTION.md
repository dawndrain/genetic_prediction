# Iterated selection from one couple: simulation results

What bounds the gain from repeatedly selecting on a polygenic score
within one couple's genetic material — and does iterated *meiosis*
(optimise each parent's haploid in vitro, then combine) avoid the
constraints that break iterated *embryo* selection?

Simulator: `examples/iterated_selection.py`. All numbers below are
in population-reference SD units (computed from a large independent
draw, so σ is comparable across modes and founding-population
sizes).

## Model

- **Trait**: 2 000 additive loci, h² = 0.5, drawn from a realistic
  allele-frequency spectrum. True phenotype y = 4·tanh(g/4) + ε, so
  the linear PGS g is the local tangent and the realised trait
  saturates near +4σ.
- **Recessive load**: 5 000 loci, ~80 het deleterious per founder,
  s_rec ~ Gamma(shape 0.2, mean 0.03) with a 2 % lethal tail —
  roughly the human ~1 lethal-equivalent estimate. Fitness =
  exp(−Σ s_rec over hom-deleterious loci). The PGS cannot see this.
- **Crossover**: 23 chromosomes, Poisson(1.5) crossovers per
  chromosome per meiosis. `--co-per-chrom inf` recovers per-locus
  independent assortment (the unrealistic ceiling).
- **Per-locus ceiling** (best allele at every het site from one
  couple's four founding haplotypes): **+20.6σ**. Reachable only in
  the independent-assortment limit; under realistic crossover you
  are choosing among ~58 chromosome segments per round, not 2 000
  loci.

## Iterated embryo selection: inbreeding is the binding constraint

Each generation: random mating within the population, n_embryos = 10
per couple, keep the top-PGS embryo, repeat. 30 generations, 4
replicates.

| gen | N=2 PGS / fitness | N=4 | N=10 | N=50 |
|---|---|---|---|---|
| 0 | −0.6 / 0.99 | −0.3 / 0.99 | +0.0 / 0.99 | −0.1 / 0.98 |
| 6 | +4.6 / **0.29** | +5.3 / 0.50 | +6.1 / 0.67 | +6.1 / 0.92 |
| 12 | +6.3 / 0.22 | +8.4 / 0.42 | +10.7 / 0.50 | +11.6 / 0.88 |
| 21 | **+6.9** stuck / 0.14 | +10.5 / 0.34 | +14.8 / 0.37 | +18.6 / 0.75 |
| 30 | +6.9 / **0.14** | +10.9 / 0.30 | +17.1 / 0.35 | +24.2 / 0.71 |

At **N = 2** (one couple, sib-sib mating from gen 2), the inbreeding
coefficient F → 0.99 and fitness collapses to 0.14 by gen 30; the
PGS plateaus at +6.9σ once heterozygosity is exhausted. At
**N = 50**, fitness holds at ~0.7 and the score keeps climbing past
+24σ — but that is a population-breeding scenario drawing on 100
founding haplotypes, not one couple. In every case the *realised
trait* saturates near +4σ regardless of how far the score goes.

## Iterated meiosis: no inbreeding, scheme determines the ceiling

Per-parent: repeatedly induce meiosis, score the haploid products,
fuse selected pairs back to diploid, repeat. The intermediate
diploid cell cultures become homozygous, but they are cultures, not
organisms — recessive load is mostly developmental, not
cell-autonomous. The final embryo is dad's-optimised-haploid ⊕
mom's-optimised-haploid: heterozygous everywhere the parents
differ, **F ≈ 0**.

25–30 rounds, 10 gametes/round, pool size 4 where applicable, 4
replicates, realistic crossover:

| scheme | what it does | embryo PGS | % of +20.6σ ceiling | realised trait | fitness | F |
|---|---|---|---|---|---|---|
| `top2` | fuse the two best gametes (greedy) | +3.0σ | 15 % | +2.4σ | 0.99 | 0.003 |
| `pool` | keep 4 diploids, fuse random top-8 | +6.7σ | 32 % | +3.7σ | 0.98 | 0.013 |
| `useful` | pool, pair by μ + i·σ (Schnell usefulness) | +7.9σ | 38 % | +3.8σ | 1.00 | 0.015 |
| `ohv` | pool, pair by optimal-haploid value | **+9.0σ** | **44 %** | **+3.9σ** | 0.99 | 0.021 |
| `ohv`, indep. assortment | (linkage removed) | +13.0σ | 63 % | +4.0σ | 0.99 | 0.045 |

Per-round convergence (haploid PGS, in haploid-SD; embryo ≈ √2 × this):

| round | top2 | pool | useful | ohv |
|---|---|---|---|---|
| 0 | +1.02 | +1.51 | +1.46 | +1.46 |
| 3 | +2.00 | +2.86 | +3.08 | +2.72 |
| 6 | **+2.11** ← plateau | +3.73 | +4.09 | +3.66 |
| 12 | +2.11 | +4.56 | +5.17 | +5.10 |
| 18 | +2.11 | +4.71 | +5.53 | +5.96 |
| 24 | +2.11 | **+4.72** ← plateau | +5.59 | +6.44 |
| 29 | +2.11 | +4.72 | **+5.60** ← plateau | **+6.66** still climbing |

`top2` collapses heterozygosity by round ~6 and never recovers.
`pool` (random pairing) holds out to ~21. `useful` and `ohv` are
explicit explore/exploit schemes: `ohv` is slower in the first ~10
rounds (it deliberately keeps diversity, exploiting less per round)
but overtakes around round 12 and is still gaining at round 30.

## Takeaways

- **For one couple, iterated meiosis with OHV (+9.0σ, fitness 0.99,
  F ≈ 0) strictly dominates iterated embryo selection (+6.9σ,
  fitness 0.14, F → 1).** The inbreeding constraint is removed
  entirely; what remains is the linkage-limited ceiling and trait
  saturation.
- **Greedy is far from the ceiling.** `top2` reaches 15 %; OHV
  reaches 44 %; the gap between OHV and the per-locus optimum is
  almost entirely linkage (independent assortment closes ~half of
  it). Larger pools and more rounds would close more.
- **The realised-trait gain saturates well before the PGS does.**
  OHV at +9σ on the score is already at +3.9σ on the trait against
  a +4σ ceiling — 97 % of the realisable gain. Past ~+8σ on the
  score, additional score gain is wasted selection differential.
  This is the "frozen linear PGS extrapolates past where it was
  trained" failure mode, separate from inbreeding.
- **Scheme ranking matches the breeding literature**: greedy <
  pool ≈ OCS-style < usefulness < OHV (one-step look-ahead). Full
  multi-generation look-ahead would add another ~10–15 % over OHV
  in expectation, but in practice OCS-class methods are what's
  deployed because the planning gain is within EBV estimation
  noise.
- **Iterated meiosis is not demonstrated in any organism.** The
  loop has a type mismatch — meiosis outputs a terminally
  differentiated gamete; the next round needs a meiosis-competent
  precursor. Closing it (probably via haploid-ESC fusion + induced
  meiosis) is an open ~3–5-year mouse-scale problem.

## Reproduce

```bash
# meiosis-mode scheme comparison
python examples/iterated_selection.py --meiosis \
    --n-founders 20 --n-gens 30 --n-embryos 10 --reps 4 \
    --co-per-chrom 1.5

# embryo-selection mode, founding-population sweep
python examples/iterated_selection.py \
    --n-founders 2,4,10,50 --n-gens 30 --n-embryos 10 --reps 4

# linkage-free ceiling
python examples/iterated_selection.py --meiosis --scheme ohv \
    --n-founders 20 --n-gens 30 --co-per-chrom inf
```

Knobs: `--sat-scale` (where the trait saturates), `--no-load`
(disable recessive load), `--fitness-filter` (cull on realised
fitness each generation — the livestock safeguard),
`--co-per-chrom`, `--scheme {top2,pool,useful,ohv,top+rand,top+far}`.
