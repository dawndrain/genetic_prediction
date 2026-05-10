# Notes: PGS limits, rare variants, iterated selection

Working notes from conversations about where the linear polygenic
model holds, where it doesn't, and what that implies for embryo
selection and editing. Not polished; pointers to literature where
they exist.

## How far does a frozen genomic predictor extrapolate?

The cleanest empirical handle is **dairy cattle genomic selection**
(Meuwissen-Hayes-Goddard 2001; deployed 2009). Young bulls are
selected on a genomic breeding value before any daughters have
records.

- The 2009-era GEBVs correlated ~0.6–0.7 with the bulls' eventual
  daughter-proven EBVs (vs ~0.30–0.35 for pedigree-only). Modern
  GEBVs reach ~0.75–0.85 for milk yield. (García-Ruiz et al. 2016
  PNAS shows the rate of genetic gain roughly doubled.)
- This is *higher* than the best human PGS for any trait (height
  r≈0.55, EA r≈0.35). Reasons are structural: each bull's
  "phenotype" is a daughter average with reliability →1; Holstein
  Nₑ≈100 so LD blocks are huge and Mₑ is small; training and target
  populations are the same closed herd.
- **Decay under continued selection without retraining**: studies
  (Lourenco, Hidalgo) suggest reliability falls ~5–10 % per
  generation if the model is frozen — allele frequencies drift,
  the new cohort is several generations from the training set. In
  practice the model is retrained continuously so this doesn't
  bite. This is the closest empirical answer to "how fast does a
  frozen PGS go stale under selection."

Why bulls need so few genomes: from r²_PGS ≈ h²/(1 + Mₑ/(n·h²)),
the daughter-averaging takes effective h² from ~0.3 to ~0.9 (≈3×
sample saving) and small Nₑ takes Mₑ from ~60 k (human) to a few
thousand (≈10–20× saving). Net: 30 k bulls ≈ a million human
genomes.

## Long-term phenotypic selection goes much further than you'd expect

The Illinois corn oil/protein experiment (1896–present, >120
generations) pushed oil content from ~5 % to >20 % — roughly 30 SD
beyond the founders — with a still-approximately-linear response.
Drosophila bristle, mouse body size, broiler chicken weight: similar.

The response is sustained by **new mutational variance** (Vₘ ≈
10⁻³·Vₑ per generation); after ~50 generations the standing
variation from the founders is exhausted and de novo mutation is
doing the work (Walsh & Lynch ch. 26; Dudley & Lambert on the
Illinois data).

Two safeguards phenotypic selection has that PGS-based embryo
selection lacks: (a) lethal/sterile combinations are filtered every
generation because you breed from realised phenotype; (b) the
mutational input. Embryo selection within one couple has neither.

## When does the linear PGS model break?

Within ±3 SD of the training population: empirically excellent.
Dominance variance is ~5 % of additive for most traits; detectable
epistasis is smaller (Hivert 2021; Hill-Goddard-Visscher 2008).
Fisher's argument: GWAS estimates the *marginal* allele effect
averaged over the population background; summing those is a
first-order Taylor expansion of whatever the true map is, accurate
near the mean.

Single-mechanism traits (LDL, IGF-1→height): replace linear with
PGS → intermediate (linear) → phenotype (known dose-response).
Lets you extrapolate further.

Far tail (10σ): the right argument is statistical — the model has
no support there, refuse to extrapolate. Biological reasons it
*would* saturate stack on top: hard physical bounds (height <
~2.7 m, LDL > ~20 mg/dL, sleep > ~3 h), canalisation, antagonistic
pleiotropy, and the selection-shadow argument (if +10σ were
unambiguously good, the population mean would already be there).

For embryo selection among siblings (±1 SD window) none of this
matters. It matters for iterated selection and editing.

## Rare large-effect protective variants

No canonical list. Church's is at
https://arep.med.harvard.edu/gmc/protect.html (started 2011 with
PCSK9/LRP5/MSTN/CCR5). Systematic source: genebass.org filtered to
pLoF burden, protective direction, p<5e-8.

~40 variants across ~30 genes at the well-replicated tier; mostly
LoF in genes that were adaptive in the ancestral environment and are
now maladaptive (PCSK9, ANGPTL3, SLC30A8 — "diseases of mismatch").
Genuine new-capability variants (FNSS short-sleep, MSTN, LRP5-GoF)
are rarer because most traits are buffered and few have a
single-gene bottleneck.

GWAS detection threshold is roughly β > c/√(n·p(1−p)), so at MAF
0.001 % even UKB-scale n misses everything; these are found by
family linkage or exome LoF-burden tests. And phenotype availability
gates everything — there's no deadlift PGS because no biobank
measures it.

Curated in `genepred/rare_variants.py` (PROTECTIVE, PATHOGENIC_GENES,
CARRIER_GENES, LPA_RISK) and `genepred/pharmgx.py` (CPIC Level-A).

## Fine-mapping, causal-SNP fraction, and editing

PGS r² doesn't depend on knowing the causal SNP — tags work for
prediction. But editing needs the causal base. Currently ~10–20 %
of GWAS loci have a single PIP>0.9 SNP (SuSiE/PolyFun on UKB).
Getting to ~50 % is plausible in 5–10 years for biomarker traits;
the dominant lever is **multi-ancestry data** (different LD breaks
ties), then functional priors, then sample size. African-ancestry
data is disproportionately valuable because LD blocks are 3–5×
shorter (no out-of-Africa bottleneck → larger ancestral Nₑ).

So the *editable* fraction of an r²=0.5 predictor today is roughly
0.05–0.15, skewed toward the large-effect loci which are both
easiest to fine-map and contribute most to r².

## Cross-species candidates

Convergent cancer resistance, divergent mechanisms (Peto's paradox):
elephants — extra TP53 retrogenes (Abegglen 2015); bowhead whales —
DNA-repair gene duplications, CIRBP (Keane 2015, Firsanov 2023);
naked mole rats — high-MW hyaluronan via HAS2 (Tian 2013;
mouse-transgenic lifespan +4 % Zhang 2023). Tardigrade Dsup
(radiation), wood-frog AFPs (cryopreservation). All cell-culture or
single-mouse-line stage; decades from human germline.

## ML for genomics — where it actually helps

For common-variant PGS, model class is not the bottleneck — sample
size and phenotype quality are. Linear with good priors
(SBayesRC/LDpred-funct) is near the ceiling. Functional priors from
sequence models (Enformer, Sei, AlphaMissense) give ~0–15 % relative
r² gain, more for biomarker/autoimmune traits, ~0 for height/EA.

Where ML earns its keep: variant-effect prediction for editing
(AlphaFold/ESM, AlphaMissense, EVE), guide-RNA design, and
**phenotype extraction from EHR text** — LLMs reading clinical
notes could supply phenotypes biobanks don't measure (pain
threshold, sleep need, healing rate, personality observations).
Demonstrated on MIMIC/single-site cohorts; not yet routine in
biobank GWAS.

Untried idea worth testing: pretrain a genome LM (Nucleotide
Transformer / Evo) on raw sequence, finetune on PGS weights for
LD-independent 90 % of the genome, predict held-out 10 %. If
held-out weights are predictable above chance, the score encodes
learnable sequence-level structure rather than pure LD-tagging.
Expect it to work on autoimmune/lipid traits, not on EA.

## Iterated meiosis vs iterated embryo selection

**Iterated embryo selection** from one couple breaks at generation 2
from inbreeding: each parent carries ~0.6–1.5 lethal-equivalents of
recessive load, and at F=0.25 (sib-sib mating) the expected fitness
cost is exp(−0.25·B) ≈ 15–30 %. The additive PGS, fit on outbred
data, can't see any of this. Simulation: at N_founders=2 over 15
generations, fitness 0.99 → 0.12, F → 0.96; at N_founders=500,
fitness stays at 0.96 and saturation of the trait (logistic ceiling
at +4σ) becomes the binding constraint instead.

**Iterated meiosis** (Metacelsus, "Meiosis is all you need")
sidesteps the inbreeding problem entirely: optimise each parent's
haploid separately by repeated meiosis-select-fuse cycles, then
combine the two optimised haploids into one embryo. The intermediate
diploid cell cultures during iteration are F→1 within that parent's
genome, but they're cultures, not organisms — most recessive lethals
are developmental, not cell-autonomous. The final embryo is
heterozygous everywhere the parents differ (F≈0); the only exposed
recessive load is at loci where both parents happen to carry the
same allele, same as a natural conception.

Simulation results (`examples/iterated_selection.py --meiosis`,
realistic ~1.5 crossovers/chromosome, 25 rounds, 10 gametes/round):

| scheme | embryo PGS | % of per-locus ceiling (+20.6σ) | fitness | F |
|---|---|---|---|---|
| top2 (greedy) | +3.0σ | 15 % | 0.99 | 0.003 |
| top+rand / top+far | +2.2–2.7σ | 11–13 % | 0.99 | ~0 |
| pool (4 diploids) | +6.7σ | 32 % | 0.98 | 0.013 |
| pool, indep. assortment | +10.6σ | 51 % | 0.98 | 0.03 |

Greedy collapses heterozygosity by round ~8. Single-diploid
"diversity-preserving" variants are *worse* — pairing the best
gamete with a random/distant one undoes the selection gain. A small
pool of diploids is what works. Even pool reaches only ~32 % of the
per-locus ceiling under realistic crossover, because linkage means
you're choosing among ~58 chromosome segments per round, not 2 000
loci independently.

**Practical status**: not demonstrated even in mouse. The pieces
exist (in-vitro meiosis, haploid ESCs, cell fusion) but the loop has
a type mismatch — meiosis outputs a terminally-differentiated
gamete, but the next round needs a meiosis-competent germ-cell
precursor. Closing it requires either reprogramming the fused
diploid back to PGCLC each round (slow), inducing meiosis directly
in fused haploid-ESCs (unsolved), or a synthetic mitotic-recombination
system. Estimated 3–5 years of focused effort (~5–10 people, ~$10–20M)
for a mouse demonstration. Funding gravity points at IVG (clinical
pull = make eggs) rather than this.

**Chromosome-level selection** (pick the best copy of each chromosome
independently, then assemble) gives ~√23 ≈ 4.8× the selection
differential of whole-gamete selection. MMCT moves single chromosomes
between cells (used for trisomy models); full 20-chromosome assembly
in mouse is 10+ years out. The "select among 10⁴ gametes" half-measure
only works if gametes are expandable haploid cell lines (sequence an
aliquot, keep the rest) — actual sperm/eggs can't be sequenced
non-destructively, except eggs via polar-body biopsy.

## Selection algorithms beyond greedy

What livestock breeders actually do: **Optimal Contribution
Selection** — max **c**ᵀĝ s.t. ½**c**ᵀ**A****c** ≤ ΔF_target,
solved as a QP each generation (Meuwissen 1997; software EVA,
AlphaMate, optiSel). The Lagrangian gives "PGS − λ·inbreeding."
Then mate-allocation pairs contributors to minimise per-cross
inbreeding and avoid known recessive carrier×carrier crosses.
Target ΔF is typically 0.5–1 %/generation. Look-ahead methods
exist (Moeinizade 2019) but aren't deployed; per-generation greedy
with the constraint captures most of the value.

For iterated meiosis the analogous constraint is heterozygosity
preservation (variance to select on next round), not inbreeding.
Two upgrades over the greedy/pool schemes:

- **Usefulness criterion** (Schnell 1983): score a diploid by
  μ + i·σ — its expected gamete mean *plus* the selection
  differential its remaining heterozygosity allows. Pure
  explore/exploit trade-off; greedy `top2` is pure exploit.
- **OHV selection** (Daetwyler 2015): score a diploid by its
  *optimal haploid value* — the best gamete it could in principle
  produce (per-chromosome max of its two homologs). One-step
  look-ahead; ~5–15 % more long-term gain in plant-breeding sims.

Both implemented in `examples/iterated_selection.py`.

## Most plausible routes to strong selection/editing by ~2035

Almost certain: better PGT-P (PGS r² ~doubles, still bounded at
~0.5σ gain/embryo, and bounded by IVF uptake ~2–10 % of births);
WGS rare-variant deselection in embryos; sanctioned monogenic
germline correction somewhere.

Plausible: IVG → best-of-100 selection (~1.4× more gain than
best-of-10); **multiplex editing in expandable germline stem cells**
(SSCs or IVG-oogonia) — install ~20–50 validated protective/causal
variants, sequence the clone to verify before use. ~30 % by 2035.
The single highest-leverage technical unlock is reliable human
germline-stem-cell culture, since #4–6 all route through it.

Long shots in humans (mouse demos plausible): iterated meiosis,
chromosome selection, iterated embryo selection.

IVG funding/effort: ~15–25 academic labs (Saitou, Hayashi, Clark,
Surani, Hanna, Orwig), ~6–8 startups (Conception, Gameto, Ivy
Natal, Dioseve, Ovelle), ~$300–500M total deployed. Egg side gets
~80 % of attention.

## Where AI helps the wet-lab process

Ranked by leverage (the constraint is *which* experiment to run,
not running it faster): literature synthesis that surfaces hidden
protocol variables across labs; multi-modal result interpretation
(the experienced-PI gut); next-experiment proposal (Bayesian DOE +
lateral analogies); video QC of bench technique to make operator
variability legible; near-zero-cost structured negative-result
capture. Image analysis is already mature.

Dexterity: routine culture is learnable in weeks (consistency, not
peak skill, is the ceiling); micro-dissection / microinjection /
patch-clamp are surgery-adjacent and have a real talent component.
Top-down (start from a working in-vivo system, ablate) beats
bottom-up (defined factors from scratch) when the readout is noisy
— you need a high-baseline positive control to distinguish signal
from variance, otherwise you're hill-climbing on sand.

## Iterated embryo selection — open questions

What to simulate: founding population, true phenotype y = f(g) + ε
with f saturating, PGS = linear fit on a training sample, each
generation pick top-PGS embryo per couple. Track when realised gain
decouples from predicted; vary frozen-vs-retrained PGS, with/without
phenotypic culling, with/without de novo mutation.

**Inbreeding is the immediate constraint for small founding
populations.** With one couple, gen-2 requires sib-sib mating
(F=0.25); each parent carries ~50–100 recessive LoF variants
(~0.6–1.5 "lethal equivalents" in humans), so homozygosity exposes
them fast. The additive PGS can't see this at all — recessive
effects are invisible to a model fit on outbred data. So iterated
selection from a single couple breaks at generation 2 for reasons
orthogonal to the saturating-f(g) story. Realistic versions need
either a population of many founding couples (the livestock model),
outcrossing each generation, or in-vitro gametogenesis tricks that
reshuffle the same 4 grandparental haplotypes without mating — and
even then the variance you can select on is bounded by what the
founders carried.
