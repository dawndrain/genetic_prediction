"""genepred — polygenic score pipeline + embryo-selection QALY model.

Public surface:
    genepred.io        load DTC/VCF genomes, write conformed VCFs
    genepred.pca       project a genome onto 1KG ancestry PCs
    genepred.scoring   compute raw PGS, normalize to z, PC-adjust
    genepred.catalog   curated PGS list + on-demand download
    genepred.impute    Beagle (local) and Michigan (submit/fetch) backends
    genepred.qaly      QALY model parameters and calculator
    genepred.embryo    meiosis simulation + selection
    genepred.report    HTML report generation

CLI entry point: `genepred` (see cli.py).
"""

from genepred.catalog import CURATED, Score  # noqa: F401
from genepred.scoring import ScoreResult, score_genome  # noqa: F401

__version__ = "0.1.0"
