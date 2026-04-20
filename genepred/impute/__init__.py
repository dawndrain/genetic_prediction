"""Imputation backends.

`beagle` is the default — fully local, ~8–12 min genome-wide on 64 cores
against 1KG Phase 3, no external account needed.

`michigan` is a two-phase submit/fetch flow because the server emails a
decryption password. Job state is persisted so a half-finished run is
resumable.
"""

from genepred.impute.beagle import impute as beagle_impute
from genepred.impute.michigan import (
    fetch as michigan_fetch,
)
from genepred.impute.michigan import (
    status as michigan_status,
)
from genepred.impute.michigan import (
    submit as michigan_submit,
)

__all__ = [
    "beagle_impute",
    "michigan_fetch",
    "michigan_status",
    "michigan_submit",
]
