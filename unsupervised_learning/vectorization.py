import os

from threadpoolctl import threadpool_limits

os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")  # hard-disable CUDA in Numba
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # hide GPUs from CUDA runtime

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import (
    MinimalFCParameters,
    EfficientFCParameters,
    ComprehensiveFCParameters,
)
from tsfresh.utilities.dataframe_functions import impute as tsf_impute

class BaseVectorizer:
    """Vectorizers must expose: partial_fit(batch), transform(batch)."""
    def __init__(self):
        self.needs_two_pass = False

    def partial_fit(self, x: np.ndarray):
        """Optional fit; no-op by default to support pure-transform vectorizers."""
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TSFreshVectorizer(BaseVectorizer):
    """
    Vectorize 1D windows using TSFresh.

    Input  : x.shape == (B, L), univariate windows.
    Output : np.ndarray with one feature vector per window.

    Parameters
    ----------
    fc_params : {'minimal', 'efficient', 'comprehensive'} or tsfresh settings instance
        Which predefined TSFresh feature set to use.
    impute : bool
        Whether to impute missing values in TSFresh output.
    n_jobs : int
        Number of parallel jobs for TSFresh (0/-1 means all cores).
    """

    def __init__(
        self,
        fc_params: str | object = "efficient",
        impute: bool = True,
        n_jobs: int = 0,
    ):
        super().__init__()
        self.impute = impute
        self.n_jobs = n_jobs

        # Choose feature set
        if isinstance(fc_params, str):
            fc_params = fc_params.lower()
            if fc_params == "minimal":
                self.fc_params = MinimalFCParameters()
            elif fc_params == "efficient":
                self.fc_params = EfficientFCParameters()
            elif fc_params == "comprehensive":
                self.fc_params = ComprehensiveFCParameters()
            else:
                raise ValueError(
                    "Unknown fc_params string. Use 'minimal', 'efficient', "
                    "'comprehensive', or pass a custom settings object."
                )
        else:
            # Assume user passed a tsfresh settings object
            self.fc_params = fc_params

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(
                f"TSFreshVectorizer expects univariate windows shaped (B, L); got {x.shape}"
            )
        B, L = x.shape
        if B == 0:
            return np.empty((0, 0), dtype=float)

        # Long-format dataframe for tsfresh: columns (id, time, value)
        ids = np.repeat(np.arange(B, dtype=np.uint32), L)
        values = x.reshape(-1)

        df = pd.DataFrame(
            {"id": ids, "value": values},
            copy=False,
        )
        with threadpool_limits(limits=1):
            feats = extract_features(
                df,
                column_id="id",
                default_fc_parameters=self.fc_params,
                n_jobs=self.n_jobs,
                disable_progressbar=True,
            )


        if self.impute:
            tsf_impute(feats)

        # Ensure rows are in window order [0..B-1] and return as float ndarray
        feats = feats.sort_index()
        if feats.shape[0] != B:
            feats = feats.reindex(range(B), copy=False)

        return feats.to_numpy(dtype=float)
