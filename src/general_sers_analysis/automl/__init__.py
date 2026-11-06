"""Classical AutoML baseline; shared data access lives in core.data."""
from .search import Candidate, build_workflow, candidate_space, metrics, cv_score, split_holdout, run_automl, run_benchmark
from ..util import subset_spectra as subset

__all__ = ["Candidate", "build_workflow", "candidate_space", "run_automl", "run_benchmark"]
