"""Classical spectral tools. Workflow execution is available; automatic workflow search is not implemented."""
from .core import Node, Edge, Condition, Spectra, UnmixingResult

__version__ = "0.1.0"
__all__ = ["Node", "Edge", "Condition", "Spectra", "UnmixingResult", "Workflow", "NodeSpec"]

from .workflow import Workflow, NodeSpec
