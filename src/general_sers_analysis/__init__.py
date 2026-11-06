"""Classical spectral tools, reusable DAG execution and blocked-CV workflow search."""
from .core import Node, Edge, Condition, Spectra, UnmixingResult

__version__ = "0.1.0"
__all__ = ["Node", "Edge", "Condition", "Spectra", "UnmixingResult", "Workflow", "NodeSpec"]

from .workflow import Workflow, NodeSpec
