"""Connection descriptions consumed by the Workflow executor and graph renderer."""
from dataclasses import dataclass
from typing import Any, Literal
import operator
import numpy as np


@dataclass(frozen=True)
class Condition:
    """Structured field comparison without executable strings; conditional Workflow routing is not implemented.

    Attributes
    ----------
    field: Name of the context field to compare.
    operation: Comparison name: eq, ne, gt, ge, lt or le.
    value: Comparison threshold or reference value.
    """
    field: str
    operation: Literal["eq", "ne", "gt", "ge", "lt", "le"]
    value: Any

    def __post_init__(self):
        """Validate aligned members and normalize their internal representation.

        Returns and state
        -----------------
        None. Normalizes members or rejects invalid data.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        if not self.field or self.operation not in {"eq", "ne", "gt", "ge", "lt", "le"}:
            raise ValueError("Condition needs a field and a supported comparison.")

    def evaluate(self, context):
        """Compare a context field safely, or execute Workflow evaluation nodes without refitting.

        Parameters
        ----------
        context: Mapping from field names to numeric values or arrays.

        Returns and state
        -----------------
        Boolean comparison result, or a Workflow mapping of evaluation node outputs.

        Raises
        ------
        KeyError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        if self.field not in context:
            raise KeyError(self.field)
        return getattr(operator, self.operation)(np.asarray(context[self.field]), self.value)


@dataclass(frozen=True)
class Edge:
    """Directed node/port connection declaration; does not itself process data.

    Attributes
    ----------
    condition: Optional Condition declaration; the current Workflow rejects conditional routing.
    data_kind: Optional coarse input-type hint; does not validate chemical units or identities.
    source: Upstream node ID; input denotes external inputs.
    source_port: Upstream port name; tool nodes currently expose only output.
    target: Downstream node ID.
    target_port: Destination argument: input binds X; fit:y and other fit: ports are training-only.
    """
    source: str
    target: str
    source_port: str = "output"
    target_port: str = "input"
    data_kind: str | None = None
    condition: Condition | None = None

    def __post_init__(self):
        """Validate aligned members and normalize their internal representation.

        Returns and state
        -----------------
        None. Normalizes members or rejects invalid data.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        if not all(isinstance(v, str) and v for v in (self.source, self.target, self.source_port, self.target_port)):
            raise ValueError("Node IDs and port names must be nonempty strings.")
        if self.source == self.target:
            raise ValueError("Self-connections are not supported.")
