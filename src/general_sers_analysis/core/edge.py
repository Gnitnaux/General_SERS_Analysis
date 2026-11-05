"""Connection descriptions consumed by the Workflow executor and graph renderer."""
from dataclasses import dataclass
from typing import Any, Literal
import operator
import numpy as np


@dataclass(frozen=True)
class Condition:
    field: str
    operation: Literal["eq", "ne", "gt", "ge", "lt", "le"]
    value: Any

    def __post_init__(self):
        if not self.field or self.operation not in {"eq", "ne", "gt", "ge", "lt", "le"}:
            raise ValueError("Condition needs a field and a supported comparison.")

    def evaluate(self, context):
        """Return a boolean or sample mask; never evaluate executable strings."""
        if self.field not in context:
            raise KeyError(self.field)
        return getattr(operator, self.operation)(np.asarray(context[self.field]), self.value)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    source_port: str = "output"
    target_port: str = "input"
    data_kind: str | None = None
    condition: Condition | None = None

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in (self.source, self.target, self.source_port, self.target_port)):
            raise ValueError("Node IDs and port names must be nonempty strings.")
        if self.source == self.target:
            raise ValueError("Self-connections are not supported.")
