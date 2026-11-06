"""Typed call bindings and a deterministic DAG executor for classical spectral tools."""
from dataclasses import dataclass, asdict
import inspect
import time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from .core import Node, Edge, Spectra


@dataclass
class NodeSpec:
    tool: Node
    method: str = "transform"
    stages: tuple = ("fit", "predict", "evaluate")
    map_samples: bool = False
    stack_output: bool = False


class Workflow:
    """External inputs are named ports of `input`; tool outputs use the `output` port.

    target_port=input binds X; fit:<name> binds only the fit call, while other ports
    bind named arguments of transform/predict. Metadata is never silently broadcast.
    """
    def __init__(self, nodes, edges, output_node):
        self.specs = dict(nodes)
        self.edges = list(edges)
        self.output_node = output_node
        self._validate()
        self.is_fitted_ = False

    def _validate(self):
        if "input" in self.specs or self.output_node not in self.specs:
            raise ValueError("input is reserved; output_node must reference a tool.")
        for key, spec in self.specs.items():
            if not isinstance(spec, NodeSpec) or not isinstance(spec.tool, Node):
                raise TypeError(f"{key}: expected NodeSpec containing a Node.")
            if spec.method not in {"transform", "predict", "predict_proba", "predict_scores"}:
                raise ValueError(f"{key}: unsupported output method {spec.method}.")
            if not spec.stages or not set(spec.stages) <= {"fit", "predict", "evaluate"}:
                raise ValueError(f"{key}: invalid stages.")
            if spec.tool.requires_fit and set(spec.stages) != {"fit", "predict", "evaluate"}:
                raise ValueError(f"{key}: learned nodes must participate in all stages.")
            if spec.map_samples and spec.tool.requires_fit:
                raise ValueError("Per-sample mapping is restricted to stateless transforms.")
            if spec.stack_output and not spec.map_samples:
                raise ValueError("stack_output requires map_samples.")
        seen = set()
        parents = {key: set() for key in self.specs}
        for edge in self.edges:
            if not isinstance(edge, Edge) or edge.target not in self.specs:
                raise ValueError("Every edge must target a declared tool.")
            if edge.source != "input" and edge.source not in self.specs:
                raise ValueError(f"Unknown edge source: {edge.source}")
            if edge.condition is not None:
                raise ValueError("Conditional routing is not supported by this executor yet.")
            if edge.source != "input" and edge.source_port != "output":
                raise ValueError("Tool nodes currently expose only source_port=output.")
            if edge.target_port == "X":
                raise ValueError("Use target_port=input for the main X argument.")
            if edge.target_port.startswith("fit:") and not self.specs[edge.target].tool.requires_fit:
                raise ValueError("fit: ports require a learned node.")
            key = (edge.target, edge.target_port)
            if key in seen:
                raise ValueError(f"Multiple edges target the same port: {key}")
            seen.add(key)
            if edge.source != "input":
                parents[edge.target].add(edge.source)
                if not edge.target_port.startswith("fit:") and not set(self.specs[edge.target].stages) <= set(self.specs[edge.source].stages):
                    raise ValueError("A consumer cannot run in a stage where its source is absent.")
        for key in self.specs:
            if (key, "input") not in seen:
                raise ValueError(f"{key}: missing input edge.")
            if (key, "fit:y") in seen and (key, "y") in seen:
                raise ValueError("Use only fit:y for supervised training labels.")
        remaining = set(self.specs)
        self.order = []
        while remaining:
            ready = [key for key in self.specs if key in remaining and not (parents[key] & remaining)]
            if not ready:
                raise ValueError("Workflow contains a directed cycle.")
            self.order.extend(ready)
            remaining.difference_update(ready)
        ancestors = {}
        for key in self.order:
            ancestors[key] = set(parents[key])
            for parent in parents[key]:
                ancestors[key].update(ancestors[parent])
            if self.specs[key].tool.requires_fit:
                if any(self.specs[a].method.startswith("predict") for a in ancestors[key]):
                    raise ValueError("Learning downstream of predictions needs out-of-fold training, not implemented yet.")
        # Check argument names against the real methods before execution.
        for key, spec in self.specs.items():
            for stage in spec.stages:
                for action in (["fit", spec.method] if stage == "fit" and spec.tool.requires_fit else [spec.method]):
                    names = {"X"}
                    for edge in self.edges:
                        if edge.target != key or edge.target_port == "input": continue
                        port = edge.target_port
                        if port.startswith("fit:"):
                            if action == "fit": names.add(port[4:])
                        else:
                            names.add(port)
                    try:
                        inspect.signature(getattr(spec.tool, action)).bind(**dict.fromkeys(names))
                    except (TypeError, AttributeError) as error:
                        raise ValueError(f"{key}.{action}: incompatible edge ports: {error}") from error

    @staticmethod
    def _kind(value, kind):
        if kind is None: return
        if kind == "spectra":
            valid = isinstance(value, Spectra) or (isinstance(value, (list, tuple)) and len(value) > 0
                                                    and all(isinstance(item, Spectra) for item in value))
        elif kind in {"matrix", "features", "target_values", "class_probabilities"}:
            valid = np.asarray(value).ndim == 2
        elif kind in {"class_labels", "groups"}:
            valid = np.asarray(value).ndim in ({1, 2} if kind == "class_labels" else {1})
        else:
            raise ValueError(f"Unsupported edge data_kind: {kind}")
        if not valid: raise ValueError(f"Edge data does not match {kind}.")

    def _execute(self, stage, external):
        outputs, trace = {}, []
        for key in self.order:
            spec = self.specs[key]
            if stage not in spec.stages: continue
            arguments, fit_arguments = {}, {}
            for edge in self.edges:
                if edge.target != key: continue
                fit_only = edge.target_port.startswith("fit:")
                if fit_only and stage != "fit": continue
                if edge.source == "input":
                    if edge.source_port not in external:
                        raise ValueError(f"{key}: missing external port {edge.source_port} in {stage}.")
                    value = external[edge.source_port]
                else:
                    if edge.source not in outputs: raise ValueError(f"{key}: source did not run: {edge.source}")
                    value = outputs[edge.source]
                self._kind(value, edge.data_kind)
                if fit_only: fit_arguments[edge.target_port[4:]] = value
                else: arguments["X" if edge.target_port == "input" else edge.target_port] = value
            tool = self.nodes_[key]
            started = time.perf_counter()
            try:
                if stage == "fit" and tool.requires_fit:
                    tool.fit(**{**arguments, **fit_arguments})
                if spec.map_samples:
                    batch = arguments["X"]
                    if not isinstance(batch, (list, tuple)) or not batch:
                        raise ValueError("map_samples expects a nonempty list of Spectra.")
                    value = [getattr(tool, spec.method)(**{**arguments, "X": item}) for item in batch]
                    if spec.stack_output:
                        axis = value[0].axis
                        if not all(isinstance(item, Spectra) and np.array_equal(item.axis, axis) for item in value):
                            raise ValueError("stack_output requires identical spectral axes.")
                        ids = tuple(identifier for item in value for identifier in item.sample_ids) if all(item.sample_ids is not None for item in value) else None
                        value = Spectra(np.vstack([item.intensity for item in value]), axis, ids)
                else:
                    value = getattr(tool, spec.method)(**arguments)
            except Exception as error:
                raise ValueError(f"{stage} node {key} ({type(tool).__name__}) failed: {error}") from error
            outputs[key] = value
            trace.append({"node": key, "stage": stage, "fit": stage == "fit" and tool.requires_fit,
                          "method": spec.method, "seconds": time.perf_counter()-started})
        self.last_trace_ = trace
        return outputs

    def fit(self, X, y=None, *, groups=None, **ports):
        self.is_fitted_ = False
        self.nodes_ = {key: clone(spec.tool) for key, spec in self.specs.items()}
        external = {"X": X, **ports}
        if y is not None: external["y"] = y
        if groups is not None: external["groups"] = groups
        self._execute("fit", external)
        self.fit_trace_ = list(self.last_trace_)
        self.is_fitted_ = True
        return self

    def _inference(self, stage, X, ports):
        if not self.is_fitted_: raise NotFittedError("Call Workflow.fit first.")
        return self._execute(stage, {"X": X, **ports})

    def predict(self, X, **ports):
        return self._inference("predict", X, ports)[self.output_node]

    def evaluate(self, X, y, *, groups=None, **ports):
        ports["y"] = y
        if groups is not None: ports["groups"] = groups
        return self._inference("evaluate", X, ports)

    def to_dict(self):
        def serial(value):
            if isinstance(value, Node):
                return {"__node__": type(value).__name__, "parameters": serial(value.get_params(deep=False))}
            if isinstance(value, np.ndarray): return value.tolist()
            if isinstance(value, np.generic): return value.item()
            if isinstance(value, Path): return str(value)
            if isinstance(value, dict): return {key: serial(item) for key,item in value.items()}
            if isinstance(value, (tuple,list)): return [serial(item) for item in value]
            return value
        return serial({"version": 1, "output_node": self.output_node,
                       "nodes": {key: {"tool": type(spec.tool).__name__,
                           "parameters": spec.tool.get_params(deep=False), "method": spec.method,
                           "stages": spec.stages, "map_samples": spec.map_samples,
                           "stack_output": spec.stack_output} for key,spec in self.specs.items()},
                       "edges": [asdict(edge) for edge in self.edges]})

    @classmethod
    def from_dict(cls, record, registry):
        """Instantiate only explicitly allowlisted tool names; never dynamically import code."""
        if record.get("version") != 1: raise ValueError("Unsupported workflow spec version.")
        def restore(value):
            if isinstance(value, dict):
                if '__node__' in value:
                    name = value['__node__']
                    if name not in registry:
                        raise ValueError(f"Tool not in registry: {name}")
                    return registry[name](**restore(value['parameters']))
                return {k: restore(v) for k, v in value.items()}
            if isinstance(value, list):
                return [restore(v) for v in value]
            return value
        nodes = {}
        for key, value in record["nodes"].items():
            config = dict(value)
            name = config.pop("tool")
            if name not in registry: raise ValueError(f"Tool not in registry: {name}")
            tool = registry[name](**restore(config.pop("parameters")))
            config["stages"] = tuple(config["stages"])
            nodes[key] = NodeSpec(tool, **config)
        return cls(nodes, [Edge(**edge) for edge in record["edges"]], record["output_node"])
