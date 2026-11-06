"""Typed call bindings and a deterministic DAG executor for classical spectral tools."""
from dataclasses import dataclass, asdict
import inspect
import time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from .core import Node, Edge, Spectra
from .core.batch import DataBatch, RowSelection


@dataclass
class NodeSpec:
    """Tool configuration, output method, execution stages and row-mapping contract.

    Attributes
    ----------
    map_samples: Apply a stateless tool separately to each spectrum in a raw input sequence.
    method: Output method: transform, predict, predict_proba or predict_scores.
    row_action: preserve keeps identities; aggregate updates group labels/lineage; select requires RowSelection.
    stack_output: Stack mapped Spectra into a shared-grid batch; coordinate arrays must match.
    stages: Execution stages selected from fit, predict and evaluate; learned nodes participate in all three.
    tool: Unfitted Node configuration; Workflow.fit clones it before execution.
    """
    tool: Node
    method: str = "transform"
    stages: tuple = ("fit", "predict", "evaluate")
    map_samples: bool = False
    stack_output: bool = False
    row_action: str = 'preserve'


class Workflow:
    """Topologically execute a Node/Edge DAG. Fit clones tools; inference reuses learned state. Supports ordinary payloads and DataBatch.

    Attributes
    ----------
    edges: Directed Edge objects defining dependencies and argument bindings.
    fit_trace_: Node execution records from the most recent successful fit.
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    last_trace_: Most recent execution records; batch execution also records input/output row counts.
    nodes_: Cloned execution tools from the most recent fit, including learned state.
    order: Validated topological execution order.
    output_node: ID of the node whose output is returned by predict.
    specs: Mapping from node IDs to original NodeSpec configurations.
    """
    def __init__(self, nodes, edges, output_node):
        """Store configuration without fitting training data. See the class attributes for constructor parameter meanings.

        Parameters
        ----------
        nodes: Mapping from node IDs to NodeSpec configurations.
        edges: Directed Edge objects defining dependencies and argument bindings.
        output_node: ID of the node whose output is returned by predict.

        Returns and state
        -----------------
        None. Stores configuration only.
        """
        self.specs = dict(nodes)
        self.edges = list(edges)
        self.output_node = output_node
        self._validate()
        self.is_fitted_ = False

    def _validate(self):
        """Validate node/port/stage bindings and DAG topology; reject unsafe predictive cascades and build order.

        Returns and state
        -----------------
        None; populates topological order or rejects the declaration.

        Raises
        ------
        TypeError, ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        if "input" in self.specs or self.output_node not in self.specs:
            raise ValueError("input is reserved; output_node must reference a tool.")
        for key, spec in self.specs.items():
            if not isinstance(spec, NodeSpec) or not isinstance(spec.tool, Node):
                raise TypeError(f"{key}: expected NodeSpec containing a Node.")
            if spec.row_action not in ('preserve', 'aggregate', 'select'):
                raise ValueError('Unknown row action.')
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
        """Check coarse payload type, not chemical meaning, units or experimental independence.

        Parameters
        ----------
        value: Comparison threshold or reference value.
        kind: Expected coarse payload category, or None to skip this check.

        Returns and state
        -----------------
        None; raises on an incompatible payload.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
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
        """Dispatch ordinary or batch execution for the requested stage and record traces.

        Parameters
        ----------
        stage: Execution stage: fit, predict or evaluate.
        external: External input mapping; X may be an ordinary payload or DataBatch.

        Returns and state
        -----------------
        Mapping from node IDs to output payloads; also updates last_trace_.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        if isinstance(external.get('X'), DataBatch):
            if external.get('y') is not None:
                batch = external['X']
                external = {**external, 'X': DataBatch(batch.X, batch.row_ids, batch.groups,
                                                       external['y'], batch.lineage)}
            return self._execute_batch(stage, external)
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

    def _execute_batch(self, stage, external):
        """Execute nodes in topological order while propagating labels, identities, pooling and selection mappings.

        Parameters
        ----------
        stage: Execution stage: fit, predict or evaluate.
        external: External input mapping; X may be an ordinary payload or DataBatch.

        Returns and state
        -----------------
        Mapping from node IDs to DataBatch outputs; also updates last_trace_.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        outputs, trace = {}, []
        for key in self.order:
            spec = self.specs[key]
            if stage not in spec.stages:
                continue
            edges = [e for e in self.edges if e.target == key]
            main = next(e for e in edges if e.target_port == 'input')
            batch = external['X'] if main.source == 'input' else outputs[main.source]
            arguments, fit_arguments = {'X': batch.X}, {}
            for edge in edges:
                port = edge.target_port
                if port == 'input' or (port.startswith('fit:') and stage != 'fit'):
                    continue
                name = port.removeprefix('fit:')
                if edge.source == 'input' and edge.source_port in ('y', 'groups'):
                    value = batch.y if edge.source_port == 'y' else batch.groups
                else:
                    value = external[edge.source_port] if edge.source == 'input' else outputs[edge.source]
                    if isinstance(value, DataBatch):
                        if set(value.row_ids) != set(batch.row_ids):
                            raise ValueError('Branch join requires matching row identities.')
                        aligned = value.take([value.row_ids.index(r) for r in batch.row_ids])
                        if aligned.groups != batch.groups or aligned.lineage != batch.lineage:
                            raise ValueError('Feature branches must describe the same source rows and groups.')
                        value = aligned.X
                (fit_arguments if port.startswith('fit:') else arguments)[name] = value
            tool = self.nodes_[key]
            tick = time.perf_counter()
            if stage == 'fit' and tool.requires_fit:
                tool.fit(**{**arguments, **fit_arguments})
            if spec.map_samples:
                result = [getattr(tool, spec.method)(**{**arguments, 'X': item}) for item in batch.X]
                if spec.stack_output:
                    axis = result[0].axis
                    if not all(np.array_equal(s.axis, axis) for s in result):
                        raise ValueError('Mapped spectra require matching axes before stacking.')
                    result = Spectra(np.vstack([s.intensity for s in result]), axis, batch.row_ids)
            else:
                result = getattr(tool, spec.method)(**arguments)
            if spec.row_action == 'aggregate':
                result = batch.aggregate(result)
            elif spec.row_action == 'select':
                if not isinstance(result, RowSelection):
                    raise ValueError('Selection nodes must return RowSelection.')
                result = batch.take(result.indices)
            else:
                result = batch.replace(result)
            outputs[key] = result
            trace.append({'node': key, 'stage': stage, 'fit': stage == 'fit' and tool.requires_fit,
                          'row_action': spec.row_action, 'input_rows': len(batch.row_ids),
                          'output_rows': len(result.row_ids), 'seconds': time.perf_counter()-tick})
        self.last_trace_ = trace
        return outputs

    def fit(self, X, y=None, *, groups=None, **ports):
        """Invalidate fitted state, clone configured tools, execute training, and save fit_trace_. Failure leaves the workflow unfitted.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.
        groups: Source-group identifiers aligned with current rows; determine pooling membership.
        ports: External named ports bound explicitly by Edge declarations.

        Returns and state
        -----------------
        self. Learned attributes are stored on the instance; see the class attribute descriptions.
        """
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
        """Require a fitted workflow and execute an inference stage without refitting.

        Parameters
        ----------
        stage: Execution stage: fit, predict or evaluate.
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        ports: External named ports bound explicitly by Edge declarations.

        Returns and state
        -----------------
        Mapping from node IDs to inference outputs.

        Raises
        ------
        NotFittedError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        if not self.is_fitted_: raise NotFittedError("Call Workflow.fit first.")
        return self._execute(stage, {"X": X, **ports})

    def predict(self, X, **ports):
        """Execute prediction without refitting and return output_node. Batch input returns a DataBatch, whose y still contains ground truth when supplied.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        ports: External named ports bound explicitly by Edge declarations.

        Returns and state
        -----------------
        Execute prediction without refitting and return output_node. Batch input returns a DataBatch, whose y still contains ground truth when supplied.
        """
        return self._inference("predict", X, ports)[self.output_node]

    def evaluate(self, X, y, *, groups=None, **ports):
        """Execute evaluation-only nodes with true labels/groups, without refitting; return all node outputs.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.
        groups: Source-group identifiers aligned with current rows; determine pooling membership.
        ports: External named ports bound explicitly by Edge declarations.

        Returns and state
        -----------------
        Boolean comparison result, or a Workflow mapping of evaluation node outputs.
        """
        ports["y"] = y
        if groups is not None: ports["groups"] = groups
        return self._inference("evaluate", X, ports)

    def to_dict(self):
        """Export configuration and nested Node declarations, excluding learned parameters.

        Returns and state
        -----------------
        JSON-compatible configuration without learned state.
        """
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
                           "stack_output": spec.stack_output, "row_action": spec.row_action} for key,spec in self.specs.items()},
                       "edges": [asdict(edge) for edge in self.edges]})

    @classmethod
    def from_dict(cls, record, registry):
        """Reconstruct an unfitted workflow using an explicit tool-class allowlist only.

        Parameters
        ----------
        record: Configuration dictionary exported by to_dict.
        registry: Explicit mapping from allowed tool names to classes, including nested tools.

        Returns and state
        -----------------
        Unfitted Workflow.

        Raises
        ------
        ValueError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
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
