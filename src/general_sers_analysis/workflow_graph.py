"""Render an actual Node/Edge DAG for inspection; this module does not execute it."""
from pathlib import Path
import json
import textwrap
import numpy as np
from .core import Node, Edge
from .util import visualization_run_dir


def draw_workflow_graph(nodes, edges, output_dir, *, external_nodes=None, note=""):
    """Save workflow_graph.png/.svg/.json using the supplied node and edge objects."""
    external_nodes = {"input": "Input spectra"} if external_nodes is None else external_nodes
    if set(nodes) & set(external_nodes):
        raise ValueError("External and tool node IDs must be distinct.")
    if not all(isinstance(node, Node) for node in nodes.values()):
        raise TypeError("All workflow tools must inherit Node.")
    ids = list(external_nodes) + list(nodes)
    incoming = {key: [] for key in ids}
    for edge in edges:
        if not isinstance(edge, Edge) or edge.source not in incoming or edge.target not in incoming:
            raise ValueError("Every edge must reference existing tool or external nodes.")
        incoming[edge.target].append(edge.source)
    levels = {}
    while len(levels) < len(ids):
        ready = [key for key in ids if key not in levels and all(source in levels for source in incoming[key])]
        if not ready:
            raise ValueError("The graph contains a directed cycle.")
        for key in ready:
            levels[key] = 1 + max((levels[source] for source in incoming[key]), default=-1)
    columns = {}
    for key in ids:
        columns.setdefault(levels[key], []).append(key)
    positions = {key: (level * 3.5, (len(column)-1)/2-index)
                 for level, column in columns.items() for index, key in enumerate(column)}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(max(9, len(columns)*3.1), max(4, max(map(len, columns.values()))*2)),
                           layout="constrained")
    width, height = 2.65, .55
    try:
        for key, (x, y) in positions.items():
            external = key in external_nodes
            tool = external_nodes[key] if external else type(nodes[key]).__name__
            detail = ""
            if not external:
                parameters = nodes[key].get_params(deep=False)
                for name in ("lower", "upper", "lam"):
                    if name in parameters:
                        detail += f" {name}={parameters[name]:g}"
                for name in ('window_length', 'polyorder', 'deriv', 'scope'):
                    if name in parameters:
                        detail += f" {name}={parameters[name]}"
                if "target_axis" in parameters:
                    grid = np.asarray(parameters["target_axis"])
                    detail += f" {len(grid)} points [{grid[0]:g}, {grid[-1]:g}]"
                if "parameters" in parameters:
                    detail += f" {parameters['parameters']}"
                for name in ('regressor', 'classifier'):
                    nested = parameters.get(name)
                    if isinstance(nested, Node):
                        detail += f" {type(nested).__name__}"
                        for k, value in (nested.get_params(deep=False).get('parameters') or {}).items():
                            if k not in ('random_state', 'n_jobs', 'verbosity', 'max_iter', 'probability'):
                                detail += f" {k}={value}"
            ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,
                         boxstyle="round,pad=0.04", facecolor="#e3eff9" if external else "#e5f3e9",
                         edgecolor="#3a576b", linewidth=1))
            ax.text(x,y+.07,f"{key}\n{tool}",ha="center",va="center",fontsize=10)
            if detail:
                ax.text(x,y-height/2-.08,textwrap.fill(detail.strip(), width=40),ha="center",va="top",fontsize=7)
        for edge in edges:
            curve = .30 if edge.target_port == "y_true" else (.22 if edge.target_port.startswith("fit:") else .14)
            x1,y1=positions[edge.source]; x2,y2=positions[edge.target]
            start=(x1+width/2+.04,y1); end=(x2-width/2-.04,y2)
            ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=14,color="#415b6a",
                connectionstyle=f"arc3,rad={curve}" if edge.source == "input" and edge.source_port != "X" and edge.source_port != "output" else "arc3,rad=0",
                linestyle="--" if edge.target_port != "input" else "-"))
            label=edge.data_kind or "data"
            if edge.target_port != "input":
                label = f"{edge.source_port} -> {edge.target_port} ({label})"
            if edge.condition is not None:
                label += f"\n{edge.condition.field} {edge.condition.operation} {edge.condition.value}"
            metadata_edge = edge.source == "input" and edge.source_port not in {"X", "output"}
            label_y = (y1+y2)/2 - (end[0]-start[0])*curve/2 if metadata_edge else (y1+y2)/2+.12
            ax.text((start[0]+end[0])/2,label_y,label,ha="center",fontsize=7,
                    bbox={"facecolor":"white","edgecolor":"none","alpha":.85,"pad":1})
        ax.set_xlim(-1.6,max(x for x,y in positions.values())+1.6)
        ax.set_ylim(min(y for x,y in positions.values())-3.2,max(y for x,y in positions.values())+.9)
        ax.set_title("Declared Node / Edge workflow",fontsize=13)
        ax.axis("off")
        if note:
            fig.supxlabel(note,fontsize=9)
        output = Path(output_dir)
        output.mkdir(parents=True,exist_ok=True)
        paths={suffix: output/f"workflow_graph.{suffix}" for suffix in ("png","svg","json")}
        for suffix in ("png","svg"):
            fig.savefig(paths[suffix],dpi=180)
        from dataclasses import asdict
        def encode(value):
            if isinstance(value, Node):
                return {'__node__': type(value).__name__, 'parameters': value.get_params(deep=False)}
            if isinstance(value,np.ndarray): return value.tolist()
            if isinstance(value,Path): return str(value)
            if isinstance(value,np.generic): return value.item()
            raise TypeError(type(value).__name__)
        record={"external_nodes":external_nodes,
                "nodes":{key:{"tool":type(node).__name__,"parameters":node.get_params(deep=False)}
                         for key,node in nodes.items()},"edges":[asdict(edge) for edge in edges],"note":note}
        paths["json"].write_text(json.dumps(record,indent=2,default=encode),encoding="utf-8")
    finally:
        plt.close(fig)
    return {suffix:str(path.resolve()) for suffix,path in paths.items()}
