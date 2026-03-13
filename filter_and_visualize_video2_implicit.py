#!/usr/bin/env python3
"""Filter hybrid implicit prerequisites for a video and create a clean DAG.

Selection rules (video-only signals, as discussed):
- Keep an edge if:
  - method == "video_explanation_overlap" and confidence >= 0.5, OR
  - method == "semantic_temporal" and confidence >= 0.5, OR
  - method == "cooccurrence" (already strong), OR
  - method == "combined" and confidence >= 0.6.

All extracted concepts are always added as nodes, even if they have no edges.
"""

import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import networkx as nx


def load_concepts(video_id: str) -> List[Dict]:
    concepts_path = Path(f"data/processed/{video_id}_concepts.json")
    with concepts_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["concepts"]


def load_implicit_edges(video_id: str) -> List[Dict]:
    implicit_path = Path(f"data/processed/{video_id}_implicit_prerequisites.json")
    with implicit_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["dependencies"]


def filter_edges(edges: List[Dict]) -> List[Dict]:
    """Apply per-method thresholds to select cleaner prerequisite edges for the graph."""
    filtered: List[Dict] = []
    for dep in edges:
        method = dep.get("method", "")
        conf = float(dep.get("confidence", 0.0))

        keep = False
        # Option A: Raise the threshold for video_explanation_overlap to 0.75
        if method == "semantic_temporal" and conf >= 0.5:
            keep = True
        elif method == "video_explanation_overlap" and conf >= 0.8:
            keep = True
        elif method == "cooccurrence":
            keep = True
        elif method == "combined" and conf >= 0.6:
            keep = True

        if keep:
            filtered.append(dep)

    return filtered


def build_graph(concepts: List[Dict], edges: List[Dict]) -> nx.DiGraph:
    """Build a directed graph with all concepts as nodes and filtered edges."""
    G = nx.DiGraph()

    # Add all concepts (ensure every extracted concept is present as a node)
    for c in concepts:
        label = c["concept"]
        G.add_node(label, **c)

    # Add edges using prerequisite -> concept direction
    for dep in edges:
        prereq = dep["prerequisite"]
        target = dep["concept"]
        conf = float(dep.get("confidence", 0.0))
        method = dep.get("method", "unknown")

        # Only add if both endpoints exist as concepts
        if prereq in G.nodes and target in G.nodes:
            G.add_edge(prereq, target, confidence=conf, edge_type=method)

    return G


def visualize_graph(G: nx.DiGraph, video_id: str) -> None:
    """Create and save a DAG-style visualization for the given graph."""
    print(f"Graph summary: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Improved layout for clarity
    # Larger canvas so all edges and labels are clearly visible
    plt.figure(figsize=(24, 14))

    # Try to build a layered layout based on a simple topological order so prerequisites are to the left
    try:
        # If the graph has cycles, fall back to a DAG of its condensation
        if nx.is_directed_acyclic_graph(G):
            topo_nodes = list(nx.topological_sort(G))
        else:
            topo_nodes = list(G.nodes())

        layers: list[list[str]] = []
        node_layer: dict[str, int] = {}

        for n in topo_nodes:
            # Place node one layer after the deepest of its predecessors
            preds = list(G.predecessors(n))
            if not preds:
                layer = 0
            else:
                layer = max(node_layer.get(p, 0) for p in preds) + 1
            node_layer[n] = layer
            while len(layers) <= layer:
                layers.append([])
            layers[layer].append(n)

        # Build positions: x by layer, y spaced within layer
        pos = {}
        max_layer = max(node_layer.values()) if node_layer else 0
        for layer_idx, nodes_in_layer in enumerate(layers):
            x_base = layer_idx if max_layer == 0 else layer_idx / max_layer
            n_in_layer = len(nodes_in_layer) or 1
            for i, n in enumerate(nodes_in_layer):
                # spread vertically in [-1.5,1.5] for more separation
                y = 0 if n_in_layer == 1 else 3 * (i / (n_in_layer - 1)) - 1.5
                # small horizontal jitter to separate parallel edges slightly
                x = x_base + (0.02 * (i - n_in_layer / 2))
                pos[n] = (x, y)
    except Exception:
        # Fallback to force-directed layout
        pos = nx.spring_layout(G, seed=42)

    # Separate nodes by llm_category if available
    core_nodes = []
    other_nodes = []
    for node, data in G.nodes(data=True):
        if data.get("llm_category") == "core":
            core_nodes.append(node)
        else:
            other_nodes.append(node)

    if core_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=core_nodes,
            node_color="lightblue",
            node_size=900,  # slightly smaller nodes
            edgecolors="blue",
            linewidths=2,
            alpha=0.9,
            label="Core concepts",
        )

    if other_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=other_nodes,
            node_color="lightgreen",
            node_size=700,  # slightly smaller nodes
            edgecolors="darkgreen",
            linewidths=1.5,
            alpha=0.85,
            label="Other concepts",
        )

    # Edge styling by method
    edge_type_to_style = {
        "video_explanation_overlap": {"color": "gray", "style": "dashed", "width": 1.8},
        "semantic_temporal": {"color": "green", "style": "dotted", "width": 2.0},
        "cooccurrence": {"color": "orange", "style": "solid", "width": 2.4},
        "combined": {"color": "purple", "style": "solid", "width": 2.6},
    }

    edge_labels_dict = {}
    for edge_type, style in edge_type_to_style.items():
        et_edges = [
            (u, v)
            for u, v, d in G.edges(data=True)
            if d.get("edge_type") == edge_type
        ]
        if et_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=et_edges,
                edge_color=style["color"],
                width=style["width"]+1.2,  # increase edge width
                style=style["style"],
                alpha=0.92,  # increase opacity
                arrows=True,
                arrowsize=22,
                arrowstyle="->",
                label=edge_type,
            )
            # Add edge labels for these edges
            for u, v in et_edges:
                edge_labels_dict[(u, v)] = f"{u}→{v}"

    # Node labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    # Edge labels (show only for visible edges)
    if edge_labels_dict:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict, font_size=6, label_pos=0.45, rotate=False)

    plt.title(
        f"Video {video_id}: Filtered Hybrid Implicit Prerequisite DAG\n"
        f"All {G.number_of_nodes()} concepts included, {G.number_of_edges()} filtered edges",
        fontsize=14,
        fontweight="bold",
    )

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.axis("off")
    plt.tight_layout(pad=2.5)

    vis_dir = Path("data/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / f"{video_id}_implicit_filtered_dag.png"
    # Higher DPI for sharper edges/labels
    plt.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {out_path}")


def save_graph_json(G: nx.DiGraph, video_id: str) -> None:
    out_dir = Path("data/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}_implicit_filtered_dag.json"

    nodes = []
    for i, (node, data) in enumerate(G.nodes(data=True)):
        nodes.append({
            "id": i,
            "label": node,
            "concept": node,
            "llm_category": data.get("llm_category", ""),
        })

    node_index = {n["label"]: n["id"] for n in nodes}

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "source": node_index[u],
            "target": node_index[v],
            "prerequisite": u,
            "target_concept": v,
            "confidence": float(data.get("confidence", 0.0)),
            "edge_type": data.get("edge_type", ""),
        })

    dag_data = {
        "video_id": video_id,
        "method": "hybrid_implicit_filtered",
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(dag_data, f, indent=2, ensure_ascii=False)

    print(f"Saved graph JSON to {out_path}")


def main() -> None:
    import sys

    video_id = sys.argv[1] if len(sys.argv) > 1 else "video_2"

    concepts = load_concepts(video_id)
    implicit_edges = load_implicit_edges(video_id)

    print(f"Loaded {len(concepts)} concepts and {len(implicit_edges)} implicit edges")

    filtered = filter_edges(implicit_edges)
    print(f"After filtering: {len(filtered)} edges kept")

    G = build_graph(concepts, filtered)

    # Ensure acyclicity if possible (remove weakest edges in cycles)
    if not nx.is_directed_acyclic_graph(G):
        print("Graph has cycles; attempting to break them by removing weakest edges")
        # Simple heuristic: iteratively remove lowest-confidence edge from any cycle
        while True:
            try:
                cycle = list(nx.find_cycle(G))
            except nx.NetworkXNoCycle:
                break

            # Pick edge with lowest confidence in this cycle
            weakest_edge = None
            weakest_conf = float("inf")
            for u, v in cycle:
                conf = float(G[u][v].get("confidence", 0.0))
                if conf < weakest_conf:
                    weakest_conf = conf
                    weakest_edge = (u, v)

            if weakest_edge is None:
                break

            print(f"  Removing edge {weakest_edge} with confidence {weakest_conf:.3f} to break cycle")
            G.remove_edge(*weakest_edge)

    visualize_graph(G, video_id)
    save_graph_json(G, video_id)


if __name__ == "__main__":
    main()
