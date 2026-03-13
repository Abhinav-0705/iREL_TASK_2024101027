#!/usr/bin/env python3
"""Method 2: Hybrid implicit prerequisite pipeline for a single video.

This file contains the code needed to run the hybrid implicit detector and
the filtered DAG visualization for one video.

Assumptions (run method1.py first):
  - data/raw/<video_id>_transcript_en.json exists
  - data/processed/<video_id>_concepts.json exists
"""

import json
import sys
from pathlib import Path

import networkx as nx


def ensure_prereqs(video_id: str) -> None:
    concepts_path = Path(f"data/processed/{video_id}_concepts.json")
    transcript_en_path = Path(f"data/raw/{video_id}_transcript_en.json")

    missing = []
    if not concepts_path.exists():
        missing.append(str(concepts_path))
    if not transcript_en_path.exists():
        missing.append(str(transcript_en_path))

    if missing:
        print("❌ method2: Required inputs not found:")
        for m in missing:
            print(f"   - {m}")
        print("Run method1.py for this video first.")
        sys.exit(1)


def run_hybrid_detector(video_id: str) -> None:
    """Run the hybrid implicit detector logic.

    This assumes you already have a script named hybrid_implicit_detector.py that
    takes <video_id> and produces data/processed/<video_id>_implicit_prerequisites.json.
    Here we simply execute that script as a subprocess.
    """
    detector_path = Path("hybrid_implicit_detector.py")
    if not detector_path.exists():
        print("❌ hybrid_implicit_detector.py not found; please keep using that script directly or add its logic here.")
        sys.exit(1)

    # Fallback to executing the script to avoid tight coupling
    from subprocess import run

    result = run([sys.executable, str(detector_path), video_id])
    if result.returncode != 0:
        print(f"❌ hybrid_run_detector.py failed for {video_id} (exit code {result.returncode})")
        sys.exit(result.returncode)


def load_concepts(video_id: str):
    concepts_path = Path(f"data/processed/{video_id}_concepts.json")
    with concepts_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["concepts"]


def load_implicit_edges(video_id: str):
    implicit_path = Path(f"data/processed/{video_id}_implicit_prerequisites.json")
    with implicit_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["dependencies"]


def filter_edges(edges):
    """Apply the same thresholds as filter_and_visualize_video2_implicit.py."""
    filtered = []
    for dep in edges:
        method = dep.get("method", "")
        conf = float(dep.get("confidence", 0.0))

        keep = False
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


def build_graph(concepts, edges):
    G = nx.DiGraph()
    for c in concepts:
        label = c["concept"]
        G.add_node(label, **c)

    for dep in edges:
        prereq = dep["prerequisite"]
        target = dep["concept"]
        conf = float(dep.get("confidence", 0.0))
        method = dep.get("method", "unknown")
        if prereq in G.nodes and target in G.nodes:
            G.add_edge(prereq, target, confidence=conf, edge_type=method)
    return G


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
    if len(sys.argv) < 2:
        print("Usage: python method2.py <video_id>")
        print("Example: python method2.py video_2")
        sys.exit(1)

    video_id = sys.argv[1]

    ensure_prereqs(video_id)

    print(f"\n[method2] Running hybrid implicit detector for {video_id}\n")
    run_hybrid_detector(video_id)

    concepts = load_concepts(video_id)
    implicit_edges = load_implicit_edges(video_id)
    print(f"Loaded {len(concepts)} concepts and {len(implicit_edges)} implicit edges")

    filtered = filter_edges(implicit_edges)
    print(f"After filtering: {len(filtered)} edges kept")

    G = build_graph(concepts, filtered)

    # Break cycles, if any, like in filter_and_visualize_video2_implicit
    if not nx.is_directed_acyclic_graph(G):
        print("Graph has cycles; attempting to break them by removing weakest edges")
        while True:
            try:
                cycle = list(nx.find_cycle(G))
            except nx.NetworkXNoCycle:
                break
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

    # Reuse the same visualization helper script for plotting
    from subprocess import run
    viz_script = Path("filter_and_visualize_video2_implicit.py")
    if not viz_script.exists():
        print("❌ filter_and_visualize_video2_implicit.py not found; cannot create PNG visualization.")
    else:
        run([sys.executable, str(viz_script), video_id])

    save_graph_json(G, video_id)
    print(f"\n✅ method2 completed successfully for {video_id}")


if __name__ == "__main__":
    main()
