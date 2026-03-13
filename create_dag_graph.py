#!/usr/bin/env python3
"""
Simple DAG graph creator from verified prerequisites
Creates a visual graph showing prerequisite relationships between concepts
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def create_dag_from_prerequisites(video_id):
    """Create and visualize DAG from verified prerequisites"""
    
    # Load verified prerequisites
    prereq_file = Path(f"data/processed/{video_id}_prerequisites.json")
    
    if not prereq_file.exists():
        print(f"✗ Prerequisites file not found: {prereq_file}")
        return
    
    with open(prereq_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load all concepts
    concepts_file = Path(f"data/processed/{video_id}_concepts.json")
    if not concepts_file.exists():
        print(f"✗ Concepts file not found: {concepts_file}")
        return
        
    with open(concepts_file, 'r', encoding='utf-8') as f:
        concepts_data = json.load(f)
    
    confirmed_edges = data['confirmed_edges']
    all_concepts = [c['concept'] for c in concepts_data['concepts']]
    
    print(f"\n{'='*60}")
    print(f"Creating DAG for: {video_id}")
    print(f"Total concepts: {len(all_concepts)}")
    print(f"Verified edges: {len(confirmed_edges)}")
    print(f"{'='*60}\n")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add ALL concepts as nodes first (even isolated ones)
    for concept in all_concepts:
        G.add_node(concept)
    
    # Add edges from verified prerequisites
    for edge in confirmed_edges:
        prerequisite = edge['prerequisite']
        target = edge['target']
        confidence = edge['llm_confidence']
        signals = edge['teaching_signals']
        
        G.add_edge(
            prerequisite, 
            target,
            confidence=confidence,
            signals=signals
        )
        
        print(f"  {prerequisite} → {target} (confidence: {confidence:.2f})")
    
    print(f"\n✓ DAG created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Check if it's a valid DAG
    if not nx.is_directed_acyclic_graph(G):
        print("⚠️  Warning: Graph contains cycles!")
    else:
        print("✓ Graph is acyclic (valid DAG)")
    
    # Create visualization
    plt.figure(figsize=(20, 16))
    
    # Use spring layout with more space for many nodes
    try:
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    except:
        pos = nx.spring_layout(G, seed=42)
    
    # Identify nodes with edges vs isolated nodes
    nodes_with_edges = set()
    for u, v in G.edges():
        nodes_with_edges.add(u)
        nodes_with_edges.add(v)
    
    isolated_nodes = [n for n in G.nodes() if n not in nodes_with_edges]
    connected_nodes = [n for n in G.nodes() if n in nodes_with_edges]
    
    print(f"  Connected nodes: {len(connected_nodes)}")
    print(f"  Isolated nodes: {len(isolated_nodes)}")
    
    # Draw connected nodes (concepts with prerequisite relationships)
    if connected_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=connected_nodes,
            node_color='lightblue',
            node_size=3500,
            alpha=0.9,
            edgecolors='blue',
            linewidths=2
        )
    
    # Draw isolated nodes (concepts without prerequisite relationships)
    if isolated_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=isolated_nodes,
            node_color='lightgray',
            node_size=2500,
            alpha=0.6,
            edgecolors='gray',
            linewidths=1
        )
    
    # Draw edges with different colors based on confidence
    edge_colors = []
    edge_widths = []
    
    for u, v, data in G.edges(data=True):
        confidence = data['confidence']
        
        # Color by confidence
        if confidence >= 0.8:
            edge_colors.append('green')
            edge_widths.append(2.5)
        elif confidence >= 0.6:
            edge_colors.append('orange')
            edge_widths.append(2.0)
        else:
            edge_colors.append('red')
            edge_widths.append(1.5)
    
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6,
        arrows=True,
        arrowsize=20,
        arrowstyle='->'
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=9,
        font_weight='bold',
        font_family='sans-serif'
    )
    
    plt.title(f'Concept Prerequisite DAG - {video_id}\n{len(all_concepts)} concepts | {len(confirmed_edges)} verified prerequisites', 
              fontsize=16, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', label='Connected concepts (with prerequisites)'),
        Patch(facecolor='lightgray', edgecolor='gray', label='Isolated concepts (no prerequisites)'),
        Patch(facecolor='green', label='High confidence edge (≥0.8)'),
        Patch(facecolor='orange', label='Medium confidence edge (≥0.6)'),
        Patch(facecolor='red', label='Low confidence edge (<0.6)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    output_dir = Path("data/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{video_id}_dag.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_file}")
    
    plt.close()
    
    # Also save DAG structure as JSON
    dag_output = Path("data/output")
    dag_output.mkdir(parents=True, exist_ok=True)
    dag_file = dag_output / f"{video_id}_dag.json"
    
    # Convert to JSON-serializable format
    nodes = [{'id': i, 'label': node} for i, node in enumerate(G.nodes())]
    edges_list = []
    
    node_to_id = {node: i for i, node in enumerate(G.nodes())}
    
    for u, v, data in G.edges(data=True):
        edges_list.append({
            'source': node_to_id[u],
            'target': node_to_id[v],
            'prerequisite': u,
            'target_concept': v,
            'confidence': data['confidence'],
            'teaching_signals': data['signals']
        })
    
    dag_data = {
        'video_id': video_id,
        'nodes': nodes,
        'edges': edges_list,
        'metadata': {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'is_dag': nx.is_directed_acyclic_graph(G)
        }
    }
    
    with open(dag_file, 'w', encoding='utf-8') as f:
        json.dump(dag_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved DAG structure to: {dag_file}")
    
    return G

def main():
    import sys
    
    if len(sys.argv) > 1:
        video_id = sys.argv[1]
    else:
        video_id = "video_4"
    
    create_dag_from_prerequisites(video_id)
    print("\n✓ Done!")

if __name__ == "__main__":
    main()
