#!/usr/bin/env python3
"""
Create enhanced DAG visualization with implicit dependencies
"""
import json
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import numpy as np

def create_enhanced_dag_visualization():
    """Create DAG with both explicit and implicit dependencies"""
    
    # Load data
    concepts_file = "data/processed/video_2_concepts.json"
    explicit_file = "data/processed/video_2_prerequisites.json"
    implicit_file = "data/processed/video_2_implicit_prerequisites.json"
    
    with open(concepts_file, 'r') as f:
        concepts_data = json.load(f)
    
    explicit_deps = []
    if Path(explicit_file).exists():
        with open(explicit_file, 'r') as f:
            explicit_data = json.load(f)
            explicit_deps = explicit_data.get('prerequisites', [])
    
    with open(implicit_file, 'r') as f:
        implicit_data = json.load(f)
        implicit_deps = implicit_data['dependencies']
    
    # Create graph
    G = nx.DiGraph()
    
    # Add all concepts as nodes
    concepts = [c['concept'] for c in concepts_data['concepts']]
    for concept in concepts:
        G.add_node(concept)
    
    # Add edges with different types
    edges_data = []
    
    # Add explicit dependencies (if any verified as True)
    for dep in explicit_deps:
        if dep.get('verified', False):
            edges_data.append({
                'source': dep['prerequisite'],
                'target': dep['concept'],
                'confidence': dep.get('confidence', 0.8),
                'type': 'explicit'
            })
    
    # Add high-confidence implicit dependencies
    for dep in implicit_deps:
        if dep['confidence'] >= 0.6:  # Only high confidence
            edges_data.append({
                'source': dep['prerequisite'],
                'target': dep['concept'],
                'confidence': dep['confidence'],
                'type': dep['method']
            })
    
    # Add edges to graph
    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'], 
                   confidence=edge['confidence'], 
                   edge_type=edge['type'])
    
    print(f"📊 Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Use hierarchical layout
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Separate nodes by type
    core_concepts = [c for c in concepts if any(
        cat in ['core'] for cat in [concepts_data['concepts'][i].get('llm_category', '') 
                                  for i, concept in enumerate([x['concept'] for x in concepts_data['concepts']]) 
                                  if concept == c]
    )]
    
    other_concepts = [c for c in concepts if c not in core_concepts]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=core_concepts,
                          node_color='lightblue', 
                          node_size=1000, 
                          alpha=0.8, 
                          label='Core Concepts')
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=other_concepts,
                          node_color='lightgreen', 
                          node_size=700, 
                          alpha=0.8, 
                          label='Supporting Concepts')
    
    # Draw edges with different styles
    explicit_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'explicit']
    domain_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'domain_pattern']
    semantic_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'semantic_temporal']
    cooccurrence_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'cooccurrence']
    combined_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'combined']
    transitive_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'transitive']
    
    # Draw different edge types
    if explicit_edges:
        nx.draw_networkx_edges(G, pos, explicit_edges, 
                             edge_color='red', width=3, alpha=0.8, 
                             style='solid', label='Explicit (LLM verified)')
    
    if domain_edges:
        nx.draw_networkx_edges(G, pos, domain_edges, 
                             edge_color='blue', width=2, alpha=0.7, 
                             style='solid', label='Domain patterns')
    
    if semantic_edges:
        nx.draw_networkx_edges(G, pos, semantic_edges, 
                             edge_color='green', width=1.5, alpha=0.6, 
                             style='dashed', label='Semantic+Temporal')
    
    if cooccurrence_edges:
        nx.draw_networkx_edges(G, pos, cooccurrence_edges, 
                             edge_color='orange', width=2.5, alpha=0.8, 
                             style='solid', label='Co-occurrence')
    
    if combined_edges:
        nx.draw_networkx_edges(G, pos, combined_edges, 
                             edge_color='purple', width=2, alpha=0.7, 
                             style='solid', label='Multi-method')
    
    if transitive_edges:
        nx.draw_networkx_edges(G, pos, transitive_edges, 
                             edge_color='gray', width=1, alpha=0.5, 
                             style='dotted', label='Transitive')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title(f'Video 2: Enhanced Prerequisite DAG\\n'
              f'{G.number_of_nodes()} concepts, {G.number_of_edges()} dependencies\\n'
              f'Hybrid Detection: Semantic+Temporal, Co-occurrence, Domain, Transitive', 
              fontsize=14, fontweight='bold')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    output_file = "data/visualizations/video_2_enhanced_dag.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Enhanced DAG saved to: {output_file}")
    
    # Save graph data
    graph_data = {
        'video_id': 'video_2',
        'method': 'hybrid_enhanced',
        'nodes': list(G.nodes()),
        'edges': [
            {
                'source': u,
                'target': v,
                'confidence': d.get('confidence', 0.5),
                'type': d.get('edge_type', 'unknown')
            }
            for u, v, d in G.edges(data=True)
        ],
        'statistics': {
            'total_concepts': G.number_of_nodes(),
            'total_dependencies': G.number_of_edges(),
            'explicit_dependencies': len(explicit_edges),
            'domain_dependencies': len(domain_edges),
            'semantic_dependencies': len(semantic_edges),
            'cooccurrence_dependencies': len(cooccurrence_edges),
            'combined_dependencies': len(combined_edges),
            'transitive_dependencies': len(transitive_edges)
        }
    }
    
    graph_output_file = "data/output/video_2_enhanced_dag.json"
    Path("data/output").mkdir(exist_ok=True)
    with open(graph_output_file, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"📊 Graph data saved to: {graph_output_file}")
    
    # Print statistics
    print(f"\n📈 ENHANCED DAG STATISTICS:")
    print(f"   🔵 Core concepts: {len(core_concepts)}")
    print(f"   🟢 Supporting concepts: {len(other_concepts)}")
    print(f"   🔴 Explicit dependencies: {len(explicit_edges)}")
    print(f"   🔵 Domain pattern dependencies: {len(domain_edges)}")
    print(f"   🟢 Semantic+temporal dependencies: {len(semantic_edges)}")
    print(f"   🟠 Co-occurrence dependencies: {len(cooccurrence_edges)}")
    print(f"   🟣 Multi-method dependencies: {len(combined_edges)}")
    print(f"   ⚪ Transitive dependencies: {len(transitive_edges)}")
    print(f"   📊 Total dependencies: {G.number_of_edges()}")
    
    # Analyze key concepts
    print(f"\n🎯 KEY CONCEPT ANALYSIS:")
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    # Most fundamental (high out-degree, low in-degree)
    fundamental_score = {node: out_degrees[node] - in_degrees[node] for node in G.nodes()}
    top_fundamental = sorted(fundamental_score.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"   🏗️  Most fundamental concepts:")
    for concept, score in top_fundamental:
        print(f"      - {concept}: feeds into {out_degrees[concept]} concepts, needs {in_degrees[concept]}")
    
    # Most complex (high in-degree)
    top_complex = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"   🧠 Most complex concepts:")
    for concept, prereqs in top_complex:
        print(f"      - {concept}: requires {prereqs} prerequisites")
    
    return G

if __name__ == "__main__":
    create_enhanced_dag_visualization()