"""
Step 8: DAG Visualization - Create prerequisite graph visualizations
"""
import json
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import networkx as nx


class DAGVisualizer:
    """Visualize concept prerequisite DAGs"""
    
    def __init__(
        self,
        input_dir: str = "data/output",
        output_dir: str = "data/output/visualizations"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dag(self, file_path: str) -> Dict:
        """Load DAG structure from JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_networkx_dag(self, dag_data: Dict) -> nx.DiGraph:
        """Create NetworkX graph from DAG structure"""
        G = nx.DiGraph()
        
        # Add nodes
        for node in dag_data['nodes']:
            G.add_node(
                node['id'],
                label=node['label']
            )
        
        # Add edges
        for edge in dag_data['edges']:
            G.add_edge(
                edge['source'],
                edge['target'],
                confidence=edge.get('confidence', 0.0),
                signals=edge.get('teaching_signals', [])
            )
        
        return G
    
    def visualize_dag(
        self,
        dag_data: Dict,
        output_file: str,
        figsize=(16, 12),
        show_confidence=True
    ):
        """Create DAG visualization"""
        
        G = self.create_networkx_dag(dag_data)
        
        if G.number_of_nodes() == 0:
            print(f"  No nodes to visualize for {dag_data['video_id']}")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout - hierarchical for DAG
        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except:
            pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        node_labels = {n: G.nodes[n]['label'][:20] for n in G.nodes()}
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color='#3498db',
            node_size=2000,
            alpha=0.9,
            ax=ax
        )
        
        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=9,
            font_weight='bold',
            ax=ax
        )
        
        # Draw edges with confidence-based styling
        if show_confidence and G.number_of_edges() > 0:
            for (u, v, data) in G.edges(data=True):
                confidence = data.get('confidence', 0.5)
                
                # Color based on confidence
                if confidence >= 0.8:
                    color = '#27ae60'  # Green - high confidence
                    width = 2.5
                elif confidence >= 0.5:
                    color = '#f39c12'  # Orange - medium confidence
                    width = 2.0
                else:
                    color = '#e74c3c'  # Red - low confidence
                    width = 1.5
                
                nx.draw_networkx_edges(
                    G, pos,
                    [(u, v)],
                    edge_color=color,
                    width=width,
                    alpha=0.7,
                    arrowsize=20,
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.1',
                    ax=ax
                )
        else:
            nx.draw_networkx_edges(
                G, pos,
                edge_color='#95a5a6',
                width=1.5,
                alpha=0.6,
                arrowsize=15,
                ax=ax
            )
        
        # Title and legend
        video_id = dag_data.get('video_id', 'Unknown')
        num_nodes = dag_data['metadata']['num_nodes']
        num_edges = dag_data['metadata']['num_edges']
        
        ax.set_title(
            f"Concept Prerequisite DAG - {video_id}\n"
            f"{num_nodes} Concepts, {num_edges} Prerequisites",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        if show_confidence:
            legend_elements = [
                plt.Line2D([0], [0], color='#27ae60', lw=2.5, label='High Confidence (≥0.8)'),
                plt.Line2D([0], [0], color='#f39c12', lw=2.0, label='Medium Confidence (≥0.5)'),
                plt.Line2D([0], [0], color='#e74c3c', lw=1.5, label='Low Confidence (<0.5)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Saved visualization to {output_file}")
    
    def process_video(self, video_id: str):
        """Create visualizations for a single video"""
        
        input_file = self.input_dir / f"{video_id}_hypergraph_structure.json"
        
        if not input_file.exists():
            print(f"  ✗ File not found: {input_file}")
            return
        
        print(f"\n{'='*60}")
        print(f"Visualizing: {video_id}")
        print(f"{'='*60}")
        
        # Load DAG
        dag_data = self.load_dag(input_file)
        
        # Create visualization
        output_file = self.output_dir / f"{video_id}_dag.png"
        self.visualize_dag(dag_data, output_file)
        
        print(f"✓ Completed visualization for {video_id}")
    
    def process_all_videos(self):
        """Process all videos"""
        dag_files = list(self.input_dir.glob("*_hypergraph_structure.json"))
        
        if not dag_files:
            print("No DAG files found to visualize")
            return
        
        print(f"\nFound {len(dag_files)} DAG files to visualize")
        
        for dag_file in dag_files:
            video_id = dag_file.stem.replace('_hypergraph_structure', '')
            
            try:
                self.process_video(video_id)
            except Exception as e:
                print(f"✗ Error visualizing {video_id}: {e}")
                import traceback
                traceback.print_exc()


def main():
    visualizer = DAGVisualizer()
    visualizer.process_all_videos()


if __name__ == "__main__":
    main()
