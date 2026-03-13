"""
Step 8: Visualization - Create graph visualizations
"""
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


class HypergraphVisualizer:
    """Visualize concept hypergraphs"""
    
    def __init__(
        self,
        input_dir: str = "data/output",
        output_dir: str = "data/output/visualizations"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme
        self.colors = {
            "concept": "#3498db",      # Blue
            "virtual_and": "#e74c3c",  # Red
            "virtual_or": "#f39c12",   # Orange
            "edge_single": "#2ecc71",  # Green
            "edge_and": "#e74c3c",     # Red
            "edge_or": "#f39c12"       # Orange
        }
    
    def load_hypergraph(self, file_path: str) -> Dict:
        """Load hypergraph from JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_networkx_from_hypergraph(self, hypergraph: Dict) -> nx.DiGraph:
        """Recreate NetworkX graph from hypergraph structure"""
        G = nx.DiGraph()
        
        # Add nodes
        for node in hypergraph['nodes']:
            node_type = "concept"
            if isinstance(node['id'], str) and ("AND" in node['id'] or "OR" in node['id']):
                node_type = "virtual"
            
            G.add_node(
                node['id'],
                label=node['label'],
                node_type=node_type
            )
        
        # Rebuild edges from hyperedges
        for hyperedge in hypergraph['hyperedges']:
            tail_ids = hyperedge['tail']
            head_id = hyperedge['head']
            edge_type = hyperedge['type']
            
            if edge_type == "SINGLE":
                G.add_edge(tail_ids[0], head_id, type=edge_type)
            else:
                # Add virtual node if not exists
                virtual_id = f"{edge_type.split('_')[0]}_{hyperedge['id']}"
                if not G.has_node(virtual_id):
                    label = "∧" if "AND" in edge_type else "∨"
                    G.add_node(virtual_id, label=label, node_type="virtual")
                
                # Add edges
                for tail_id in tail_ids:
                    G.add_edge(tail_id, virtual_id, type="input")
                G.add_edge(virtual_id, head_id, type=edge_type)
        
        return G
    
    def visualize_simple(
        self,
        hypergraph: Dict,
        video_id: str,
        layout: str = "spring"
    ):
        """
        Create a simple visualization without virtual nodes
        
        Args:
            hypergraph: Hypergraph structure
            video_id: Video identifier
            layout: Layout algorithm (spring, circular, kamada_kawai)
        """
        G = nx.DiGraph()
        
        # Add only concept nodes
        concept_nodes = [n for n in hypergraph['nodes'] if not isinstance(n['id'], str)]
        for node in concept_nodes:
            G.add_node(node['id'], label=node['label'])
        
        # Add direct edges (simplified)
        edge_colors = []
        edge_labels = {}
        
        for hyperedge in hypergraph['hyperedges']:
            tail_ids = hyperedge['tail']
            head_id = hyperedge['head']
            edge_type = hyperedge['type']
            
            # For multi-tail edges, create edges from each tail to head
            for tail_id in tail_ids:
                if G.has_node(tail_id) and G.has_node(head_id):
                    G.add_edge(tail_id, head_id)
                    
                    # Determine color
                    if edge_type == "SINGLE":
                        edge_colors.append(self.colors['edge_single'])
                    elif "AND" in edge_type:
                        edge_colors.append(self.colors['edge_and'])
                    else:
                        edge_colors.append(self.colors['edge_or'])
                    
                    # Add label for multi-tail
                    if len(tail_ids) > 1:
                        edge_labels[(tail_id, head_id)] = edge_type.replace("_", " ")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            try:
                pos = graphviz_layout(G, prog='dot')
            except:
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes
        node_labels = nx.get_node_attributes(G, 'label')
        
        # Wrap long labels intelligently
        import textwrap
        wrapped_labels = {
            k: "\n".join(textwrap.wrap(v, width=15)) for k, v in node_labels.items()
        }
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color=self.colors['concept'],
            node_size=3500,
            alpha=0.9,
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors if edge_colors else self.colors['edge_single'],
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            width=2,
            alpha=0.7,
            ax=ax,
            connectionstyle="arc3,rad=0.1"
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            node_labels,
            font_size=10,
            font_weight='bold',
            font_color='white',
            ax=ax
        )
        
        # Title and legend
        metadata = hypergraph['metadata']
        title = f"Concept Hypergraph: {video_id}\n"
        title += f"Domain: {metadata.get('domain', 'Unknown')} | "
        title += f"Concepts: {metadata['num_nodes']} | "
        title += f"Relations: {metadata['num_hyperedges']}"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=self.colors['edge_single'], label='SINGLE'),
            mpatches.Patch(color=self.colors['edge_and'], label='AND (Hard/Soft)'),
            mpatches.Patch(color=self.colors['edge_or'], label='OR')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save
        output_file = self.output_dir / f"{video_id}_simple.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved simple visualization to {output_file}")
    
    def visualize_with_virtual_nodes(
        self,
        hypergraph: Dict,
        video_id: str
    ):
        """
        Create visualization with virtual AND/OR nodes
        
        Args:
            hypergraph: Hypergraph structure
            video_id: Video identifier
        """
        G = self.create_networkx_from_hypergraph(hypergraph)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Layout
        try:
            pos = graphviz_layout(G, prog='dot')
        except:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Separate nodes by type
        concept_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'concept']
        virtual_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'virtual']
        
        # Draw concept nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=concept_nodes,
            node_color=self.colors['concept'],
            node_size=3000,
            alpha=0.9,
            ax=ax,
            node_shape='o'
        )
        
        # Draw virtual nodes (smaller, different color)
        if virtual_nodes:
            virtual_colors = [
                self.colors['virtual_and'] if 'AND' in str(n) else self.colors['virtual_or']
                for n in virtual_nodes
            ]
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=virtual_nodes,
                node_color=virtual_colors,
                node_size=1500,
                alpha=0.8,
                ax=ax,
                node_shape='s'
            )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            arrows=True,
            arrowsize=15,
            arrowstyle='->',
            width=2,
            alpha=0.6,
            ax=ax,
            edge_color='#7f8c8d',
            connectionstyle="arc3,rad=0.1"
        )
        
        # Labels
        labels = nx.get_node_attributes(G, 'label')
        
        # Wrap long labels
        import textwrap
        wrapped_labels = {
            k: "\n".join(textwrap.wrap(v, width=15)) for k, v in labels.items()
        }
        
        nx.draw_networkx_labels(
            G, pos,
            wrapped_labels,
            font_size=8,
            font_weight='bold',
            font_color='white',
            ax=ax
        )
        
        # Title
        metadata = hypergraph['metadata']
        title = f"Concept Hypergraph (with Virtual Nodes): {video_id}\n"
        title += f"Domain: {metadata.get('domain', 'Unknown')}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=self.colors['concept'], label='Concept'),
            mpatches.Patch(color=self.colors['virtual_and'], label='AND Node'),
            mpatches.Patch(color=self.colors['virtual_or'], label='OR Node')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save
        output_file = self.output_dir / f"{video_id}_with_virtual.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved visualization with virtual nodes to {output_file}")
    
    def visualize_video(self, video_id: str, both_styles: bool = True):
        """
        Visualize a single video's hypergraph
        
        Args:
            video_id: Video identifier
            both_styles: If True, create both simple and virtual node visualizations
        """
        # Load hypergraph
        hypergraph_file = self.input_dir / f"{video_id}_hypergraph_structure.json"
        
        if not hypergraph_file.exists():
            print(f"✗ Hypergraph file not found for {video_id}")
            return
        
        hypergraph = self.load_hypergraph(hypergraph_file)
        
        print(f"Visualizing: {video_id}")
        
        # Create simple visualization
        self.visualize_simple(hypergraph, video_id)
        
        # Create visualization with virtual nodes
        if both_styles:
            self.visualize_with_virtual_nodes(hypergraph, video_id)
    
    def visualize_all(self):
        """Visualize all hypergraphs"""
        hypergraph_files = list(self.input_dir.glob("*_hypergraph_structure.json"))
        
        if not hypergraph_files:
            print("No hypergraph files found!")
            return
        
        for hg_file in hypergraph_files:
            video_id = hg_file.stem.replace("_hypergraph_structure", "")
            
            print(f"\n{'='*60}")
            print(f"Visualizing: {video_id}")
            print(f"{'='*60}")
            
            try:
                self.visualize_video(video_id)
            except Exception as e:
                print(f"✗ Error visualizing {video_id}: {e}")


if __name__ == "__main__":
    visualizer = HypergraphVisualizer()
    visualizer.visualize_all()
