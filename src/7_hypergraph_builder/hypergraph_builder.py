"""
Step 7: Build Concept Hypergraph - Create labeled directed hypergraph
"""
import json
from pathlib import Path
from typing import Dict, List, Set
import networkx as nx


class HypergraphBuilder:
    """Build concept hypergraphs from classified relations"""
    
    def __init__(
        self,
        input_dir: str = "data/processed",
        output_dir: str = "data/output"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_hypergraph(self, hypergraph_data: Dict) -> Dict:
        """
        Create hypergraph structure from classified relations
        
        Args:
            hypergraph_data: Formatted hypergraph data
            
        Returns:
            Hypergraph dictionary with nodes and hyperedges
        """
        concepts = hypergraph_data['concepts']
        relations = hypergraph_data['relations']
        
        # Create nodes (concepts)
        nodes = [{"id": i, "label": concept} for i, concept in enumerate(concepts)]
        
        # Create concept to ID mapping
        concept_to_id = {concept: i for i, concept in enumerate(concepts)}
        
        # Create hyperedges
        hyperedges = []
        
        for idx, relation in enumerate(relations):
            tail_concepts = relation['tail']
            head_concept = relation['head']
            
            # Get node IDs
            tail_ids = [concept_to_id[c] for c in tail_concepts if c in concept_to_id]
            head_id = concept_to_id.get(head_concept)
            
            if not tail_ids or head_id is None:
                continue
            
            hyperedge = {
                "id": idx,
                "tail": tail_ids,
                "head": head_id,
                "tail_labels": tail_concepts,
                "head_label": head_concept,
                "type": relation['type'],
                "confidence": relation['confidence'],
                "sentence": relation.get('sentence', '')
            }
            
            hyperedges.append(hyperedge)
        
        return {
            "nodes": nodes,
            "hyperedges": hyperedges,
            "metadata": {
                "num_nodes": len(nodes),
                "num_hyperedges": len(hyperedges),
                "video": hypergraph_data['video'],
                "domain": hypergraph_data['domain']
            }
        }
    
    def to_networkx_graph(self, hypergraph: Dict) -> nx.DiGraph:
        """
        Convert hypergraph to NetworkX directed graph
        
        For hyperedges with multiple tail nodes, we create:
        - Individual edges from each tail to head (for visualization)
        - Or a "virtual node" representing the conjunction
        
        Args:
            hypergraph: Hypergraph dictionary
            
        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()
        
        # Add nodes
        for node in hypergraph['nodes']:
            G.add_node(node['id'], label=node['label'])
        
        # Add edges
        for hyperedge in hypergraph['hyperedges']:
            tail_ids = hyperedge['tail']
            head_id = hyperedge['head']
            edge_type = hyperedge['type']
            confidence = hyperedge['confidence']
            
            if edge_type == "SINGLE":
                # Simple edge: tail[0] → head
                G.add_edge(
                    tail_ids[0], 
                    head_id,
                    type=edge_type,
                    confidence=confidence,
                    label=""
                )
            
            elif edge_type in ["HARD_AND", "SOFT_AND"]:
                # Multiple prerequisites: create virtual AND node
                virtual_node_id = f"AND_{hyperedge['id']}"
                G.add_node(
                    virtual_node_id,
                    label=f"{'∧' if edge_type == 'HARD_AND' else '∧?'}",
                    node_type="virtual"
                )
                
                # Connect tail concepts to virtual node
                for tail_id in tail_ids:
                    G.add_edge(tail_id, virtual_node_id, type="input", confidence=confidence)
                
                # Connect virtual node to head
                G.add_edge(virtual_node_id, head_id, type=edge_type, confidence=confidence)
            
            elif edge_type == "OR":
                # OR relationship: create virtual OR node
                virtual_node_id = f"OR_{hyperedge['id']}"
                G.add_node(
                    virtual_node_id,
                    label="∨",
                    node_type="virtual"
                )
                
                # Connect tail concepts to virtual node
                for tail_id in tail_ids:
                    G.add_edge(tail_id, virtual_node_id, type="input", confidence=confidence)
                
                # Connect virtual node to head
                G.add_edge(virtual_node_id, head_id, type=edge_type, confidence=confidence)
        
        return G
    
    def build_from_file(self, hypergraph_file: str, video_id: str) -> Dict:
        """
        Build hypergraph from hypergraph JSON file
        
        Args:
            hypergraph_file: Path to hypergraph JSON
            video_id: Video identifier
            
        Returns:
            Complete hypergraph structure
        """
        # Load hypergraph data
        with open(hypergraph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Building hypergraph with {data['statistics']['total_concepts']} concepts...")
        
        # Create hypergraph
        hypergraph = self.create_hypergraph(data)
        
        # Create NetworkX graph
        nx_graph = self.to_networkx_graph(hypergraph)
        
        # Add NetworkX graph as adjacency list
        hypergraph['networkx_adjacency'] = nx.to_dict_of_lists(nx_graph)
        
        # Add graph statistics
        hypergraph['metadata']['graph_statistics'] = {
            "num_edges": nx_graph.number_of_edges(),
            "density": nx.density(nx_graph),
            "is_dag": nx.is_directed_acyclic_graph(nx_graph)
        }
        
        return hypergraph
    
    def save_hypergraph(self, hypergraph: Dict, video_id: str):
        """Save hypergraph structure"""
        output_file = self.output_dir / f"{video_id}_hypergraph_structure.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(hypergraph, f, ensure_ascii=False, indent=2)
        
        meta = hypergraph['metadata']
        print(f"✓ Built hypergraph:")
        print(f"    Nodes: {meta['num_nodes']}")
        print(f"    Hyperedges: {meta['num_hyperedges']}")
        if 'graph_statistics' in meta:
            print(f"    Is DAG: {meta['graph_statistics']['is_dag']}")
        print(f"✓ Saved to {output_file}")
    
    def process_all_hypergraphs(self):
        """Process all hypergraph files"""
        hypergraph_files = list(self.input_dir.glob("*_hypergraph.json"))
        
        if not hypergraph_files:
            print("No hypergraph files found!")
            return
        
        for hg_file in hypergraph_files:
            video_id = hg_file.stem.replace("_hypergraph", "")
            
            print(f"\n{'='*60}")
            print(f"Building hypergraph: {video_id}")
            print(f"{'='*60}")
            
            try:
                hypergraph = self.build_from_file(hg_file, video_id)
                self.save_hypergraph(hypergraph, video_id)
                
            except Exception as e:
                print(f"✗ Error building hypergraph for {video_id}: {e}")


if __name__ == "__main__":
    builder = HypergraphBuilder()
    builder.process_all_hypergraphs()
