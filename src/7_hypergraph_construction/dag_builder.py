"""
Step 7: Build Concept DAG from verified prerequisites
"""
import json
from pathlib import Path
from typing import List, Dict
import networkx as nx


class ConceptDAGBuilder:
    """Build Directed Acyclic Graph from verified prerequisite relationships"""
    
    def __init__(self,
                 prerequisites_dir: str = "data/processed",
                 output_dir: str = "data/output"):
        self.prerequisites_dir = Path(prerequisites_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build_dag(self, concepts: List[Dict], edges: List[Dict]) -> nx.DiGraph:
        """
        Build DAG from concepts and prerequisite edges
        
        Args:
            concepts: List of concept dictionaries
            edges: List of verified prerequisite edges
        
        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()
        
        # Add nodes
        for i, concept_data in enumerate(concepts):
            concept = concept_data['concept']
            G.add_node(i, label=concept, **concept_data)
        
        # Create concept → node_id mapping
        concept_to_id = {c['concept']: i for i, c in enumerate(concepts)}
        
        # Add edges (prerequisite → target)
        for edge in edges:
            prereq = edge['prerequisite']
            target = edge['target']
            
            if prereq in concept_to_id and target in concept_to_id:
                prereq_id = concept_to_id[prereq]
                target_id = concept_to_id[target]
                
                G.add_edge(
                    prereq_id,
                    target_id,
                    prerequisite=prereq,
                    target=target,
                    confidence=edge.get('llm_confidence', 0.0),
                    teaching_signals=edge.get('teaching_signals', []),
                    time_gap=edge.get('time_gap', 0)
                )
        
        return G
    
    def process_video(self, video_id: str) -> Dict:
        """Build DAG for a single video"""
        
        # Load concepts
        concepts_file = self.prerequisites_dir / f"{video_id}_concepts.json"
        with open(concepts_file, 'r', encoding='utf-8') as f:
            concepts_data = json.load(f)
        
        concepts = concepts_data['concepts']
        
        # Load verified prerequisites
        prerequisites_file = self.prerequisites_dir / f"{video_id}_prerequisites.json"
        with open(prerequisites_file, 'r', encoding='utf-8') as f:
            prereq_data = json.load(f)
        
        confirmed_edges = prereq_data['confirmed_edges']
        
        print(f"\n{'='*60}")
        print(f"Building DAG: {video_id}")
        print(f"Concepts: {len(concepts)}")
        print(f"Verified edges: {len(confirmed_edges)}")
        print(f"{'='*60}")
        
        # Build DAG
        G = self.build_dag(concepts, confirmed_edges)
        
        # Check if it's actually a DAG
        is_dag = nx.is_directed_acyclic_graph(G)
        
        if not is_dag:
            print("⚠️  Warning: Graph contains cycles! Removing cycles...")
            # Remove cycles by breaking weakest edges
            while not nx.is_directed_acyclic_graph(G):
                try:
                    cycle = nx.find_cycle(G)
                    # Remove the first edge in the cycle
                    G.remove_edge(cycle[0][0], cycle[0][1])
                    print(f"  Removed edge: {cycle[0]}")
                except:
                    break
        
        print(f"✓ DAG constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Convert to JSON format
        nodes = []
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            nodes.append({
                'id': node_id,
                'label': node_data['label'],
                'concept': node_data.get('concept', node_data['label'])
            })
        
        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                'source': u,
                'target': v,
                'prerequisite': data.get('prerequisite', ''),
                'target_concept': data.get('target', ''),
                'confidence': data.get('confidence', 0.0),
                'teaching_signals': data.get('teaching_signals', [])
            })
        
        # Calculate graph statistics
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_dag': nx.is_directed_acyclic_graph(G)
        }
        
        result = {
            'video_id': video_id,
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'video': video_id,
                'domain': concepts_data.get('domain', 'Unknown'),
                'graph_statistics': stats
            },
            'networkx_adjacency': nx.to_dict_of_lists(G)
        }
        
        return result
    
    def save_dag(self, result: Dict, video_id: str):
        """Save DAG structure"""
        output_file = self.output_dir / f"{video_id}_hypergraph_structure.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved DAG to {output_file}")
    
    def process_all_videos(self):
        """Process all videos"""
        prereq_files = list(self.prerequisites_dir.glob("*_prerequisites.json"))
        
        for prereq_file in prereq_files:
            video_id = prereq_file.stem.replace('_prerequisites', '')
            
            try:
                result = self.process_video(video_id)
                self.save_dag(result, video_id)
            except Exception as e:
                print(f"✗ Error processing {video_id}: {e}")
                import traceback
                traceback.print_exc()


def main():
    builder = ConceptDAGBuilder()
    builder.process_all_videos()


if __name__ == "__main__":
    main()
