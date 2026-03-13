"""
Step 6: Relation Type Classification - Classify prerequisite relations
"""
import json
from pathlib import Path
from typing import Dict, List


class RelationClassifier:
    """Classify prerequisite relations into SINGLE, HARD_AND, SOFT_AND, OR"""
    
    RELATION_TYPES = {
        "SINGLE": "Single prerequisite (A → B)",
        "HARD_AND": "Both prerequisites required ({A, B} → C)",
        "SOFT_AND": "Both helpful but not strictly required ({A, B} → C)",
        "OR": "Either prerequisite sufficient ({A | B} → C)"
    }
    
    def __init__(
        self,
        input_dir: str = "data/processed",
        output_dir: str = "data/processed"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
    
    def classify_relation(self, prerequisite: Dict) -> str:
        """
        Classify a prerequisite relationship
        
        Args:
            prerequisite: Prerequisite dictionary from detector
            
        Returns:
            Relation type (SINGLE, HARD_AND, SOFT_AND, OR)
        """
        tail_count = len(prerequisite['tail'])
        conjunction = prerequisite['conjunction_type']
        strength = prerequisite['strength']
        
        # Single prerequisite
        if tail_count == 1:
            return "SINGLE"
        
        # Multiple prerequisites with OR
        if conjunction == "or":
            return "OR"
        
        # Multiple prerequisites with AND
        if conjunction == "and":
            if strength == "hard":
                return "HARD_AND"
            else:
                return "SOFT_AND"
        
        # Default for multiple concepts without clear conjunction
        if tail_count > 1:
            # If strong indicator, assume HARD_AND
            if strength == "hard":
                return "HARD_AND"
            else:
                return "SOFT_AND"
        
        return "SINGLE"
    
    def classify_all_relations(
        self,
        prerequisites_file: str,
        video_id: str
    ) -> Dict:
        """
        Classify all prerequisite relations from a file
        
        Args:
            prerequisites_file: Path to prerequisites JSON
            video_id: Video identifier
            
        Returns:
            Dictionary with classified relations
        """
        # Load prerequisites
        with open(prerequisites_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prerequisites = data.get('prerequisites', [])
        
        print(f"Classifying {len(prerequisites)} relations...")
        
        # Classify each prerequisite
        classified_relations = []
        relation_type_counts = {rt: 0 for rt in self.RELATION_TYPES.keys()}
        
        for prereq in prerequisites:
            relation_type = self.classify_relation(prereq)
            
            # Add relation type to prerequisite
            classified_prereq = {**prereq, "relation_type": relation_type}
            classified_relations.append(classified_prereq)
            
            relation_type_counts[relation_type] += 1
        
        return {
            "video_id": video_id,
            "domain": data.get('domain'),
            "total_relations": len(classified_relations),
            "relation_type_counts": relation_type_counts,
            "relations": classified_relations,
            "concepts": data.get('concepts', [])
        }
    
    def format_for_hypergraph(self, classified: Dict) -> Dict:
        """
        Format classified relations for hypergraph construction
        
        Args:
            classified: Classified relations dictionary
            
        Returns:
            Formatted dictionary ready for hypergraph
        """
        formatted_relations = []
        
        for relation in classified['relations']:
            formatted_relations.append({
                "tail": relation['tail'],
                "head": relation['head'],
                "type": relation['relation_type'],
                "confidence": relation['confidence'],
                "sentence": relation['sentence_text']
            })
        
        return {
            "video": classified['video_id'],
            "domain": classified['domain'],
            "concepts": classified['concepts'],
            "relations": formatted_relations,
            "statistics": {
                "total_concepts": len(classified['concepts']),
                "total_relations": classified['total_relations'],
                "relation_types": classified['relation_type_counts']
            }
        }
    
    def save_classified(self, classified: Dict, video_id: str):
        """Save classified relations"""
        output_file = self.output_dir / f"{video_id}_classified.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(classified, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Classified {classified['total_relations']} relations")
        print(f"  Relation type breakdown:")
        for rel_type, count in classified['relation_type_counts'].items():
            if count > 0:
                print(f"    - {rel_type}: {count}")
        print(f"✓ Saved to {output_file}")
    
    def save_hypergraph_format(self, formatted: Dict, video_id: str):
        """Save in hypergraph-ready format"""
        output_file = self.output_dir / f"{video_id}_hypergraph.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved hypergraph format to {output_file}")
    
    def process_all_prerequisites(self):
        """Process all prerequisite files"""
        prerequisite_files = list(self.input_dir.glob("*_prerequisites.json"))
        
        if not prerequisite_files:
            print("No prerequisite files found!")
            return
        
        for prereq_file in prerequisite_files:
            video_id = prereq_file.stem.replace("_prerequisites", "")
            
            print(f"\n{'='*60}")
            print(f"Classifying relations: {video_id}")
            print(f"{'='*60}")
            
            try:
                classified = self.classify_all_relations(prereq_file, video_id)
                self.save_classified(classified, video_id)
                
                # Also save in hypergraph format
                formatted = self.format_for_hypergraph(classified)
                self.save_hypergraph_format(formatted, video_id)
                
            except Exception as e:
                print(f"✗ Error classifying {video_id}: {e}")


if __name__ == "__main__":
    classifier = RelationClassifier()
    classifier.process_all_prerequisites()
