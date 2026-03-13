"""
Step 4: Linguistic Normalization - Convert code-mixed phrases to standard terminology
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class LinguisticNormalizer:
    """Normalize code-mixed concepts to standard terminology"""
    
    def __init__(
        self, 
        input_dir: str = "data/processed",
        output_dir: str = "data/processed",
        config_dir: str = "config"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config_dir = Path(config_dir)
        
        # Load linguistic mappings
        self.mappings = self._load_mappings()
        
        # Load sentence transformer for semantic similarity
        print("Loading multilingual sentence transformer...")
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
    def _load_mappings(self) -> Dict:
        """Load linguistic mappings from config"""
        import yaml
        
        mapping_file = self.config_dir / "linguistic_mappings.yaml"
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def normalize_with_dictionary(self, concept: str) -> str:
        """Normalize using dictionary mapping"""
        concept_lower = concept.lower()
        
        # Check all domain mappings
        for domain, mappings in self.mappings.items():
            if domain.endswith('_mappings'):
                if concept_lower in mappings:
                    return mappings[concept_lower]
        
        return concept
    
    def normalize_with_similarity(
        self, 
        concept: str, 
        standard_terms: List[str],
        threshold: float = 0.7
    ) -> Tuple[str, float]:
        """
        Normalize using semantic similarity
        
        Args:
            concept: Code-mixed concept
            standard_terms: List of standard terminology
            threshold: Similarity threshold
            
        Returns:
            Tuple of (normalized_concept, similarity_score)
        """
        # Encode concept and standard terms
        concept_embedding = self.model.encode([concept])
        term_embeddings = self.model.encode(standard_terms)
        
        # Calculate similarities
        similarities = cosine_similarity(concept_embedding, term_embeddings)[0]
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= threshold:
            return standard_terms[best_idx], float(best_score)
        
        return concept, 0.0
    
    def get_standard_terms_for_domain(self, domain: str) -> List[str]:
        """Get list of standard terms for a domain"""
        domain_key = f"{domain.lower()}_mappings"
        
        if domain_key in self.mappings:
            # Get unique values (standard terms)
            return list(set(self.mappings[domain_key].values()))
        
        return []
    
    def normalize_concept(
        self, 
        concept: str, 
        domain: str = None,
        use_similarity: bool = True
    ) -> Dict:
        """
        Normalize a single concept
        
        Args:
            concept: Input concept
            domain: Optional domain for better matching
            use_similarity: Whether to use similarity matching
            
        Returns:
            Dictionary with normalized concept and metadata
        """
        original = concept
        
        # First try dictionary mapping
        normalized = self.normalize_with_dictionary(concept)
        method = "dictionary" if normalized != concept else None
        confidence = 1.0 if method == "dictionary" else 0.0
        
        # If no dictionary match and similarity enabled, try semantic matching
        if not method and use_similarity and domain:
            standard_terms = self.get_standard_terms_for_domain(domain)
            
            if standard_terms:
                normalized, sim_score = self.normalize_with_similarity(
                    concept, 
                    standard_terms
                )
                
                if sim_score > 0:
                    method = "similarity"
                    confidence = sim_score
        
        return {
            "original": original,
            "normalized": normalized,
            "method": method,
            "confidence": confidence,
            "changed": normalized != original
        }
    
    def normalize_video_concepts(
        self, 
        concepts_file: str, 
        video_id: str,
        domain: str = None
    ) -> Dict:
        """
        Normalize all concepts from a video
        
        Args:
            concepts_file: Path to concepts JSON file
            video_id: Video identifier
            domain: Video domain
            
        Returns:
            Dictionary with normalized concepts
        """
        # Load concepts
        with open(concepts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        concepts = data.get('concepts', [])
        
        print(f"Normalizing {len(concepts)} concepts...")
        
        # Normalize each concept and group by normalized name
        normalized_map = {}
        changes_count = 0
        
        for concept_data in concepts:
            concept = concept_data['concept']
            
            normalized_result = self.normalize_concept(concept, domain)
            norm_name = normalized_result['normalized']
            
            if normalized_result['changed']:
                changes_count += 1
                
            merged_data = {**concept_data, **normalized_result}
            
            if norm_name not in normalized_map:
                normalized_map[norm_name] = merged_data
            else:
                # Merge logic: if we have duplicates, keep the one with the highest combined_score
                existing = normalized_map[norm_name]
                if merged_data.get('combined_score', 0) > existing.get('combined_score', 0):
                    normalized_map[norm_name] = merged_data
                elif merged_data.get('confidence', 0) > existing.get('confidence', 0):
                    normalized_map[norm_name] = merged_data
                    
        normalized_concepts = list(normalized_map.values())
        
        # Update sentences with normalized concepts
        sentences = data.get('sentences_with_concepts', [])
        for sentence in sentences:
            if 'concepts' in sentence:
                # Keep unique normalized concepts in sentences
                sentence['normalized_concepts'] = list(set([
                    self.normalize_concept(c, domain)['normalized']
                    for c in sentence['concepts']
                ]))
        
        # Count changes
        changes = changes_count
        
        return {
            "video_id": video_id,
            "domain": domain,
            "total_concepts": len(normalized_concepts),
            "concepts_normalized": changes,
            "concepts": normalized_concepts,
            "sentences_with_concepts": sentences,
            "metadata": {
                "normalization_rate": changes / len(normalized_concepts) if normalized_concepts else 0
            }
        }
    
    def save_normalized(self, normalized: Dict, video_id: str):
        """Save normalized concepts"""
        output_file = self.output_dir / f"{video_id}_normalized.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Normalized {normalized['concepts_normalized']}/{normalized['total_concepts']} concepts")
        print(f"✓ Saved to {output_file}")
    
    def process_all_concepts(self, videos_config: List[Dict] = None):
        """Process all concept files"""
        concept_files = list(self.input_dir.glob("*_concepts.json"))
        
        if not concept_files:
            print("No concept files found!")
            return
        
        # Create domain mapping
        domain_map = {}
        if videos_config:
            domain_map = {v['id']: v['domain'] for v in videos_config}
        
        for concept_file in concept_files:
            video_id = concept_file.stem.replace("_concepts", "")
            domain = domain_map.get(video_id)
            
            print(f"\n{'='*60}")
            print(f"Normalizing: {video_id}")
            if domain:
                print(f"Domain: {domain}")
            print(f"{'='*60}")
            
            try:
                normalized = self.normalize_video_concepts(concept_file, video_id, domain)
                self.save_normalized(normalized, video_id)
                
            except Exception as e:
                print(f"✗ Error normalizing {video_id}: {e}")


if __name__ == "__main__":
    from src.utils.config_loader import ConfigLoader
    
    config = ConfigLoader()
    videos_config = config.load_videos()
    
    normalizer = LinguisticNormalizer()
    normalizer.process_all_concepts(videos_config['videos'])
