#!/usr/bin/env python3
"""
Hybrid Implicit Prerequisite Detection (Methods 1-4, No LLM)
Combines: Semantic+Temporal, Co-occurrence, Domain Patterns, Transitivity
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
from typing import List, Dict, Tuple, Any

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers not available, using basic similarity")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import networkx as nx

class HybridImplicitDetector:
    def __init__(self):
        """Initialize the hybrid detector"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("� Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = None
        
        # NOTE: we no longer use hard-coded chemistry_patterns for inference,
        # but keep the attribute here if you want to log or inspect later.
        self.chemistry_patterns = {}

    def find_concept_timestamp(self, concept: str, transcript: List[Dict]) -> float:
        """Find first occurrence timestamp of a concept"""
        concept_terms = concept.lower().split()
        
        for chunk in transcript:
            text = chunk.get('text', '').lower()
            if any(term in text for term in concept_terms):
                return chunk.get('start', 0)
        return 0
    
    def get_transcript_window(self, transcript: List[Dict], start_time: float, end_time: float) -> str:
        """Extract transcript text within time window"""
        window_text = []
        for chunk in transcript:
            chunk_start = chunk.get('start', 0)
            chunk_end = chunk_start + chunk.get('duration', 0)
            
            if chunk_start >= start_time and chunk_end <= end_time:
                window_text.append(chunk.get('text', ''))
        
        return ' '.join(window_text)
    
    def calculate_semantic_similarity(self, concept_a: str, concept_b_window: str) -> float:
        """Calculate semantic similarity between concept and explanation window.

        Note: This is still video-only in the sense that it uses only the
        words present in the transcript window; the embedding model is
        pre-trained but does not inject explicit prerequisite rules.
        """
        if not self.model:
            # Fallback: simple keyword overlap
            concept_words = set(concept_a.lower().split())
            window_words = set(concept_b_window.lower().split())
            if len(concept_words) == 0:
                return 0
            overlap = len(concept_words.intersection(window_words))
            return overlap / len(concept_words)
        
        # Use sentence transformer
        embeddings = self.model.encode([concept_a, concept_b_window])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    # ──────────────────────────────────────────────
    #  VIDEO-ONLY EXPLANATION OVERLAP (NO DOMAIN PATTERNS)
    # ──────────────────────────────────────────────

    def _build_explanation_windows(self, concepts: List[Dict], transcript: List[Dict], window_before: float = 20.0, window_after: float = 60.0) -> Dict[str, str]:
        """Build local explanation windows around each concept based only on this video.

        For each concept, we find its first timestamp in the transcript and
        take a window [t - window_before, t + window_after].
        """

        windows: Dict[str, str] = {}
        for c in concepts:
            name = c["concept"]
            t = self.find_concept_timestamp(name, transcript)
            # If not found, skip
            if t is None:
                continue
            window_text = self.get_transcript_window(transcript, t - window_before, t + window_after)
            if window_text.strip():
                windows[name] = window_text
        return windows

    def _bag_of_words(self, text: str) -> Dict[str, int]:
        """Very simple bag-of-words from transcript text."""
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        # Drop very short/common tokens later via frequency filtering if needed
        bow: Dict[str, int] = defaultdict(int)
        for tok in tokens:
            if len(tok) > 2:  # ignore very short words like "is", "to"
                bow[tok] += 1
        return bow

    def _jaccard(self, set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union

    def method3_video_explanation_overlap(self, concepts: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Method 3 (redefined): video-only explanation-window overlap.

        This replaces the previous domain-pattern-based method. It only
        uses this video's transcript and timestamps, looking at how much
        the vocabulary of concept A's explanation reappears inside the
        explanation window of concept B (with A earlier than B).
        """

        print("\n🧩 Method 3: Video Explanation Overlap (no external knowledge)...")
        dependencies: List[Dict] = []

        concept_names = [c["concept"] for c in concepts]
        windows = self._build_explanation_windows(concepts, transcript)
        bows: Dict[str, Dict[str, int]] = {name: self._bag_of_words(windows[name]) for name in windows}

        for concept_a in concept_names:
            for concept_b in concept_names:
                if concept_a == concept_b:
                    continue

                # Temporal: A must be introduced before B
                t_a = self.find_concept_timestamp(concept_a, transcript)
                t_b = self.find_concept_timestamp(concept_b, transcript)
                if t_a is None or t_b is None or t_a >= t_b:
                    continue

                bow_a = bows.get(concept_a)
                bow_b = bows.get(concept_b)
                if not bow_a or not bow_b:
                    continue

                # Use Jaccard over word types as a simple overlap score
                vocab_a = set(bow_a.keys())
                vocab_b = set(bow_b.keys())
                overlap = self._jaccard(vocab_a, vocab_b)

                # Ignore very low overlap to avoid noise; threshold is tunable
                if overlap > 0.10:
                    dependencies.append({
                        "prerequisite": concept_a,
                        "concept": concept_b,
                        "confidence": float(overlap),
                        "method": "video_explanation_overlap",
                        "details": {
                            "overlap": float(overlap),
                            "vocab_a_size": len(vocab_a),
                            "vocab_b_size": len(vocab_b),
                            "t_a": t_a,
                            "t_b": t_b
                        }
                    })

        print(f"   Found {len(dependencies)} video-explanation-overlap dependencies")
        return dependencies
    
    def concept_appears_in_window(self, concept: str, window_text: str) -> bool:
        """Check if concept terminology appears in window"""
        concept_terms = concept.lower().split()
        window_lower = window_text.lower()
        
        # Check for exact concept name or key terms
        if concept.lower() in window_lower:
            return True
        
        # Check if majority of concept terms appear
        term_count = sum(1 for term in concept_terms if term in window_lower)
        return term_count >= len(concept_terms) * 0.6
    
    def method1_semantic_temporal(self, concepts: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Method 1: Semantic Similarity + Temporal Analysis"""
        print("\n🔍 Method 1: Semantic + Temporal Analysis...")
        dependencies = []
        
        concept_names = [c['concept'] for c in concepts]
        
        for i, concept_a in enumerate(concept_names):
            for j, concept_b in enumerate(concept_names):
                if i >= j:  # Only check A before B
                    continue
                
                # Get timestamps
                timestamp_a = self.find_concept_timestamp(concept_a, transcript)
                timestamp_b = self.find_concept_timestamp(concept_b, transcript)
                
                if timestamp_b <= timestamp_a:  # B must come after A
                    continue
                
                # Get explanation window around concept B
                window_b = self.get_transcript_window(
                    transcript, 
                    timestamp_b - 30,  # 30s before
                    timestamp_b + 90   # 90s after
                )
                
                if not window_b.strip():
                    continue
                
                # Semantic overlap analysis
                overlap_score = self.calculate_semantic_similarity(concept_a, window_b)
                
                # Temporal proximity bonus
                time_gap = timestamp_b - timestamp_a
                proximity_score = max(0, 1 - (time_gap / 300))  # 5min max
                
                combined_score = overlap_score * 0.7 + proximity_score * 0.3
                
                if combined_score > 0.35:  # Threshold
                    dependencies.append({
                        'prerequisite': concept_a,
                        'concept': concept_b,
                        'confidence': combined_score,
                        'method': 'semantic_temporal',
                        'details': {
                            'semantic_score': overlap_score,
                            'temporal_score': proximity_score,
                            'time_gap': time_gap
                        }
                    })
        
        print(f"   Found {len(dependencies)} semantic-temporal dependencies")
        return dependencies
    
    def method2_cooccurrence(self, concepts: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Method 2: Keyword Co-occurrence + Window Analysis"""
        print("\n📊 Method 2: Co-occurrence Analysis...")
        dependencies = []
        window_size = 60  # 1 minute windows
        
        concept_names = [c['concept'] for c in concepts]
        
        # Create sliding windows
        windows = []
        for i in range(0, len(transcript) - 1):
            start_time = transcript[i].get('start', 0)
            window_text = self.get_transcript_window(
                transcript, start_time, start_time + window_size
            )
            if window_text.strip():
                windows.append({
                    'start': start_time,
                    'text': window_text
                })
        
        for concept_a in concept_names:
            for concept_b in concept_names:
                if concept_a == concept_b:
                    continue
                
                cooccurrence_count = 0
                total_windows_with_b = 0
                
                for window in windows:
                    has_a = self.concept_appears_in_window(concept_a, window['text'])
                    has_b = self.concept_appears_in_window(concept_b, window['text'])
                    
                    if has_b:
                        total_windows_with_b += 1
                        if has_a:
                            cooccurrence_count += 1
                
                if total_windows_with_b > 0:
                    cooccurrence_rate = cooccurrence_count / total_windows_with_b
                    
                    if cooccurrence_rate > 0.25:  # 25% co-occurrence
                        # Check temporal order
                        first_a = self.find_concept_timestamp(concept_a, transcript)
                        first_b = self.find_concept_timestamp(concept_b, transcript)
                        
                        if first_a < first_b:
                            dependencies.append({
                                'prerequisite': concept_a,
                                'concept': concept_b,
                                'confidence': cooccurrence_rate,
                                'method': 'cooccurrence',
                                'details': {
                                    'cooccurrence_rate': cooccurrence_rate,
                                    'windows_with_both': cooccurrence_count,
                                    'total_windows_with_concept': total_windows_with_b
                                }
                            })
        
        print(f"   Found {len(dependencies)} co-occurrence dependencies")
        return dependencies
    
    def find_matching_concept(self, pattern: str, concepts: List[str]) -> str:
        """Find concept that matches domain pattern"""
        pattern_words = set(pattern.lower().split('_'))
        
        best_match = None
        best_score = 0
        
        for concept in concepts:
            concept_words = set(concept.lower().split())
            overlap = len(pattern_words.intersection(concept_words))
            if overlap > best_score:
                best_score = overlap
                best_match = concept
        
        return best_match if best_score > 0 else None
    
    # NOTE: The old domain-pattern-based method has been fully removed to
    # ensure we rely only on information present in this video's transcript.
    
    def method4_transitivity(self, explicit_deps: List[Dict], concepts: List[Dict]) -> List[Dict]:
        """Method 4: Graph-Based Transitivity Analysis"""
        print("\n🔗 Method 4: Transitivity Analysis...")
        
        concept_names = [c['concept'] for c in concepts]
        G = nx.DiGraph()
        
        # Add all concepts as nodes
        for concept_name in concept_names:
            G.add_node(concept_name)
        
        # Add explicit dependencies as edges
        for dep in explicit_deps:
            G.add_edge(dep['prerequisite'], dep['concept'], weight=dep['confidence'])
        
        transitive_deps = []
        
        for concept in concept_names:
            # Find all concepts with paths to this concept
            for other_concept in concept_names:
                if other_concept != concept and G.has_node(other_concept):
                    try:
                        if nx.has_path(G, other_concept, concept):
                            path_length = nx.shortest_path_length(G, other_concept, concept)
                            if 2 <= path_length <= 3:  # 1-2 intermediate concepts
                                path = nx.shortest_path(G, other_concept, concept)
                                
                                # Calculate confidence based on path strength
                                path_weights = []
                                for i in range(len(path) - 1):
                                    if G.has_edge(path[i], path[i + 1]):
                                        edge_data = G.get_edge_data(path[i], path[i + 1])
                                        path_weights.append(edge_data.get('weight', 0.5))
                                
                                if path_weights:
                                    # Geometric mean of path weights
                                    confidence = np.prod(path_weights) ** (1.0 / len(path_weights))
                                    confidence = confidence * (0.8 if path_length == 2 else 0.6)  # Decay for longer paths
                                    
                                    if confidence > 0.3:
                                        transitive_deps.append({
                                            'prerequisite': other_concept,
                                            'concept': concept,
                                            'confidence': confidence,
                                            'method': 'transitive',
                                            'details': {
                                                'path': path,
                                                'path_length': path_length,
                                                'path_weights': path_weights
                                            }
                                        })
                    except:
                        continue
        
        print(f"   Found {len(transitive_deps)} transitive dependencies")
        return transitive_deps
    
    def combine_and_rank_dependencies(self, all_dependencies: List[Dict]) -> List[Dict]:
        """Combine dependencies from all methods and remove duplicates"""
        print("\n🔄 Combining and ranking dependencies...")
        
        # Group by (prerequisite, concept) pair
        dep_groups = defaultdict(list)
        for dep in all_dependencies:
            key = (dep['prerequisite'], dep['concept'])
            dep_groups[key].append(dep)
        
        final_deps = []
        for (prereq, concept), deps in dep_groups.items():
            if len(deps) == 1:
                # Single method detection
                final_deps.append(deps[0])
            else:
                # Multiple methods - combine confidences
                methods = [d['method'] for d in deps]
                confidences = [d['confidence'] for d in deps]
                
                # Weighted combination (higher weight for multiple methods)
                combined_confidence = np.mean(confidences) * (1 + 0.1 * (len(deps) - 1))
                combined_confidence = min(combined_confidence, 0.95)  # Cap at 0.95
                
                final_deps.append({
                    'prerequisite': prereq,
                    'concept': concept,
                    'confidence': combined_confidence,
                    'method': 'combined',
                    'details': {
                        'methods': methods,
                        'individual_confidences': confidences,
                        'method_count': len(deps)
                    }
                })
        
        # Sort by confidence
        final_deps.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"   Final: {len(final_deps)} unique dependencies")
        return final_deps
    
    def detect_implicit_prerequisites(self, video_id: str) -> Dict:
        """Main function to detect implicit prerequisites using hybrid approach"""
        print(f"\n🔍 HYBRID IMPLICIT PREREQUISITE DETECTION: {video_id}")
        print("=" * 60)
        
        # Load data
        concepts_file = f"data/processed/{video_id}_concepts.json"
        transcript_file = f"data/raw/{video_id}_transcript_en.json"
        
        if not Path(concepts_file).exists():
            raise FileNotFoundError(f"Concepts file not found: {concepts_file}")
        if not Path(transcript_file).exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_file}")
        
        with open(concepts_file, 'r') as f:
            concepts_data = json.load(f)
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
        
        concepts = concepts_data['concepts']
        transcript = transcript_data['transcript']
        
        print(f"📋 Processing {len(concepts)} concepts from {len(transcript)} transcript chunks")
        
        # Apply all methods
        all_dependencies = []
        
        # Method 1: Semantic + Temporal
        semantic_deps = self.method1_semantic_temporal(concepts, transcript)
        all_dependencies.extend(semantic_deps)
        
        # Method 2: Co-occurrence
        cooccurrence_deps = self.method2_cooccurrence(concepts, transcript)
        all_dependencies.extend(cooccurrence_deps)
        
        # Method 3: Video Explanation Overlap (no domain patterns)
        overlap_deps = self.method3_video_explanation_overlap(concepts, transcript)
        all_dependencies.extend(overlap_deps)
        
        # Method 4: Transitivity (based on dependencies found so far)
        current_deps = semantic_deps + cooccurrence_deps + overlap_deps
        transitive_deps = self.method4_transitivity(current_deps, concepts)
        all_dependencies.extend(transitive_deps)
        
        # Combine and rank
        final_deps = self.combine_and_rank_dependencies(all_dependencies)
        
        # Create output structure
        result = {
            'video_id': video_id,
            'method': 'hybrid_implicit_detection',
            'total_dependencies': len(final_deps),
            'method_breakdown': {
                'semantic_temporal': len(semantic_deps),
                'cooccurrence': len(cooccurrence_deps),
                'video_explanation_overlap': len(overlap_deps),
                'transitive': len(transitive_deps),
                'final_unique': len(final_deps)
            },
            'dependencies': final_deps
        }
        
        return result


def main():
    """Run hybrid implicit detection for a given video id.

    Usage:
      python hybrid_implicit_detector.py           # defaults to video_2
      python hybrid_implicit_detector.py video_4   # runs for video_4
    """

    import sys

    video_id = sys.argv[1] if len(sys.argv) > 1 else "video_2"

    detector = HybridImplicitDetector()

    try:
        result = detector.detect_implicit_prerequisites(video_id)

        # Save results
        output_file = f"data/processed/{video_id}_implicit_prerequisites.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        print("\n✅ RESULTS SUMMARY")
        print("=" * 50)
        print(f"Video: {video_id}")
        print(f"📊 Total dependencies found: {result['total_dependencies']}")
        print(f"🔤 Semantic+Temporal: {result['method_breakdown']['semantic_temporal']}")
        print(f"📈 Co-occurrence: {result['method_breakdown']['cooccurrence']}")
        print(f"🧩 Video explanation overlap: {result['method_breakdown']['video_explanation_overlap']}")
        print(f"🔗 Transitive: {result['method_breakdown']['transitive']}")
        print(f"💾 Saved to: {output_file}")

        # Show top dependencies
        print("\n🏆 TOP 10 DEPENDENCIES:")
        for i, dep in enumerate(result["dependencies"][:10], 1):
            method = dep["method"]
            confidence = dep["confidence"]
            prereq = dep["prerequisite"]
            concept = dep["concept"]
            print(f"   {i:2d}. {prereq} → {concept}")
            print(f"       Method: {method}, Confidence: {confidence:.3f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()