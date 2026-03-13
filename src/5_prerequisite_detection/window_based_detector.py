"""
Step 5: Prerequisite Detection using ±30 second windows + teaching signals
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set


class WindowBasedPrerequisiteDetector:
    """Detect prerequisites using timestamp windows and teaching signals"""
    
    def __init__(self,
                 raw_dir: str = "data/raw",
                 concepts_dir: str = "data/processed",
                 output_dir: str = "data/processed",
                 window_seconds: int = 30):
        self.raw_dir = Path(raw_dir)
        self.concepts_dir = Path(concepts_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.window_seconds = window_seconds
        
        # Teaching signal words (English - will work with translated transcripts)
        self.teaching_signals = {
            "before", "first", "recall", "remember", "based on", "using",
            "as we saw", "earlier", "previously", "after", "then", "next",
            "requires", "need", "prerequisite", "foundation", "builds on",
            "depends on", "assuming", "given that", "now that", "since we know",
            "let's see", "let's understand", "we saw", "we discussed",
            "we learned", "as discussed", "as mentioned", "as shown"
        }
    
    def find_concept_context(self, 
                            concept: str, 
                            transcript: List[Dict],
                            window_seconds: int = 30) -> Tuple[str, float, List[Dict]]:
        """
        Find the context around where a concept is mentioned
        
        Returns:
            (context_text, timestamp, context_chunks)
        """
        # Search for concept in transcript
        for chunk in transcript:
            if concept.lower() in chunk['text'].lower():
                timestamp = chunk['start']
                
                # Get ±window_seconds around this timestamp
                context_chunks = [
                    c for c in transcript
                    if abs(c['start'] - timestamp) <= window_seconds
                ]
                
                context_text = " ".join(c['text'] for c in context_chunks)
                
                return context_text, timestamp, context_chunks
        
        # If concept not found in transcript, return empty
        return "", None, []
    
    def detect_teaching_signals(self, text: str) -> List[str]:
        """Detect teaching signal words in text"""
        text_lower = text.lower()
        found_signals = []
        
        for signal in self.teaching_signals:
            if signal in text_lower:
                found_signals.append(signal)
        
        return found_signals
    
    def find_candidate_edges(self, 
                            concepts: List[Dict], 
                            transcript: List[Dict]) -> List[Dict]:
        """
        Find candidate prerequisite edges based on:
        1. Concepts appearing in each other's ±30 sec window
        2. Timestamp ordering (earlier concept → later concept)
        3. Teaching signals present
        """
        candidate_edges = []
        
        # Build concept-to-context mapping
        concept_contexts = {}
        for concept_data in concepts:
            concept = concept_data['concept']
            context_text, timestamp, context_chunks = self.find_concept_context(
                concept, transcript, self.window_seconds
            )
            
            if timestamp is not None:
                concept_contexts[concept] = {
                    'context': context_text,
                    'timestamp': timestamp,
                    'chunks': context_chunks
                }
        
        # Find edges
        for concept_a_data in concepts:
            concept_a = concept_a_data['concept']
            
            if concept_a not in concept_contexts:
                continue
            
            context_a = concept_contexts[concept_a]
            
            for concept_b_data in concepts:
                concept_b = concept_b_data['concept']
                
                if concept_a == concept_b:
                    continue
                
                if concept_b not in concept_contexts:
                    continue
                
                context_b = concept_contexts[concept_b]
                
                # Check timestamp ordering: A must come before B
                if context_a['timestamp'] >= context_b['timestamp']:
                    continue
                
                # Check if concept_a appears in concept_b's context window
                if concept_a.lower() in context_b['context'].lower():
                    # Detect teaching signals
                    signals = self.detect_teaching_signals(context_b['context'])
                    
                    candidate_edges.append({
                        'prerequisite': concept_a,
                        'target': concept_b,
                        'prerequisite_time': context_a['timestamp'],
                        'target_time': context_b['timestamp'],
                        'time_gap': context_b['timestamp'] - context_a['timestamp'],
                        'teaching_signals': signals,
                        'signal_count': len(signals),
                        'strength': 'strong' if len(signals) > 0 else 'weak',
                        'context_window': context_b['context'][:500]  # First 500 chars for verification
                    })
        
        return candidate_edges
    
    def process_video(self, video_id: str) -> Dict:
        """Process a single video to detect prerequisite candidates"""
        
        # Load TRANSLATED transcript (English)
        transcript_file_en = self.raw_dir / f"{video_id}_transcript_en.json"
        transcript_file_original = self.raw_dir / f"{video_id}_transcript.json"
        
        # Try translated version first, fallback to original
        if transcript_file_en.exists():
            print(f"  Using translated transcript: {transcript_file_en.name}")
            with open(transcript_file_en, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
        elif transcript_file_original.exists():
            print(f"  ⚠ Warning: Using original transcript (not translated)")
            print(f"  Run Step 1.5 (translation) first for better results!")
            with open(transcript_file_original, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
        else:
            raise FileNotFoundError(f"No transcript found for {video_id}")
        
        transcript = transcript_data['transcript']
        
        # Load concepts
        concepts_file = self.concepts_dir / f"{video_id}_concepts.json"
        with open(concepts_file, 'r', encoding='utf-8') as f:
            concepts_data = json.load(f)
        
        concepts = concepts_data['concepts']
        
        print(f"\n{'='*60}")
        print(f"Processing: {video_id}")
        print(f"Concepts: {len(concepts)}")
        print(f"Transcript chunks: {len(transcript)}")
        print(f"{'='*60}")
        
        # Find candidate edges
        print(f"Finding candidate prerequisite edges...")
        candidate_edges = self.find_candidate_edges(concepts, transcript)
        
        print(f"Found {len(candidate_edges)} candidate edges")
        
        # Count by strength
        strong_edges = [e for e in candidate_edges if e['strength'] == 'strong']
        weak_edges = [e for e in candidate_edges if e['strength'] == 'weak']
        
        print(f"  - Strong (with teaching signals): {len(strong_edges)}")
        print(f"  - Weak (no teaching signals): {len(weak_edges)}")
        
        result = {
            'video_id': video_id,
            'total_concepts': len(concepts),
            'candidate_edges': candidate_edges,
            'total_candidates': len(candidate_edges),
            'strong_candidates': len(strong_edges),
            'weak_candidates': len(weak_edges),
            'window_seconds': self.window_seconds
        }
        
        return result
    
    def save_candidates(self, result: Dict, video_id: str):
        """Save candidate edges for LLM verification"""
        output_file = self.output_dir / f"{video_id}_prerequisite_candidates.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved candidates to {output_file}")
    
    def process_all_videos(self):
        """Process all videos"""
        concept_files = list(self.concepts_dir.glob("*_concepts.json"))
        
        for concept_file in concept_files:
            video_id = concept_file.stem.replace('_concepts', '')
            
            try:
                result = self.process_video(video_id)
                self.save_candidates(result, video_id)
            except Exception as e:
                print(f"✗ Error processing {video_id}: {e}")


def main():
    detector = WindowBasedPrerequisiteDetector(window_seconds=30)
    detector.process_all_videos()


if __name__ == "__main__":
    main()
