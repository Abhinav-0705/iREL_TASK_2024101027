"""
Step 5: Prerequisite Detection - Detect dependency statements in sentences
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class PrerequisiteDetector:
    """Detect prerequisite relationships from sentences"""
    
    def __init__(
        self,
        input_dir: str = "data/processed",
        output_dir: str = "data/processed",
        config_dir: str = "config"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config_dir = Path(config_dir)
        
        # Load dependency indicators
        self.indicators = self._load_dependency_indicators()
        
    def _load_dependency_indicators(self) -> Dict:
        """Load dependency indicators from config"""
        import yaml
        
        mapping_file = self.config_dir / "linguistic_mappings.yaml"
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings = yaml.safe_load(f)
        
        return mappings.get('dependency_indicators', {})
    
    def detect_dependency_patterns(self, sentence: str) -> List[Dict]:
        """
        Detect dependency patterns in a sentence
        
        Returns:
            List of detected patterns with metadata
        """
        sentence_lower = sentence.lower()
        detected = []
        
        # Check English indicators
        for indicator in self.indicators.get('english', []):
            if indicator in sentence_lower:
                detected.append({
                    "indicator": indicator,
                    "language": "english",
                    "position": sentence_lower.find(indicator)
                })
        
        # Check Hindi indicators
        for indicator in self.indicators.get('hindi', []):
            if indicator in sentence_lower:
                detected.append({
                    "indicator": indicator,
                    "language": "hindi",
                    "position": sentence_lower.find(indicator)
                })
        
        # Check Hinglish indicators
        for indicator in self.indicators.get('hinglish', []):
            if indicator in sentence_lower:
                detected.append({
                    "indicator": indicator,
                    "language": "hinglish",
                    "position": sentence_lower.find(indicator)
                })
        
        return detected
    
    def extract_concepts_from_dependency(
        self, 
        sentence: str,
        all_concepts: List[str],
        indicator: str
    ) -> Tuple[List[str], List[str]]:
        """
        Extract tail (prerequisite) and head (target) concepts
        
        Args:
            sentence: The sentence with dependency
            all_concepts: List of all normalized concepts
            indicator: The dependency indicator phrase
            
        Returns:
            Tuple of (tail_concepts, head_concepts)
        """
        sentence_lower = sentence.lower()
        concepts_lower = [c.lower() for c in all_concepts]
        
        # Split sentence at indicator
        parts = sentence_lower.split(indicator)
        
        if len(parts) < 2:
            return [], []
        
        # Before indicator = tail (prerequisites)
        # After indicator = head (target concept)
        before_text = parts[0]
        after_text = parts[1]
        
        tail_concepts = []
        head_concepts = []
        
        # Find concepts in each part
        for i, concept in enumerate(concepts_lower):
            original_concept = all_concepts[i]
            
            if concept in before_text:
                tail_concepts.append(original_concept)
            if concept in after_text:
                head_concepts.append(original_concept)
        
        return tail_concepts, head_concepts
    
    def detect_conjunction_type(self, text: str) -> str:
        """
        Detect if multiple prerequisites are connected with AND or OR
        
        Returns:
            "and", "or", or "single"
        """
        text_lower = text.lower()
        
        # AND indicators
        and_patterns = [
            r'\band\b', r'\baur\b', r'\bऔर\b',
            r'\bdono\b', r'\bदोनों\b',
            r'\btatha\b', r'\bतथा\b',
            r',.*,',  # Multiple commas suggest AND
        ]
        
        # OR indicators
        or_patterns = [
            r'\bor\b', r'\bya\b', r'\bया\b',
            r'\beither\b', r'\bathwa\b',
        ]
        
        for pattern in and_patterns:
            if re.search(pattern, text_lower):
                return "and"
        
        for pattern in or_patterns:
            if re.search(pattern, text_lower):
                return "or"
        
        return "single"
    
    def classify_dependency_strength(
        self, 
        indicator: str,
        sentence: str
    ) -> str:
        """
        Classify if dependency is hard (required) or soft (helpful)
        
        Returns:
            "hard" or "soft"
        """
        sentence_lower = sentence.lower()
        
        # Hard dependency indicators
        hard_keywords = [
            "must", "required", "zaroori", "jaruri", "chahiye",
            "necessary", "cannot", "without", "bina",
            "pehle", "first", "before"
        ]
        
        # Soft dependency indicators
        soft_keywords = [
            "helpful", "better", "recommended", "should",
            "accha", "useful", "prefer"
        ]
        
        for keyword in hard_keywords:
            if keyword in sentence_lower:
                return "hard"
        
        for keyword in soft_keywords:
            if keyword in sentence_lower:
                return "soft"
        
        # Default: hard for strong indicators, soft for weak
        strong_indicators = ["requires", "zaroori hai", "pehle", "depends on"]
        
        for strong in strong_indicators:
            if strong in indicator:
                return "hard"
        
        return "soft"
    
    def analyze_sentence(
        self, 
        sentence: Dict,
        all_concepts: List[str]
    ) -> Optional[Dict]:
        """
        Analyze a sentence for prerequisite relationships
        
        Args:
            sentence: Sentence dictionary with text and concepts
            all_concepts: List of all normalized concepts
            
        Returns:
            Prerequisite relationship dict or None
        """
        text = sentence['text']
        
        # Detect dependency patterns
        patterns = self.detect_dependency_patterns(text)
        
        if not patterns:
            return None
            
        # Check if the sentence has mapped concepts from Step 4
        sentence_concepts = sentence.get('normalized_concepts', [])
        if len(sentence_concepts) < 2:
            return None
        
        # Use the first (or strongest) pattern
        pattern = patterns[0]
        indicator = pattern['indicator']
        
        # We know there's a dependency indicator and at least 2 concepts.
        # Split sentence at indicator to figure out which concept is head vs tail
        sentence_lower = text.lower()
        parts = sentence_lower.split(indicator)
        
        if len(parts) < 2:
            return None
            
        before_text = parts[0]
        after_text = parts[1]
        
        tail_concepts = []
        head_concepts = []
        
        for concept in sentence_concepts:
            concept_lower = concept.lower()
            if concept_lower in before_text:
                tail_concepts.append(concept)
            elif concept_lower in after_text:
                head_concepts.append(concept)
        
        # If we couldn't neatly split them across the indicator, fall back 
        # to assigning all but the last to tail, and the last to head
        if not tail_concepts or not head_concepts:
             tail_concepts = sentence_concepts[:-1]
             head_concepts = [sentence_concepts[-1]]
        
        if not tail_concepts or not head_concepts:
            return None
        
        # Determine conjunction type
        tail_text = text.lower().split(indicator)[0]
        conjunction = self.detect_conjunction_type(tail_text)
        
        # Determine dependency strength
        strength = self.classify_dependency_strength(indicator, text)
        
        return {
            "sentence_id": sentence['id'],
            "sentence_text": text,
            "tail": tail_concepts,
            "head": head_concepts[0] if head_concepts else None,
            "conjunction_type": conjunction,
            "strength": strength,
            "indicator": indicator,
            "language": pattern['language'],
            "confidence": self._calculate_confidence(tail_concepts, head_concepts, indicator)
        }
    
    def _calculate_confidence(
        self,
        tail_concepts: List[str],
        head_concepts: List[str],
        indicator: str
    ) -> float:
        """Calculate confidence score for the detected prerequisite"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we found concepts on both sides
        if tail_concepts and head_concepts:
            confidence += 0.2
        
        # Increase for strong indicators
        strong_indicators = ["requires", "zaroori hai", "pehle", "depends on", "necessary"]
        if any(strong in indicator for strong in strong_indicators):
            confidence += 0.2
        
        # Decrease if too many concepts (might be noisy)
        if len(tail_concepts) > 3:
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def detect_prerequisites(
        self,
        normalized_file: str,
        video_id: str
    ) -> Dict:
        """
        Detect prerequisites from normalized concepts
        
        Args:
            normalized_file: Path to normalized concepts JSON
            video_id: Video identifier
            
        Returns:
            Dictionary with detected prerequisites
        """
        # Load normalized data
        with open(normalized_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get all normalized concepts
        all_concepts = [c['normalized'] for c in data['concepts']]
        
        # Get sentences
        sentences = data.get('sentences_with_concepts', [])
        
        print(f"Analyzing {len(sentences)} sentences for prerequisites...")
        
        # Analyze each sentence
        prerequisites = []
        
        for sentence in sentences:
            prereq = self.analyze_sentence(sentence, all_concepts)
            
            if prereq:
                prerequisites.append(prereq)
        
        return {
            "video_id": video_id,
            "domain": data.get('domain'),
            "total_prerequisites": len(prerequisites),
            "prerequisites": prerequisites,
            "concepts": all_concepts
        }
    
    def save_prerequisites(self, prerequisites: Dict, video_id: str):
        """Save detected prerequisites"""
        output_file = self.output_dir / f"{video_id}_prerequisites.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prerequisites, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Detected {prerequisites['total_prerequisites']} prerequisite relationships")
        print(f"✓ Saved to {output_file}")
    
    def process_all_normalized(self):
        """Process all normalized files"""
        normalized_files = list(self.input_dir.glob("*_normalized.json"))
        
        if not normalized_files:
            print("No normalized files found!")
            return
        
        for normalized_file in normalized_files:
            video_id = normalized_file.stem.replace("_normalized", "")
            
            print(f"\n{'='*60}")
            print(f"Detecting prerequisites: {video_id}")
            print(f"{'='*60}")
            
            try:
                prerequisites = self.detect_prerequisites(normalized_file, video_id)
                self.save_prerequisites(prerequisites, video_id)
                
            except Exception as e:
                print(f"✗ Error detecting prerequisites for {video_id}: {e}")


if __name__ == "__main__":
    detector = PrerequisiteDetector()
    detector.process_all_normalized()
