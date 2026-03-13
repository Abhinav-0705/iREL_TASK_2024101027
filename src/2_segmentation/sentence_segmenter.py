"""
Step 2: Sentence Segmentation - Break transcript into teaching units
"""
import json
import re
from pathlib import Path
from typing import List, Dict
import spacy
from spacy.lang.hi import Hindi
from spacy.lang.en import English


class SentenceSegmenter:
    """Segment transcripts into teaching units (sentences)"""
    
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load spaCy models for Hindi and English
        try:
            self.nlp_hi = spacy.load("xx_ent_wiki_sm")  # Multilingual model
        except:
            print("Multilingual model not found, using basic Hindi")
            self.nlp_hi = Hindi()
            self.nlp_hi.add_pipe("sentencizer")
        
        self.nlp_en = English()
        self.nlp_en.add_pipe("sentencizer")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common issues
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        
        return text.strip()
    
    def segment_by_punctuation(self, text: str) -> List[str]:
        """
        Simple punctuation-based segmentation
        Useful for code-mixed text where spaCy might struggle
        """
        # Split on sentence boundaries
        sentences = re.split(r'[.!?।]+', text)
        
        # Clean and filter
        sentences = [self.clean_text(s) for s in sentences if s.strip()]
        
        return sentences
    
    def segment_with_spacy(self, text: str) -> List[str]:
        """Segment using spaCy (works better for proper sentences)"""
        doc = self.nlp_hi(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences
    
    def hybrid_segmentation(self, text: str, min_length: int = 5, max_length: int = 200) -> List[Dict]:
        """
        Hybrid segmentation combining multiple methods
        
        Args:
            text: Input text
            min_length: Minimum sentence length in characters
            max_length: Maximum sentence length in characters
            
        Returns:
            List of sentence dictionaries with metadata
        """
        # Try spaCy first
        try:
            sentences = self.segment_with_spacy(text)
        except:
            # Fallback to punctuation-based
            sentences = self.segment_by_punctuation(text)
        
        # Filter and create structured output
        segmented = []
        for idx, sentence in enumerate(sentences, 1):
            sentence = self.clean_text(sentence)
            
            # Apply length filters
            if len(sentence) < min_length:
                continue
            
            # Split long sentences
            if len(sentence) > max_length:
                # Try to split on conjunctions
                sub_sentences = re.split(r'\s+(aur|और|and|toh|तो|then|but|par|पर)\s+', sentence)
                for sub in sub_sentences:
                    if len(sub) >= min_length:
                        segmented.append({
                            "id": f"sent_{len(segmented) + 1}",
                            "text": sub.strip(),
                            "length": len(sub),
                            "position": len(segmented) + 1
                        })
            else:
                segmented.append({
                    "id": f"sent_{idx}",
                    "text": sentence,
                    "length": len(sentence),
                    "position": idx
                })
        
        return segmented
    
    def segment_transcript(self, transcript_file: str, video_id: str) -> Dict:
        """
        Segment a transcript file
        
        Args:
            transcript_file: Path to transcript JSON file
            video_id: Video identifier
            
        Returns:
            Dictionary with segmented sentences
        """
        # Load transcript
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        
        # Get full text
        full_text = transcript.get('full_text', '')
        
        if not full_text:
            raise ValueError(f"No full text found in transcript for {video_id}")
        
        # Segment
        sentences = self.hybrid_segmentation(full_text)
        
        return {
            "video_id": video_id,
            "total_sentences": len(sentences),
            "sentences": sentences,
            "metadata": {
                "source": transcript.get('source', 'unknown'),
                "original_text_length": len(full_text)
            }
        }
    
    def save_segmented(self, segmented: Dict, video_id: str):
        """Save segmented transcript"""
        output_file = self.output_dir / f"{video_id}_segmented.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(segmented, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved {segmented['total_sentences']} sentences to {output_file}")
    
    def process_all_transcripts(self):
        """Process all transcript files in input directory"""
        transcript_files = list(self.input_dir.glob("*_transcript.json"))
        
        if not transcript_files:
            print("No transcript files found!")
            return
        
        for transcript_file in transcript_files:
            video_id = transcript_file.stem.replace("_transcript", "")
            
            print(f"\nSegmenting: {video_id}")
            
            try:
                segmented = self.segment_transcript(transcript_file, video_id)
                self.save_segmented(segmented, video_id)
                
            except Exception as e:
                print(f"✗ Error segmenting {video_id}: {e}")


if __name__ == "__main__":
    segmenter = SentenceSegmenter()
    segmenter.process_all_transcripts()
