"""
Step 3: Concept Extraction - Extract technical concepts from sentences
Research-quality pipeline with proper text cleaning and noun phrase extraction
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter
import spacy
from keybert import KeyBERT
import string


class ConceptExtractor:
    """Extract technical concepts using cleaned noun phrases + domain vocabulary + KeyBERT"""
    
    def __init__(self, input_dir: str = "data/processed", output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize KeyBERT (for semantic keyphrase extraction)
        self.keybert = KeyBERT(model='paraphrase-multilingual-mpnet-base-v2')
        
        # Load spaCy for noun phrase extraction (need English model for noun_chunks)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Warning: English spaCy model not found. Attempting to download...")
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Error: Could not load spaCy model")
                self.nlp = None
        
        # Domain-specific vocabulary (controlled vocabulary)
        self.domain_vocab = self._load_domain_vocabulary()
        
        # Hindi/Telugu transliteration mappings (Devanagari → English concepts)
        self.hindi_concept_map = self._load_hindi_mappings()
        
        # Transcript filler words (to remove during cleaning)
        self.filler_words = {
            "uh", "um", "ah", "er", "hmm", "like", "you know", "i mean",
            "basically", "actually", "literally", "kind of", "sort of",
            "gonna", "wanna", "gotta", "kinda", "sorta",
            "yeah", "yes", "okay", "ok", "right", "well", "so",
            "guys", "please", "now", "then", "here", "there"
        }
        
        # Meta words to remove (video-specific, not concepts)
        self.meta_words = {
            "video", "lecture", "playlist", "series", "channel", "tutorial",
            "episode", "part", "session", "course", "class", "lesson",
            "problem", "question", "example", "thing", "way", "time",
            "value", "number", "step", "something", "anything", "everything"
        }
        
        # Pronouns and determiners to filter
        self.pronouns_and_determiners = {
            "i", "you", "he", "she", "it", "we", "they",
            "me", "him", "her", "us", "them",
            "my", "your", "his", "her", "its", "our", "their",
            "mine", "yours", "hers", "ours", "theirs",
            "this", "that", "these", "those",
            "what", "which", "who", "whom", "whose",
            "the", "a", "an", "some", "any", "many", "much",
            "all", "both", "each", "every", "either", "neither"
        }
        
        # Combined stoplist
        self.stoplist = self.filler_words | self.meta_words | self.pronouns_and_determiners
    
    def is_valid_concept(self, concept: str) -> bool:
        """Check if concept is valid (not numbers, not pure Hindi, not junk)"""
        concept_lower = concept.lower()
        
        # Reject if starts with a number
        if concept and concept[0].isdigit():
            return False
        
        # Reject if contains numbers mixed with short text (e.g., "217 आ गया")
        if any(c.isdigit() for c in concept) and len(concept.split()) <= 3:
            return False
        
        # Reject if in stoplist
        if concept_lower in self.stoplist:
            return False
        
        # Reject if it's pure Devanagari/Telugu script (no technical value)
        ascii_chars = sum(1 for c in concept if ord(c) < 128)
        if len(concept) > 0:
            ascii_ratio = ascii_chars / len(concept)
            # If <10% ASCII and not in our Hindi mappings, it's probably junk
            if ascii_ratio < 0.1 and concept not in self.hindi_concept_map:
                return False
        
        # Must be at least 3 characters
        if len(concept) < 3:
            return False
        
        return True
        """Check if text contains technical terms (English or Romanized)"""
        # Remove spaces and punctuation
        text_clean = text.lower().replace(' ', '')
        
        # Check if it matches known technical patterns or domain vocabulary
        for domain_concepts in self.domain_vocab.values():
            for concept in domain_concepts:
                if concept.replace(' ', '') in text_clean:
                    return True
        
        # Check if it's primarily ASCII (English/Romanized technical terms)
        ascii_chars = sum(1 for c in text_clean if ord(c) < 128)
        if len(text_clean) > 0 and (ascii_chars / len(text_clean)) > 0.5:
            return True
        
        return False
    
    def is_english_text(self, text: str) -> bool:
        """Check if text is primarily English/Roman script (not pure Devanagari)"""
        # Remove punctuation and spaces
        text_clean = ''.join(c for c in text if c not in string.punctuation and c != ' ')
        
        if not text_clean:
            return False
        
        # Count ASCII (English/Roman) vs non-ASCII (Indic scripts) characters
        ascii_count = sum(1 for c in text_clean if ord(c) < 128)
        total_count = len(text_clean)
        
        # Consider it English/Roman if >40% ASCII characters (lowered threshold for code-mixed)
        return (ascii_count / total_count) > 0.4
    
    def _load_domain_vocabulary(self) -> Dict[str, Set[str]]:
        """Load controlled domain vocabulary"""
        return {
            "cs": {
                "dynamic programming", "recursion", "memoization", "tabulation",
                "recursion tree", "dp array", "overlapping subproblems",
                "space optimization", "time complexity", "space complexity",
                "fibonacci", "knapsack", "longest common subsequence",
                "matrix chain multiplication", "edit distance",
                "stack", "queue", "array", "linked list", "tree", "graph",
                "iteration", "base case", "recursive case", "optimal substructure"
            },
            "ml": {
                "neural network", "backpropagation", "gradient descent",
                "loss function", "chain rule", "partial derivative",
                "activation function", "weight", "bias", "learning rate",
                "forward propagation", "backward propagation",
                "sigmoid", "relu", "tanh", "softmax",
                "gradient", "optimizer", "epoch", "batch"
            },
            "algorithms": {
                "graph", "tree", "dfs", "bfs", "depth first search",
                "breadth first search", "stack", "queue",
                "adjacency list", "adjacency matrix", "traversal",
                "node", "edge", "vertex", "path", "cycle",
                "connected component", "spanning tree"
            },
            "physics": {
                "charge", "electric field", "coulomb law", "coulomb's law",
                "force", "vector", "potential", "electrostatics",
                "field lines", "electric potential", "voltage",
                "vector addition", "scalar", "magnitude", "direction"
            },
            "chemistry": {
                "ph", "poh", "hydrogen ion", "concentration", "logarithm",
                "acid", "base", "equilibrium", "ionization",
                "kw", "ka", "kb", "pka", "pkb", "buffer",
                "strong acid", "weak acid", "dissociation"
            }
        }
    
    def _load_hindi_mappings(self) -> Dict[str, str]:
        """Map Hindi/Telugu technical terms to English"""
        return {
            # Hindi Physics terms
            "चार्ज": "charge",
            "चार्जेस": "charges",
            "पॉइंट चार्जेस": "point charges",
            "इलेक्ट्रिकल चार्जेस": "electrical charges",
            "कलम्स लॉ": "coulomb law",
            "कूलाम का नियम": "coulomb law",
            "फोर्स": "force",
            "फोर्सेस": "forces",
            "वेक्टर": "vector",
            "इलेक्ट्रिक फील्ड": "electric field",
            "विद्युत क्षेत्र": "electric field",
            
            # Hindi CS/Algorithms terms
            "ग्राफ": "graph",
            "ट्री": "tree",
            "डेप्थ फर्स्ट सर्च": "depth first search",
            "ब्रेडथ फर्स्ट सर्च": "breadth first search",
            "डीएफएस": "dfs",
            "बीएफएस": "bfs",
            "रिकर्शन": "recursion",
            "डायनामिक प्रोग्रामिंग": "dynamic programming",
            "मेमोराइजेशन": "memoization",
            "स्टैक": "stack",
            "क्यू": "queue",
            "नोड": "node",
            "एज": "edge",
            "वर्टेक्स": "vertex",
            "एल्गोरिथम": "algorithm",
            "ट्रैवर्सल": "traversal",
            "पाथ": "path",
            "साइकिल": "cycle",
            
            # Hindi ML terms  
            "न्यूरल नेटवर्क": "neural network",
            "बैकप्रोपगेशन": "backpropagation",
            "ग्रेडिएंट डिसेंट": "gradient descent",
            "लॉस फंक्शन": "loss function",
            "वेट": "weight",
            "बायस": "bias",
            
            # Hindi Chemistry terms
            "एसिड": "acid",
            "बेस": "base",
            "पीएच": "ph",
            "कंसंट्रेशन": "concentration",
            
            # Telugu/Common transliterations
            "రికర్షన్": "recursion",
            "డైనమిక్": "dynamic",
            "ప్రోగ్రామింగ్": "programming",
        }
    
    def extract_hindi_concepts(self, text: str) -> List[Tuple[str, str]]:
        """Extract Hindi technical terms and convert to English"""
        found_concepts = []
        
        for hindi_term, english_term in self.hindi_concept_map.items():
            if hindi_term in text:
                found_concepts.append((english_term, "hindi_mapping"))
        
        return found_concepts
    
    def clean_text(self, text: str) -> str:
        """Clean transcript text by removing fillers and normalizing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove filler words (with word boundaries)
        for filler in self.filler_words:
            text = re.sub(r'\b' + re.escape(filler) + r'\b', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases using spaCy noun chunks - ENGLISH ONLY for code-mixed text"""
        if self.nlp is None:
            return []
        
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        doc = self.nlp(cleaned_text)
        noun_phrases = []
        
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            
            # **FILTER: Only keep English phrases (skip Hindi/Telugu)**
            if not self.is_english_text(phrase):
                continue
            
            # Skip single-word pronouns/determiners
            if phrase in self.stoplist:
                continue
            
            # Skip if starts with stoplist word (e.g., "the value", "this problem")
            words = phrase.split()
            if words and words[0] in self.stoplist:
                # Remove the leading stoplist word
                phrase = ' '.join(words[1:])
            
            # Skip if phrase is too short after filtering
            if len(phrase) < 3:
                continue
            
            # Skip if all words are in stoplist
            remaining_words = [w for w in phrase.split() if w not in self.stoplist]
            if not remaining_words:
                continue
            
            # Keep only if it has at least one meaningful word
            noun_phrases.append(phrase)
        
        return noun_phrases
    
    def extract_domain_concepts(self, text: str) -> List[Tuple[str, str]]:
        """Extract concepts from domain vocabulary with exact matching"""
        text_lower = self.clean_text(text)
        found_concepts = []
        
        for domain, concepts in self.domain_vocab.items():
            for concept in concepts:
                # Use word boundary matching for accuracy
                pattern = r'\b' + re.escape(concept) + r'\b'
                if re.search(pattern, text_lower):
                    found_concepts.append((concept, domain))
        
        return found_concepts
    
    def extract_with_keybert(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract keyphrases using KeyBERT on cleaned text - ENGLISH ONLY"""
        try:
            # Clean text first
            cleaned_text = self.clean_text(text)
            
            keywords = self.keybert.extract_keywords(
                cleaned_text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=top_n * 3,  # Extract more, then filter
                diversity=0.7,
                use_mmr=True,
                nr_candidates=50
            )
            
            # Filter and clean results
            filtered = []
            for phrase, score in keywords:
                phrase_lower = phrase.lower()
                
                # **Validate concept first**
                if not self.is_valid_concept(phrase):
                    continue
                
                # **FILTER: Only keep English phrases**
                if not self.is_english_text(phrase):
                    continue
                
                # Skip if in stoplist
                if phrase_lower in self.stoplist:
                    continue
                
                # Skip if all words are in stoplist
                words = phrase_lower.split()
                if all(w in self.stoplist for w in words):
                    continue
                
                # Require minimum score
                if score > 0.2:  # Lowered threshold since we're filtering heavily
                    filtered.append((phrase_lower, score))
            
            return filtered[:top_n]
            
        except Exception as e:
            print(f"KeyBERT extraction failed: {e}")
            return []
    
    def rank_concepts(self, 
                     noun_phrases: List[str],
                     domain_concepts: List[Tuple[str, str]],
                     keybert_concepts: List[Tuple[str, float]],
                     top_n: int = 15) -> List[Dict]:
        """
        Rank concepts using:
        1. Frequency (from noun phrases)
        2. Domain match (bonus for domain vocabulary)
        3. KeyBERT score (semantic relevance)
        4. Phrase length (prefer 2-3 word phrases)
        """
        concept_data = {}
        
        # Count noun phrase frequency
        noun_phrase_counts = Counter(noun_phrases)
        
        # Process all unique concepts
        all_concepts = set(noun_phrases) | {c[0] for c in domain_concepts} | {c[0] for c in keybert_concepts}
        
        for concept in all_concepts:
            # Validate concept first
            if not self.is_valid_concept(concept):
                continue
            
            if concept in self.stoplist:
                continue
                
            # Calculate scores
            frequency = noun_phrase_counts.get(concept, 0)
            domain_match = 1 if any(concept == dc[0] for dc in domain_concepts) else 0
            keybert_score = next((score for phrase, score in keybert_concepts if phrase == concept), 0)
            phrase_length = len(concept.split())
            
            # Length preference: 2-3 words get bonus, 1 word and 4+ words get penalty
            length_bonus = 1.5 if 2 <= phrase_length <= 3 else (0.7 if phrase_length == 1 else 0.5)
            
            # Combined score with weights
            combined_score = (
                frequency * 1.0 +            # Frequency weight (reduced)
                domain_match * 50.0 +        # Domain match weight (HEAVILY BOOSTED)
                keybert_score * 5.0 +        # KeyBERT semantic score (increased)
                length_bonus * 2.0           # Phrase length bonus (increased)
            )
            
            concept_data[concept] = {
                'concept': concept.title(),
                'frequency': frequency,
                'domain_match': domain_match,
                'keybert_score': round(keybert_score, 3),
                'phrase_length': phrase_length,
                'combined_score': round(combined_score, 3)
            }
        
        # Sort by combined score
        ranked = sorted(concept_data.values(), key=lambda x: x['combined_score'], reverse=True)
        
        return ranked[:top_n]
    
    def extract_concepts_from_video(self, segmented_file: str, video_id: str, domain: str = None) -> Dict:
        """
        Extract concepts from a segmented video file
        
        Args:
            segmented_file: Path to segmented JSON file
            video_id: Video identifier
            domain: Video domain (optional)
            
        Returns:
            Dictionary with extracted concepts
        """
        # Load segmented data
        with open(segmented_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sentences = [s['text'] for s in data['sentences']]
        full_text = " ".join(sentences)
        
        print(f"Extracting concepts from {len(sentences)} sentences...")
        
        # Detect if this is a code-mixed video (>50% non-ASCII characters)
        full_text_clean = ''.join(c for c in full_text if c not in string.punctuation)
        ascii_ratio = sum(1 for c in full_text_clean if ord(c) < 128) / max(len(full_text_clean), 1)
        is_code_mixed = ascii_ratio < 0.5
        
        if is_code_mixed:
            print(f"  ⚠️  Detected code-mixed content (Hindi/Telugu dominant)")
            print(f"     Using domain vocabulary + Hindi mappings only")
        
        all_noun_phrases = []
        
        # STEP 1: Extract noun phrases ONLY if primarily English
        if not is_code_mixed:
            print("  - Extracting noun phrases...")
            for sentence in sentences:
                noun_phrases = self.extract_noun_phrases(sentence)
                all_noun_phrases.extend(noun_phrases)
            print(f"    Found {len(all_noun_phrases)} noun phrases")
        else:
            print("  - Skipping noun phrase extraction (code-mixed text)")
        
        # STEP 2: Match against domain vocabulary (works for all languages)
        print("  - Matching domain vocabulary...")
        domain_concepts = self.extract_domain_concepts(full_text)
        print(f"    Found {len(domain_concepts)} domain matches")
        
        # STEP 3: Extract Hindi/Telugu concepts and convert to English
        print("  - Extracting Hindi/Telugu technical terms...")
        hindi_concepts = self.extract_hindi_concepts(full_text)
        # Convert to domain concept format
        hindi_as_domain = [(concept, "hindi") for concept, _ in hindi_concepts]
        domain_concepts.extend(hindi_as_domain)
        print(f"    Found {len(hindi_concepts)} Hindi/Telugu technical terms")
        
        # STEP 4: Extract keyphrases with KeyBERT (on full text, it will find English terms)
        print("  - Running KeyBERT extraction...")
        keybert_concepts = self.extract_with_keybert(full_text, top_n=20)
        print(f"    Found {len(keybert_concepts)} KeyBERT concepts")
        
        # STEP 5: Rank and combine
        print("  - Running KeyBERT extraction...")
        keybert_concepts = self.extract_with_keybert(full_text, top_n=15)
        print(f"    Found {len(keybert_concepts)} KeyBERT concepts")
        
        # STEP 4: Rank and combine
        print("  - Ranking concepts...")
        ranked_concepts = self.rank_concepts(
            all_noun_phrases,
            domain_concepts,
            keybert_concepts,
            top_n=15
        )
        
        return {
            "video_id": video_id,
            "domain": domain,
            "total_concepts": len(ranked_concepts),
            "concepts": ranked_concepts,
            "sentences_with_concepts": self._map_concepts_to_sentences(
                data['sentences'],
                ranked_concepts
            )
        }
    
    def _map_concepts_to_sentences(self, sentences: List[Dict], concepts: List[Dict]) -> List[Dict]:
        """Map which concepts appear in which sentences"""
        concept_names = [c['concept'].lower() for c in concepts]
        
        for sentence in sentences:
            text_lower = sentence['text'].lower()
            sentence['concepts'] = [
                c for c in concept_names
                if c in text_lower
            ]
        
        return sentences
    
    def save_concepts(self, concepts: Dict, video_id: str):
        """Save extracted concepts"""
        output_file = self.output_dir / f"{video_id}_concepts.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(concepts, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved {concepts['total_concepts']} concepts to {output_file}")
    
    def process_all_segmented(self, videos_config: List[Dict] = None):
        """Process all segmented files"""
        segmented_files = list(self.input_dir.glob("*_segmented.json"))
        
        if not segmented_files:
            print("No segmented files found!")
            return
        
        # Create domain mapping if config provided
        domain_map = {}
        if videos_config:
            domain_map = {v['id']: v['domain'] for v in videos_config}
        
        for segmented_file in segmented_files:
            video_id = segmented_file.stem.replace("_segmented", "")
            domain = domain_map.get(video_id)
            
            print(f"\n{'='*60}")
            print(f"Extracting concepts: {video_id}")
            if domain:
                print(f"Domain: {domain}")
            print(f"{'='*60}")
            
            try:
                concepts = self.extract_concepts_from_video(segmented_file, video_id, domain)
                self.save_concepts(concepts, video_id)
                
            except Exception as e:
                print(f"✗ Error extracting concepts from {video_id}: {e}")


if __name__ == "__main__":
    from src.utils.config_loader import ConfigLoader
    
    config = ConfigLoader()
    videos_config = config.load_videos()
    
    extractor = ConceptExtractor()
    extractor.process_all_segmented(videos_config['videos'])
