"""
Step 3 (LLM): Concept Extraction using Iterative LLM Refinement (Pipeline 3)
2-Pass approach using Groq (Llama 3.3 70B) as the primary LLM:
  Pass 1 (Raw Extract)      → Groq / Llama 3.3 70B
  Pass 2 (Refine & Score)   → Groq / Llama 3.3 70B
  
Falls back to Gemini if Groq is unavailable.
"""
import json
import re
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMConceptExtractor:
    """Extract technical concepts using a 2-pass LLM pipeline.
    
    Primary: Groq (Llama 3.3 70B) — fast, free
    Fallback: Google Gemini (if Groq unavailable)
    """

    def __init__(
        self,
        input_dir: str = "data/processed",
        output_dir: str = "data/processed",
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_concepts: int = 20,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        groq_model: str = "llama-3.3-70b-versatile",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.temperature = temperature
        self.max_concepts = max_concepts
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.groq_model = groq_model

        # Initialize LLM clients
        self.groq_client = None
        self.gemini_model = None
        self._init_groq()
        # Skip Gemini initialization - using Groq only
        # self._init_gemini()

        # Ensure Groq is available
        if not self.groq_client:
            raise ValueError(
                "GROQ_API_KEY not found!\n"
                "Please set GROQ_API_KEY in environment:\n"
                "  export GROQ_API_KEY='your_key'\n"
                "Get a free API key at: https://console.groq.com/keys"
            )

    def _init_groq(self):
        """Initialize the Groq client (PRIMARY)"""
        try:
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key or api_key == "your_groq_api_key_here":
                print("ℹ  No GROQ_API_KEY found — will try Gemini")
                return

            self.groq_client = Groq(api_key=api_key)
            print(f"✓ Groq client initialized (PRIMARY): {self.groq_model}")

        except ImportError:
            print("ℹ  groq package not installed — will try Gemini")

    def _init_gemini(self):
        """Initialize the Google Gemini client (FALLBACK)"""
        try:
            import google.generativeai as genai

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("ℹ  No GEMINI_API_KEY found")
                return

            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(
                self.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                ),
            )
            label = "FALLBACK" if self.groq_client else "PRIMARY"
            print(f"✓ Gemini model initialized ({label}): {self.model_name}")

        except ImportError:
            print("ℹ  google-generativeai not installed")
        except Exception as e:
            print(f"⚠  Gemini init error: {e}")

    # ──────────────────────────────────────────────
    #  Unified LLM Call with Retry
    # ──────────────────────────────────────────────

    def _call_llm(self, prompt: str, system_prompt: str = None, max_retries: int = 3) -> str:
        """Call Groq LLM with retry logic."""

        # Use Groq only
        if not self.groq_client:
            raise ValueError("Groq client not initialized!")

        for attempt in range(max_retries):
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                if "invalid_api_key" in error_str.lower() or "401" in error_str:
                    raise ValueError(f"Groq API key invalid: {error_str}")
                if "429" in error_str or "rate" in error_str.lower():
                    wait_time = (attempt + 1) * 10
                    print(f"    ⏳ Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ⚠ Groq error: {error_str[:150]}")
                    if attempt == max_retries - 1:
                        raise

        raise Exception("All LLM call attempts failed")

    # ──────────────────────────────────────────────
    #  Text Chunking
    # ──────────────────────────────────────────────

    def chunk_text(self, sentences: List[str]) -> List[str]:
        """Split sentences into overlapping chunks of ~chunk_size words"""
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            word_count = len(sentence.split())
            if current_word_count + word_count > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep last few sentences for overlap
                overlap_words = 0
                overlap_start = len(current_chunk)
                for i in range(len(current_chunk) - 1, -1, -1):
                    overlap_words += len(current_chunk[i].split())
                    if overlap_words >= self.chunk_overlap:
                        overlap_start = i
                        break
                current_chunk = current_chunk[overlap_start:]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            current_chunk.append(sentence)
            current_word_count += word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # ──────────────────────────────────────────────
    #  PASS 1: Raw Extraction
    # ──────────────────────────────────────────────

    def _pass1_extract(self, chunk: str, domain: str = None) -> List[str]:
        """Pass 1: Extract all potential technical concepts from a transcript chunk"""

        domain_hint = f'The video is about "{domain}".' if domain else ""

        prompt = f"""You are an expert at extracting technical and pedagogical concepts from educational video transcripts.

The transcript may be code-mixed (Hindi + English, Hinglish, or Telugu + English). 
Extract ALL technical concepts regardless of language — convert Hindi/Hinglish/Telugu terms to their standard English equivalents.

{domain_hint}

RULES:
1. Extract ONLY technical/academic concepts (not filler words, meta-words like "video", "lecture", "example")
2. Convert code-mixed terms to English: "recursion wala approach" → "Recursion", "samajhte hain DP ko" → "Dynamic Programming"
3. Include multi-word concepts: "depth first search", "time complexity", "linked list"
4. Do NOT include generic words: "problem", "question", "value", "number", "thing"
5. Include both the concept name AND any sub-concepts or variations

Transcript chunk:
---
{chunk}
---

Return a JSON object with this exact structure:
{{"concepts": ["concept1", "concept2", ...]}}

Extract generously — later passes will filter noise."""

        try:
            response_text = self._call_llm(
                prompt,
                system_prompt="You are a technical concept extractor. Always respond with valid JSON only."
            )
            if not response_text:
                return []
            result = json.loads(response_text)
            return result.get("concepts", [])
        except (json.JSONDecodeError, Exception) as e:
            print(f"  ⚠ Pass 1 parse error: {e}")
            return []

    # ──────────────────────────────────────────────
    #  PASS 2: Refine, Deduplicate & Score
    # ──────────────────────────────────────────────

    def _pass2_refine(
        self, raw_concepts: List[str], full_text: str, domain: str = None
    ) -> List[Dict]:
        """Pass 2: Deduplicate, standardize, assign confidence scores, and categorize"""

        domain_hint = f'Domain: "{domain}".' if domain else ""

        prompt = f"""You are an expert at structuring educational concept data.

{domain_hint}

Here is a raw list of technical concepts extracted from an educational video (contains noise and duplicates):
{json.dumps(raw_concepts, indent=2)}

For context, here's a snippet of the transcript (first 1000 chars):
---
{full_text[:1000]}
---

YOUR TASK:
1. **DEDUPLICATE & MERGE**: Combine synonyms and related terms into a single canonical concept. Do NOT repeat concepts (e.g. "Charge" and "Electrical Charge" -> "Electrical Charge").
2. **CLEAN**: Remove non-technical terms, filler words ("video", "example"), and vague terms ("method", "value").
3. **STANDARDIZE**: Use the canonical English name (Title Case).
4. **SCORE & CATEGORIZE**: For each valid, unique concept provide:
   - **concept**: The standardized name
   - **confidence**: How confident you are this is a distinct technical concept taught in the video (0.0 to 1.0)
   - **category**: One of: "core" (main topic), "prerequisite" (needed to understand core), "related" (mentioned but not the focus), "definition" (a term being defined)
   - **domain**: The specific domain (e.g., "algorithms", "machine_learning", "physics", "chemistry", "mathematics")

Return EXACTLY a JSON object in this format:
{{
  "concepts": [
    {{
      "concept": "Dynamic Programming",
      "confidence": 0.95,
      "category": "core",
      "domain": "algorithms"
    }}
  ]
}}

Include at most {self.max_concepts} distinct concepts, ranked by relevance to the video content.
Exclude any concept with confidence below 0.4. Ensure absolutely ZERO logical duplicates."""

        try:
            response_text = self._call_llm(
                prompt,
                system_prompt="You are an expert technical concept reviewer and structurer. Always respond with valid JSON only."
            )
            if not response_text:
                return []
            result = json.loads(response_text)
            return result.get("concepts", [])
        except (json.JSONDecodeError, Exception) as e:
            print(f"  ⚠ Pass 2 parse error: {e}")
            return []

    # ──────────────────────────────────────────────
    #  Main Extraction Pipeline
    # ──────────────────────────────────────────────

    def extract_concepts_from_video(
        self, segmented_file, video_id: str, domain: str = None
    ) -> Dict:
        """Run the full 3-pass LLM concept extraction pipeline on a video."""
        # Load segmented data
        with open(segmented_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        sentences = [s["text"] for s in data["sentences"]]
        full_text = " ".join(sentences)

        engine = f"Groq/{self.groq_model}" if self.groq_client else f"Gemini/{self.model_name}"
        print(f"\n🤖 LLM Concept Extraction for: {video_id} (using {engine})")
        print(f"   Sentences: {len(sentences)}, Words: {len(full_text.split())}")

        # ── CHUNK the transcript ──
        chunks = self.chunk_text(sentences)
        print(f"   Chunks: {len(chunks)}")

        # ── PASS 1: Raw extraction from each chunk ──
        print(f"\n   📝 Pass 1: Raw Extraction...")
        all_raw_concepts = []
        for i, chunk in enumerate(chunks):
            print(f"      Chunk {i+1}/{len(chunks)}...", end=" ")
            concepts = self._pass1_extract(chunk, domain)
            print(f"→ {len(concepts)} concepts")
            all_raw_concepts.extend(concepts)
            time.sleep(1)  # Rate limiting (1s between chunks)

        # Deduplicate (case-insensitive) before Pass 2
        seen = set()
        unique_raw = []
        for c in all_raw_concepts:
            c_lower = c.lower().strip()
            if c_lower and c_lower not in seen:
                seen.add(c_lower)
                unique_raw.append(c)

        print(f"      Total raw: {len(all_raw_concepts)} → Unique: {len(unique_raw)}")

        # ── PASS 2: Refine & Score (Combines old Pass 2 + Pass 3) ──
        print(f"\n   ✨ Pass 2: Refine, Dedup & Score...")
        structured_concepts = self._pass2_refine(unique_raw, full_text, domain)
        print(f"      Final: {len(structured_concepts)} structured concepts")

        # ── Format output to match existing schema ──
        formatted_concepts = self._format_output(structured_concepts)

        result = {
            "video_id": video_id,
            "domain": domain,
            "total_concepts": len(formatted_concepts),
            "extraction_method": "llm_iterative_refinement",
            "llm_model": self.groq_model if self.groq_client else self.model_name,
            "concepts": formatted_concepts,
            "sentences_with_concepts": self._map_concepts_to_sentences(
                data["sentences"], formatted_concepts
            ),
            "llm_metadata": {
                "pass1_raw_count": len(all_raw_concepts),
                "pass1_unique_count": len(unique_raw),
                "pass2_final_count": len(structured_concepts),
                "chunks_processed": len(chunks),
            },
        }

        return result

    def _format_output(self, structured_concepts: List[Dict]) -> List[Dict]:
        """Format LLM output to match the existing ConceptExtractor schema"""
        formatted = []

        for i, concept_data in enumerate(structured_concepts):
            concept_name = concept_data.get("concept", "Unknown")
            confidence = concept_data.get("confidence", 0.5)
            category = concept_data.get("category", "related")
            concept_domain = concept_data.get("domain", "unknown")

            # Map confidence to a combined_score (for compatibility with ranking)
            domain_bonus = 50.0 if category == "core" else (30.0 if category == "prerequisite" else 10.0)
            combined_score = confidence * 10.0 + domain_bonus

            formatted.append({
                "concept": concept_name,
                "frequency": 0,
                "domain_match": 1 if category in ("core", "prerequisite") else 0,
                "keybert_score": 0.0,
                "combined_score": round(combined_score, 3),
                "extraction_method": "llm_pass3",
                "llm_confidence": confidence,
                "llm_category": category,
                "llm_domain": concept_domain,
            })

        formatted.sort(key=lambda x: x["combined_score"], reverse=True)
        return formatted

    def _map_concepts_to_sentences(
        self, sentences: List[Dict], concepts: List[Dict]
    ) -> List[Dict]:
        """Map which concepts appear in which sentences"""
        concept_names = [c["concept"].lower() for c in concepts]

        for sentence in sentences:
            text_lower = sentence["text"].lower()
            sentence["concepts"] = [c for c in concept_names if c in text_lower]

        return sentences

    def save_concepts(self, concepts: Dict, video_id: str):
        """Save extracted concepts"""
        output_file = self.output_dir / f"{video_id}_concepts.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(concepts, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Saved {concepts['total_concepts']} concepts to {output_file}")

    def process_all_segmented(self, videos_config: List[Dict] = None):
        """Process all segmented files"""
        segmented_files = list(self.input_dir.glob("*_segmented.json"))

        if not segmented_files:
            print("No segmented files found!")
            return

        # Create domain mapping if config provided
        domain_map = {}
        if videos_config:
            domain_map = {v["id"]: v["domain"] for v in videos_config}

        for segmented_file in segmented_files:
            video_id = segmented_file.stem.replace("_segmented", "")
            domain = domain_map.get(video_id)

            print(f"\n{'='*60}")
            print(f"LLM Concept Extraction: {video_id}")
            if domain:
                print(f"Domain: {domain}")
            print(f"{'='*60}")

            try:
                concepts = self.extract_concepts_from_video(
                    segmented_file, video_id, domain
                )
                self.save_concepts(concepts, video_id)

            except Exception as e:
                print(f"✗ Error extracting concepts from {video_id}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    from src.utils.config_loader import ConfigLoader

    config = ConfigLoader()
    videos_config = config.load_videos()

    extractor = LLMConceptExtractor()
    extractor.process_all_segmented(videos_config["videos"])
