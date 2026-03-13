"""
Step 1.5: Transcript Translation - Translate Hinglish/Telugu transcripts to English
"""
import json
import os
from pathlib import Path
from typing import Dict, List
from groq import Groq


class TranscriptTranslator:
    """Translate code-mixed transcripts to English using Groq"""
    
    def __init__(self, 
                 input_dir: str = "data/raw",
                 output_dir: str = "data/raw"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set!")
        
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        
        print(f"✓ Groq API initialized (using {self.model})")
    
    def translate_text(self, text: str) -> str:
        """
        Translate Hinglish/Telugu text to English
        
        Args:
            text: Code-mixed text (Hindi+English or Telugu+English)
            
        Returns:
            English translation
        """
        prompt = f"""Translate the following educational video transcript to English.

RULES:
1. Translate Hindi/Telugu words to English
2. Keep technical terms in English (e.g., "recursion", "dynamic programming")
3. Preserve the natural flow and meaning
4. Do NOT add explanations or extra text
5. ONLY return the translated text, nothing else

Original text:
{text}

Translated text:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate Hindi/Hinglish/Telugu to English. Return ONLY the translation, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            
            if response and response.choices:
                return response.choices[0].message.content.strip()
            else:
                print(f"  ⚠ Empty response from Groq")
                return text  # Return original if translation fails
                
        except Exception as e:
            print(f"  ✗ Translation error: {e}")
            return text  # Return original if translation fails
    
    def translate_transcript(self, transcript: List[Dict]) -> List[Dict]:
        """
        Translate entire transcript while preserving timestamps
        
        Args:
            transcript: List of {text, start, duration} chunks
            
        Returns:
            Translated transcript with same structure
        """
        # Translate each chunk individually for accuracy
        # (Batching was causing LLM to only translate first item)
        translated_transcript = []
        
        for i, chunk in enumerate(transcript):
            # Progress indicator every 10 chunks
            if i % 10 == 0:
                print(f"  Translating chunk {i+1} of {len(transcript)}...")
            
            # Translate this chunk
            translated_text = self.translate_text(chunk['text'])
            
            translated_transcript.append({
                'text': translated_text,
                'start': chunk['start'],
                'duration': chunk['duration'],
                'original_text': chunk['text']  # Keep original for reference
            })
            
            # Small delay every 20 chunks to avoid rate limiting
            if (i + 1) % 20 == 0:
                import time
                time.sleep(1)
        
        return translated_transcript
    
    def process_video(self, video_id: str):
        """Translate transcript for a single video"""
        
        # Load original transcript
        transcript_file = self.input_dir / f"{video_id}_transcript.json"
        
        if not transcript_file.exists():
            print(f"  ✗ Transcript not found: {transcript_file}")
            return
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_transcript = data['transcript']
        
        print(f"\n{'='*60}")
        print(f"Translating: {video_id}")
        print(f"Chunks: {len(original_transcript)}")
        print(f"{'='*60}")
        
        # Translate
        translated_transcript = self.translate_transcript(original_transcript)
        
        # Create translated version
        translated_data = {
            'video_id': data['video_id'],
            'source': 'translated_from_' + data.get('source', 'youtube_api'),
            'transcript': translated_transcript,
            'full_text': " ".join([chunk['text'] for chunk in translated_transcript]),
            'original_full_text': data.get('full_text', ''),
            'translation_model': 'llama-3.3-70b-versatile (Groq)'
        }
        
        # Save translated transcript
        output_file = self.output_dir / f"{video_id}_transcript_en.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Translated transcript saved to {output_file}")
        print(f"  Original chars: {len(data.get('full_text', ''))}")
        print(f"  Translated chars: {len(translated_data['full_text'])}")
    
    def process_all_videos(self):
        """Translate all video transcripts"""
        transcript_files = list(self.input_dir.glob("*_transcript.json"))
        
        # Exclude already translated files
        transcript_files = [f for f in transcript_files if not f.stem.endswith('_en')]
        
        if not transcript_files:
            print("No transcripts found to translate")
            return
        
        print(f"\nFound {len(transcript_files)} transcripts to translate")
        
        for transcript_file in transcript_files:
            video_id = transcript_file.stem.replace('_transcript', '')
            
            # Skip if already translated
            translated_file = self.output_dir / f"{video_id}_transcript_en.json"
            if translated_file.exists():
                print(f"\n⏭  Skipping {video_id} (already translated)")
                continue
            
            try:
                self.process_video(video_id)
            except Exception as e:
                print(f"\n✗ Error translating {video_id}: {e}")
                import traceback
                traceback.print_exc()


def main():
    import sys
    
    translator = TranscriptTranslator()
    
    # Check if specific video ID provided as command line argument
    if len(sys.argv) > 1:
        video_id = sys.argv[1]
        print(f"\n{'='*60}")
        print(f"Translating single video: {video_id}")
        print('='*60)
        translator.process_video(video_id)
    else:
        translator.process_all_videos()


if __name__ == "__main__":
    main()
