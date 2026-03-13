"""
Step 6: LLM-based Prerequisite Verification
Verify candidate edges using Groq LLM
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from groq import Groq


class LLMPrerequisiteVerifier:
    """Verify prerequisite relationships using LLM"""
    
    def __init__(self,
                 candidates_dir: str = "data/processed",
                 output_dir: str = "data/processed",
                 model: str = "llama-3.3-70b-versatile"):
        self.candidates_dir = Path(candidates_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        
        # Initialize Groq client
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.client = Groq(api_key=api_key)
    
    def build_verification_prompt(self, prerequisite: str, target: str, context: str) -> str:
        """Build prompt for LLM verification"""
        
        prompt = f"""Given this lecture transcript segment, determine if "{prerequisite}" is a prerequisite for "{target}" based ONLY on the teacher's explanation.

**Concept A (Potential Prerequisite):** {prerequisite}
**Concept B (Target Concept):** {target}

**Transcript Context:**
{context}

**Task:**
Analyze whether understanding Concept A is necessary BEFORE learning Concept B, based on how the teacher explains it in this specific context.

Answer in this EXACT format:
ANSWER: [YES or NO]
CONFIDENCE: [0.0 to 1.0]
REASONING: [2-3 sentences explaining your decision]

Consider:
- Does the teacher explicitly reference Concept A when explaining Concept B?
- Does Concept B build upon or extend Concept A?
- Would a student need to understand Concept A first to grasp Concept B?
- Are there teaching signals like "before", "first", "recall", "using", "based on"?

Answer:"""
        
        return prompt
    
    def verify_edge(self, prerequisite: str, target: str, context: str) -> Dict:
        """
        Verify if prerequisite → target relationship is valid
        
        Returns:
            {
                'is_prerequisite': bool,
                'confidence': float,
                'reasoning': str,
                'llm_response': str
            }
        """
        
        prompt = self.build_verification_prompt(prerequisite, target, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educational content analyzer specializing in prerequisite relationship detection in lecture transcripts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent responses
                max_tokens=300
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Parse response
            is_prerequisite = False
            confidence = 0.0
            reasoning = ""
            
            for line in llm_response.split('\n'):
                if line.startswith('ANSWER:'):
                    answer = line.split(':', 1)[1].strip().upper()
                    is_prerequisite = answer == 'YES'
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except:
                        confidence = 0.5
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            return {
                'is_prerequisite': is_prerequisite,
                'confidence': confidence,
                'reasoning': reasoning,
                'llm_response': llm_response
            }
            
        except Exception as e:
            print(f"  ✗ LLM verification error: {e}")
            return {
                'is_prerequisite': False,
                'confidence': 0.0,
                'reasoning': f"Error: {str(e)}",
                'llm_response': ""
            }
    
    def process_video(self, video_id: str) -> Dict:
        """Process all candidate edges for a video"""
        
        # Load candidates
        candidates_file = self.candidates_dir / f"{video_id}_prerequisite_candidates.json"
        
        if not candidates_file.exists():
            print(f"✗ No candidates file for {video_id}")
            return None
        
        with open(candidates_file, 'r', encoding='utf-8') as f:
            candidates_data = json.load(f)
        
        candidate_edges = candidates_data['candidate_edges']
        
        print(f"\n{'='*60}")
        print(f"Verifying: {video_id}")
        print(f"Candidate edges: {len(candidate_edges)}")
        print(f"{'='*60}")
        
        verified_edges = []
        
        for i, edge in enumerate(candidate_edges, 1):
            print(f"\n[{i}/{len(candidate_edges)}] Verifying: {edge['prerequisite']} → {edge['target']}")
            
            # Verify with LLM
            verification = self.verify_edge(
                edge['prerequisite'],
                edge['target'],
                edge['context_window']
            )
            
            # Combine with edge data
            verified_edge = {
                **edge,
                'llm_verified': verification['is_prerequisite'],
                'llm_confidence': verification['confidence'],
                'llm_reasoning': verification['reasoning'],
                'llm_response': verification['llm_response']
            }
            
            verified_edges.append(verified_edge)
            
            status = "✓" if verification['is_prerequisite'] else "✗"
            print(f"  {status} {verification['is_prerequisite']} (confidence: {verification['confidence']:.2f})")
            print(f"  Reason: {verification['reasoning'][:100]}...")
        
        # Filter to keep only verified edges
        confirmed_edges = [e for e in verified_edges if e['llm_verified']]
        
        print(f"\n{'='*60}")
        print(f"✓ Verified edges: {len(confirmed_edges)}/{len(candidate_edges)}")
        print(f"{'='*60}")
        
        result = {
            'video_id': video_id,
            'total_candidates': len(candidate_edges),
            'total_verified': len(confirmed_edges),
            'all_edges': verified_edges,
            'confirmed_edges': confirmed_edges
        }
        
        return result
    
    def save_results(self, result: Dict, video_id: str):
        """Save verification results"""
        output_file = self.output_dir / f"{video_id}_prerequisites.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to {output_file}")
    
    def process_all_videos(self):
        """Process all videos"""
        candidate_files = list(self.candidates_dir.glob("*_prerequisite_candidates.json"))
        
        for candidate_file in candidate_files:
            video_id = candidate_file.stem.replace('_prerequisite_candidates', '')
            
            try:
                result = self.process_video(video_id)
                if result:
                    self.save_results(result, video_id)
            except Exception as e:
                print(f"✗ Error processing {video_id}: {e}")
                import traceback
                traceback.print_exc()


def main():
    verifier = LLMPrerequisiteVerifier()
    verifier.process_all_videos()


if __name__ == "__main__":
    main()
