#!/usr/bin/env python3
"""
Run only Steps 5-8 for video_2 (assuming concepts already extracted)
"""
import os
import sys

# Add paths
sys.path.insert(0, 'src/5_prerequisite_detection')
sys.path.insert(0, 'src/6_relation_classification')

from window_based_detector import WindowBasedPrerequisiteDetector
from llm_verifier import LLMPrerequisiteVerifier
import subprocess

def main():
    video_id = "video_2"
    
    print(f"\n{'='*70}")
    print(f"🚀 Running Steps 5-8 for {video_id}")
    print(f"{'='*70}\n")
    
    # Check for API key
    if not os.getenv('GROQ_API_KEY'):
        print("❌ GROQ_API_KEY not set!")
        return
    
    # Step 5: Detect prerequisites
    print("▶ Step 5: Detecting prerequisite candidates...")
    detector = WindowBasedPrerequisiteDetector()
    candidates_result = detector.process_video(video_id)
    
    if candidates_result:
        detector.save_candidates(candidates_result, video_id)
        num_candidates = len(candidates_result.get('candidates', []))
        print(f"✓ Found {num_candidates} candidates\n")
    else:
        print("✗ Failed to detect candidates\n")
        return
    
    # Step 6: Verify with LLM
    print("▶ Step 6: Verifying prerequisites with LLM...")
    verifier = LLMPrerequisiteVerifier()
    verified_result = verifier.process_video(video_id)
    
    if verified_result:
        verifier.save_results(verified_result, video_id)
        confirmed = len([e for e in verified_result.get('all_edges', []) if e.get('llm_verified')])
        total = verified_result.get('total_candidates', 0)
        print(f"✓ Verified {confirmed}/{total} prerequisites\n")
    else:
        print("✗ Failed to verify prerequisites\n")
        return
    
    # Step 7 & 8: Create visualization
    print("▶ Steps 7-8: Creating DAG visualization...")
    result = subprocess.run(
        ['python', 'create_dag_graph.py', video_id],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Visualization created\n")
        print(f"\n{'='*70}")
        print(f"✅ COMPLETED!")
        print(f"{'='*70}")
        print(f"\n📊 Visualization: data/visualizations/{video_id}_dag.png")
        print(f"📄 Prerequisites: data/processed/{video_id}_prerequisites.json\n")
    else:
        print("✗ Failed to create visualization")
        print(result.stderr)

if __name__ == "__main__":
    main()
