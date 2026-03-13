#!/usr/bin/env python3
"""
Single Video Pipeline Runner
Runs complete pipeline (Steps 1-8) for one video at a time
Usage: python run_single_video.py <video_id>
Example: python run_single_video.py video_2
"""

import os
import sys
import json
import time
import yaml
from pathlib import Path

# Add src directories to path
sys.path.insert(0, 'src/1_data_collection')
sys.path.insert(0, 'src/2_segmentation')
sys.path.insert(0, 'src/3_concept_extraction')
sys.path.insert(0, 'src/4_normalization')

# Import modules
from transcript_extractor import TranscriptExtractor
from sentence_segmenter import SentenceSegmenter
from llm_concept_extractor import LLMConceptExtractor
from linguistic_normalizer import LinguisticNormalizer
from transcript_translator import TranscriptTranslator

def load_video_config(video_id):
    """Load configuration for specific video"""
    with open('config/videos.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    for video in config['videos']:
        if video['id'] == video_id:
            return video
    
    print(f"❌ Video {video_id} not found in config/videos.yaml")
    return None

def check_file_exists(filepath):
    """Check if a file exists"""
    return Path(filepath).exists()

def run_step1_extract_transcript(video):
    """Step 1: Extract transcript from YouTube"""
    video_id = video['id']
    url = video['url']
    
    transcript_file = f"data/raw/{video_id}_transcript.json"
    
    if check_file_exists(transcript_file):
        print(f"  ✓ Transcript already exists, skipping")
        return True
    
    print(f"  📥 Extracting transcript from: {url}")
    extractor = TranscriptExtractor()
    result = extractor.extract_transcript(url, video_id, prefer_whisper=False)
    
    if result:
        print(f"  ✓ Saved transcript")
        return True
    else:
        print(f"  ✗ Failed to extract transcript")
        return False

def run_step2_segment_transcript(video_id):
    """Step 2: Segment transcript into chunks"""
    # Check if transcript is already segmented (from YouTube API)
    transcript_file = f"data/raw/{video_id}_transcript.json"
    
    if check_file_exists(transcript_file):
        with open(transcript_file, 'r') as f:
            data = json.load(f)
            if 'transcript' in data and isinstance(data['transcript'], list):
                print(f"  ✓ Transcript already segmented from YouTube API")
                print(f"  ℹ️  {len(data['transcript'])} chunks found")
                return True
    
    segments_file = f"data/processed/{video_id}_segments.json"
    
    if check_file_exists(segments_file):
        print(f"  ✓ Segments already exist, skipping")
        return True
    
    print(f"  ⚠️  No segmentation needed - using YouTube transcript chunks directly")
    return True

def run_step3_extract_concepts(video_id):
    """Step 3: Extract concepts using 2-pass LLM method"""
    concepts_file = f"data/processed/{video_id}_concepts.json"
    
    if check_file_exists(concepts_file):
        print(f"  ✓ Concepts already extracted, skipping")
        with open(concepts_file, 'r') as f:
            data = json.load(f)
            print(f"  ℹ️  {data.get('total_concepts', 0)} concepts found")
        return True
    
    print(f"  🧠 Extracting concepts (2-pass LLM method)...")
    
    # Use the transcript directly as segmented data
    transcript_file = f"data/raw/{video_id}_transcript.json"
    if not check_file_exists(transcript_file):
        print(f"  ✗ Transcript file not found: {transcript_file}")
        return False
    
    # Load transcript and convert to segmented format
    with open(transcript_file, 'r') as f:
        transcript_data = json.load(f)
    
    # Create segmented data format
    segmented_data = {
        'video_id': video_id,
        'sentences': [{'text': chunk['text']} for chunk in transcript_data['transcript']]
    }
    
    # Save temporary segmented file
    temp_segments_file = f"data/processed/{video_id}_segments.json"
    with open(temp_segments_file, 'w') as f:
        json.dump(segmented_data, f, ensure_ascii=False, indent=2)
    
    extractor = LLMConceptExtractor()
    result = extractor.extract_concepts_from_video(temp_segments_file, video_id)
    
    if result:
        # Save the extracted concepts
        extractor.save_concepts(result, video_id)
        print(f"  ✓ Extracted {result.get('total_concepts', 0)} concepts")
        return True
    else:
        print(f"  ✗ Failed to extract concepts")
        return False

def run_step4_normalize_concepts(video_id):
    """Step 4: Normalize concepts (SKIPPED - not needed)"""
    print(f"  ⏭  Skipping normalization (concepts already in final form)")
    return True

def run_step1_5_translate_transcript(video_id):
    """Step 1.5: Translate transcript to English"""
    translated_file = f"data/raw/{video_id}_transcript_en.json"
    
    if check_file_exists(translated_file):
        print(f"  ✓ Translation already exists, skipping")
        with open(translated_file, 'r') as f:
            data = json.load(f)
            print(f"  ℹ️  {len(data.get('transcript', []))} chunks translated")
        return True
    
    print(f"  🌐 Translating transcript to English...")
    print(f"  ⏳ This may take a few minutes...")
    translator = TranscriptTranslator()
    translator.process_video(video_id)
    
    # Check if translation was successful
    if check_file_exists(translated_file):
        print(f"  ✓ Translation completed")
        return True
    else:
        print(f"  ✗ Translation failed")
        return False

def run_steps_5_to_8(video_id):
    """Steps 5-8: Run prerequisite detection, verification, DAG building, and visualization"""
    print(f"  🔍 Running Steps 5-8 (detection → verification → DAG → visualization)...")
    
    # Import here to avoid early failures
    sys.path.insert(0, 'src/5_prerequisite_detection')
    sys.path.insert(0, 'src/6_relation_classification')
    
    from window_based_detector import WindowBasedPrerequisiteDetector
    from llm_verifier import LLMPrerequisiteVerifier
    
    # Step 5: Detect candidates
    print(f"\n  ▶ Step 5: Detecting prerequisite candidates...")
    detector = WindowBasedPrerequisiteDetector()
    candidates_result = detector.process_video(video_id)
    
    if candidates_result:
        detector.save_candidates(candidates_result, video_id)
        num_candidates = len(candidates_result.get('candidates', []))
        print(f"  ✓ Found {num_candidates} candidates")
    else:
        print(f"  ✗ Failed to detect candidates")
        return False
    
    # Step 6: Verify with LLM
    print(f"\n  ▶ Step 6: Verifying prerequisites with LLM...")
    verifier = LLMPrerequisiteVerifier()
    verified_result = verifier.process_video(video_id)
    
    if verified_result:
        verifier.save_results(verified_result, video_id)
        confirmed = len([e for e in verified_result.get('all_edges', []) if e.get('llm_verified')])
        total = verified_result.get('total_candidates', 0)
        print(f"  ✓ Verified {confirmed}/{total} prerequisites")
    else:
        print(f"  ✗ Failed to verify prerequisites")
        return False
    
    # Step 7 & 8: Create visualization
    print(f"\n  ▶ Steps 7-8: Creating DAG visualization...")
    import subprocess
    result = subprocess.run(
        ['python', 'create_dag_graph.py', video_id],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"  ✓ Visualization created")
        return True
    else:
        print(f"  ✗ Failed to create visualization")
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("\n❌ Usage: python run_single_video.py <video_id>")
        print("Example: python run_single_video.py video_2")
        print("\nAvailable videos:")
        with open('config/videos.yaml', 'r') as f:
            config = yaml.safe_load(f)
            for video in config['videos']:
                print(f"  - {video['id']}: {video['title']}")
        sys.exit(1)
    
    video_id = sys.argv[1]
    
    print("\n" + "="*70)
    print(f"🚀 SINGLE VIDEO PIPELINE: {video_id}")
    print("="*70)
    
    # Check for API key
    if not os.getenv('GROQ_API_KEY'):
        print("\n⚠️  Warning: GROQ_API_KEY not set!")
        print("Set it with: export GROQ_API_KEY='your-key'")
        sys.exit(1)
    
    # Load video config
    video = load_video_config(video_id)
    if not video:
        sys.exit(1)
    
    print(f"\n📹 Video: {video['title']}")
    print(f"🌐 Domain: {video['domain']}")
    print(f"🗣️  Language: {video['language']}")
    print(f"🔗 URL: {video['url']}\n")
    
    # Run pipeline steps
    steps = [
        ("Step 1: Extract Transcript", lambda: run_step1_extract_transcript(video)),
        ("Step 2: Segment Transcript", lambda: run_step2_segment_transcript(video_id)),
        ("Step 3: Extract Concepts (2-pass)", lambda: run_step3_extract_concepts(video_id)),
        ("Step 4: Normalize Concepts", lambda: run_step4_normalize_concepts(video_id)),
        ("Step 1.5: Translate to English", lambda: run_step1_5_translate_transcript(video_id)),
        ("Steps 5-8: Prerequisites → DAG → Viz", lambda: run_steps_5_to_8(video_id)),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'─'*70}")
        print(f"▶ {step_name}")
        print(f"{'─'*70}")
        
        try:
            success = step_func()
            if not success:
                print(f"\n⚠️  Step failed, stopping pipeline")
                sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        time.sleep(0.5)
    
    print(f"\n" + "="*70)
    print(f"✅ PIPELINE COMPLETED FOR {video_id}!")
    print("="*70)
    print(f"\n📊 Visualization: data/visualizations/{video_id}_dag.png")
    print(f"📄 DAG structure: data/output/{video_id}_dag.json")
    print(f"📋 Prerequisites: data/processed/{video_id}_prerequisites.json\n")

if __name__ == "__main__":
    main()
