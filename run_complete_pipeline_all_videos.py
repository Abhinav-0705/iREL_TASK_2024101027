#!/usr/bin/env python3
"""
Complete Pipeline Runner for All Videos
Runs Steps 1-8 for all configured videos:
1. Extract transcript
2. Segment into chunks
3. Extract concepts (2-pass LLM method)
4. Normalize concepts
1.5. Translate to English
5. Window-based prerequisite detection
6. LLM verification
7. Build DAG
8. Visualize
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
sys.path.insert(0, 'src/5_prerequisite_detection')
sys.path.insert(0, 'src/6_llm_verification')
sys.path.insert(0, 'src/7_hypergraph_construction')
sys.path.insert(0, 'src/8_visualization')

# Import all required modules
from transcript_extractor import TranscriptExtractor
from sentence_segmenter import SentenceSegmenter
from llm_concept_extractor import LLMConceptExtractor
from linguistic_normalizer import LinguisticNormalizer
from transcript_translator import TranscriptTranslator
from window_based_detector import WindowBasedPrerequisiteDetector
from llm_verifier import LLMPrerequisiteVerifier
from dag_builder import ConceptDAGBuilder

def load_videos_config():
    """Load video configurations"""
    with open('config/videos.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['videos']

def check_file_exists(filepath):
    """Check if a file exists"""
    return Path(filepath).exists()

def run_step1_extract_transcript(video):
    """Step 1: Extract transcript from YouTube"""
    video_id = video['id']
    url = video['url']
    
    transcript_file = f"data/raw/{video_id}_transcript.json"
    
    if check_file_exists(transcript_file):
        print(f"  ✓ Transcript already exists: {transcript_file}")
        return True
    
    print(f"  📥 Extracting transcript from YouTube...")
    extractor = TranscriptExtractor()
    result = extractor.extract_transcript(url, video_id, prefer_whisper=False)
    
    if result:
        print(f"  ✓ Saved transcript: {transcript_file}")
        return True
    else:
        print(f"  ✗ Failed to extract transcript")
        return False

def run_step2_segment_transcript(video_id):
    """Step 2: Segment transcript into chunks"""
    segments_file = f"data/processed/{video_id}_segments.json"
    
    if check_file_exists(segments_file):
        print(f"  ✓ Segments already exist: {segments_file}")
        return True
    
    print(f"  📝 Segmenting transcript...")
    segmenter = SentenceSegmenter()
    result = segmenter.segment_video(video_id)
    
    if result:
        print(f"  ✓ Created {result['total_segments']} segments")
        return True
    else:
        print(f"  ✗ Failed to segment transcript")
        return False

def run_step3_extract_concepts(video_id):
    """Step 3: Extract concepts using 2-pass LLM method"""
    concepts_file = f"data/processed/{video_id}_concepts.json"
    
    if check_file_exists(concepts_file):
        print(f"  ✓ Concepts already extracted: {concepts_file}")
        return True
    
    print(f"  🧠 Extracting concepts (2-pass LLM method)...")
    extractor = LLMConceptExtractor()
    result = extractor.extract_concepts(video_id)
    
    if result:
        print(f"  ✓ Extracted {result.get('total_concepts', 0)} concepts")
        return True
    else:
        print(f"  ✗ Failed to extract concepts")
        return False

def run_step4_normalize_concepts(video_id):
    """Step 4: Normalize concepts"""
    normalized_file = f"data/processed/{video_id}_concepts_normalized.json"
    
    if check_file_exists(normalized_file):
        print(f"  ✓ Concepts already normalized: {normalized_file}")
        return True
    
    print(f"  🔧 Normalizing concepts...")
    normalizer = LinguisticNormalizer()
    result = normalizer.normalize_video(video_id)
    
    if result:
        print(f"  ✓ Normalized concepts")
        return True
    else:
        print(f"  ✗ Failed to normalize concepts")
        return False

def run_step1_5_translate_transcript(video_id):
    """Step 1.5: Translate transcript to English"""
    translated_file = f"data/raw/{video_id}_transcript_en.json"
    
    if check_file_exists(translated_file):
        print(f"  ✓ Translation already exists: {translated_file}")
        return True
    
    print(f"  🌐 Translating transcript to English...")
    translator = TranscriptTranslator()
    result = translator.translate_video(video_id)
    
    if result:
        print(f"  ✓ Translated transcript saved")
        return True
    else:
        print(f"  ✗ Failed to translate transcript")
        return False

def run_step5_detect_prerequisites(video_id):
    """Step 5: Window-based prerequisite detection"""
    candidates_file = f"data/processed/{video_id}_prerequisite_candidates.json"
    
    print(f"  🔍 Detecting prerequisite candidates...")
    detector = WindowBasedPrerequisiteDetector()
    result = detector.process_video(video_id)
    
    if result:
        detector.save_candidates(result, video_id)
        print(f"  ✓ Found {len(result.get('candidates', []))} candidates")
        return True
    else:
        print(f"  ✗ Failed to detect prerequisites")
        return False

def run_step6_verify_prerequisites(video_id):
    """Step 6: LLM verification of prerequisites"""
    verified_file = f"data/processed/{video_id}_prerequisites.json"
    
    print(f"  ✅ Verifying prerequisites with LLM...")
    verifier = LLMPrerequisiteVerifier()
    result = verifier.process_video(video_id)
    
    if result:
        verifier.save_results(result, video_id)
        confirmed = len([e for e in result.get('all_edges', []) if e.get('llm_verified')])
        print(f"  ✓ Verified {confirmed}/{result.get('total_candidates', 0)} prerequisites")
        return True
    else:
        print(f"  ✗ Failed to verify prerequisites")
        return False

def run_step7_build_dag(video_id):
    """Step 7: Build DAG structure"""
    dag_file = f"data/output/{video_id}_dag.json"
    
    print(f"  🔗 Building DAG structure...")
    builder = ConceptDAGBuilder()
    result = builder.process_video(video_id)
    
    if result:
        builder.save_dag(result, video_id)
        print(f"  ✓ Built DAG with {result.get('num_nodes', 0)} nodes, {result.get('num_edges', 0)} edges")
        return True
    else:
        print(f"  ✗ Failed to build DAG")
        return False

def run_step8_visualize(video_id):
    """Step 8: Create visualization"""
    print(f"  📊 Creating visualization...")
    
    # Use the create_dag_graph script
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
        print(result.stderr)
        return False

def run_pipeline_for_video(video):
    """Run complete pipeline for a single video"""
    video_id = video['id']
    title = video['title']
    
    print(f"\n{'='*70}")
    print(f"🎬 Processing: {video_id} - {title}")
    print(f"{'='*70}")
    
    steps = [
        ("Step 1: Extract Transcript", lambda: run_step1_extract_transcript(video)),
        ("Step 2: Segment Transcript", lambda: run_step2_segment_transcript(video_id)),
        ("Step 3: Extract Concepts", lambda: run_step3_extract_concepts(video_id)),
        ("Step 4: Normalize Concepts", lambda: run_step4_normalize_concepts(video_id)),
        ("Step 1.5: Translate to English", lambda: run_step1_5_translate_transcript(video_id)),
        ("Step 5: Detect Prerequisites", lambda: run_step5_detect_prerequisites(video_id)),
        ("Step 6: Verify Prerequisites", lambda: run_step6_verify_prerequisites(video_id)),
        ("Step 7: Build DAG", lambda: run_step7_build_dag(video_id)),
        ("Step 8: Visualize", lambda: run_step8_visualize(video_id)),
    ]
    
    for step_name, step_func in steps:
        print(f"\n▶ {step_name}")
        try:
            success = step_func()
            if not success:
                print(f"  ⚠️  Step failed, but continuing...")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            print(f"  ⚠️  Continuing with next step...")
        
        # Small delay to avoid rate limits
        time.sleep(1)
    
    print(f"\n✅ Completed pipeline for {video_id}\n")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("🚀 COMPLETE PIPELINE RUNNER - ALL VIDEOS")
    print("="*70)
    
    # Check for API keys
    if not os.getenv('GROQ_API_KEY'):
        print("\n⚠️  Warning: GROQ_API_KEY not set!")
        print("Set it with: export GROQ_API_KEY='your-key'")
        return
    
    # Load video configurations
    videos = load_videos_config()
    print(f"\n📹 Found {len(videos)} videos to process\n")
    
    # Process each video
    for video in videos:
        run_pipeline_for_video(video)
        print("\n" + "-"*70)
    
    print("\n" + "="*70)
    print("🎉 ALL VIDEOS PROCESSED!")
    print("="*70)
    print("\n📊 Visualizations saved to: data/visualizations/")
    print("📄 DAG structures saved to: data/output/")
    print("\n")

if __name__ == "__main__":
    main()
