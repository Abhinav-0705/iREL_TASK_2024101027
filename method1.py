#!/usr/bin/env python3
"""Method 1: Full explicit prerequisite pipeline for a single video.

This file contains all the logic needed to run the 1→8 pipeline for a
single video ID, without depending on run_single_video.py.

Usage:
    python method1.py video_3
"""

import json
import os
import sys
from pathlib import Path

import yaml

# Make src packages importable
sys.path.insert(0, "src/1_data_collection")
sys.path.insert(0, "src/2_segmentation")
sys.path.insert(0, "src/3_concept_extraction")
sys.path.insert(0, "src/4_normalization")
sys.path.insert(0, "src/5_prerequisite_detection")
sys.path.insert(0, "src/6_relation_classification")

from transcript_extractor import TranscriptExtractor
from llm_concept_extractor import LLMConceptExtractor
from transcript_translator import TranscriptTranslator
from window_based_detector import WindowBasedPrerequisiteDetector
from llm_verifier import LLMPrerequisiteVerifier


def load_video_config(video_id: str) -> dict | None:
    """Load configuration for specific video from config/videos.yaml."""
    with open("config/videos.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for video in config["videos"]:
        if video["id"] == video_id:
            return video
    print(f"❌ Video {video_id} not found in config/videos.yaml")
    return None


def check_file_exists(path: str) -> bool:
    return Path(path).exists()


def step1_extract_transcript(video: dict) -> bool:
    video_id = video["id"]
    url = video["url"]
    transcript_file = f"data/raw/{video_id}_transcript.json"

    if check_file_exists(transcript_file):
        print("  ✓ Transcript already exists, skipping")
        return True

    print(f"  📥 Extracting transcript from: {url}")
    extractor = TranscriptExtractor()
    result = extractor.extract_transcript(url, video_id, prefer_whisper=False)
    if not result:
        print("  ✗ Failed to extract transcript")
        return False

    # Persist transcript
    # The extractor in your project has save_transcript; use same path format.
    try:
        extractor.save_transcript(result, video_id)
    except Exception:
        with open(transcript_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print("  ✓ Saved transcript")
    return True


def step2_segment_transcript(video_id: str) -> bool:
    transcript_file = f"data/raw/{video_id}_transcript.json"
    if not check_file_exists(transcript_file):
        print(f"  ✗ Transcript file missing: {transcript_file}")
        return False

    with open(transcript_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data.get("transcript"), list):
        print("  ✓ Transcript already segmented from YouTube API")
        print(f"  ℹ️  {len(data['transcript'])} chunks found")
        return True

    print("  ⚠️  No explicit segmentation step (using transcript as-is)")
    return True


def step3_extract_concepts(video_id: str) -> bool:
    concepts_file = f"data/processed/{video_id}_concepts.json"
    if check_file_exists(concepts_file):
        print("  ✓ Concepts already extracted, skipping")
        with open(concepts_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  ℹ️  {data.get('total_concepts', 0)} concepts found")
        return True

    print("  🧠 Extracting concepts (2-pass LLM method)...")
    transcript_file = f"data/raw/{video_id}_transcript.json"
    if not check_file_exists(transcript_file):
        print(f"  ✗ Transcript not found: {transcript_file}")
        return False

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    segmented = {
        "video_id": video_id,
        "sentences": [{"text": c["text"]} for c in transcript_data["transcript"]],
    }
    segments_path = Path(f"data/processed/{video_id}_segments.json")
    segments_path.parent.mkdir(parents=True, exist_ok=True)
    with segments_path.open("w", encoding="utf-8") as f:
        json.dump(segmented, f, ensure_ascii=False, indent=2)

    extractor = LLMConceptExtractor()
    result = extractor.extract_concepts_from_video(str(segments_path), video_id)
    if not result:
        print("  ✗ Failed to extract concepts")
        return False

    extractor.save_concepts(result, video_id)
    print(f"  ✓ Extracted {result.get('total_concepts', 0)} concepts")
    return True


def step4_normalize_concepts(video_id: str) -> bool:
    print("  ⏭  Skipping normalization (concepts already in final form)")
    return True


def step1_5_translate_transcript(video_id: str) -> bool:
    translated_file = f"data/raw/{video_id}_transcript_en.json"
    if check_file_exists(translated_file):
        print("  ✓ Translation already exists, skipping")
        with open(translated_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  ℹ️  {len(data.get('transcript', []))} chunks translated")
        return True

    print("  🌐 Translating transcript to English...")
    print("  ⏳ This may take a few minutes...")
    translator = TranscriptTranslator()
    translator.process_video(video_id)

    if not check_file_exists(translated_file):
        print("  ✗ Translation failed")
        return False

    print("  ✓ Translation completed")
    return True


def steps5_to_8(video_id: str) -> bool:
    print("  🔍 Running Steps 5-8 (detection → verification → DAG → visualization)...")

    # Step 5: detect candidates
    print("\n  ▶ Step 5: Detecting prerequisite candidates...")
    detector = WindowBasedPrerequisiteDetector()
    candidates = detector.process_video(video_id)
    if not candidates:
        print("  ✗ Failed to detect candidates")
        return False
    detector.save_candidates(candidates, video_id)
    num = len(candidates.get("candidates", []))
    print(f"  ✓ Found {num} candidates")

    # Step 6: verify with LLM
    print("\n  ▶ Step 6: Verifying prerequisites with LLM...")
    verifier = LLMPrerequisiteVerifier()
    verified = verifier.process_video(video_id)
    if not verified:
        print("  ✗ Failed to verify prerequisites")
        return False
    verifier.save_results(verified, video_id)
    confirmed = len([e for e in verified.get("all_edges", []) if e.get("llm_verified")])
    total = verified.get("total_candidates", 0)
    print(f"  ✓ Verified {confirmed}/{total} prerequisites")

    # Step 7-8: create DAG visualization using existing script
    print("\n  ▶ Steps 7-8: Creating DAG visualization...")
    from subprocess import run

    result = run([sys.executable, "create_dag_graph.py", video_id])
    if result.returncode != 0:
        print("  ✗ Failed to create visualization")
        return False
    print("  ✓ Visualization created")
    return True


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python method1.py <video_id>")
        print("Example: python method1.py video_2")
        sys.exit(1)

    video_id = sys.argv[1]

    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠️  GROQ_API_KEY not set. Set it before running:")
        print("   export GROQ_API_KEY='your-key'")
        sys.exit(1)

    video = load_video_config(video_id)
    if not video:
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"🚀 SINGLE VIDEO PIPELINE (method1): {video_id}")
    print("=" * 70)
    print(f"\n📹 Video: {video['title']}")
    print(f"🌐 Domain: {video['domain']}")
    print(f"🗣️  Language: {video['language']}")
    print(f"🔗 URL: {video['url']}\n")

    steps = [
        ("Step 1: Extract Transcript", lambda: step1_extract_transcript(video)),
        ("Step 2: Segment Transcript", lambda: step2_segment_transcript(video_id)),
        ("Step 3: Extract Concepts (2-pass)", lambda: step3_extract_concepts(video_id)),
        ("Step 4: Normalize Concepts", lambda: step4_normalize_concepts(video_id)),
        ("Step 1.5: Translate to English", lambda: step1_5_translate_transcript(video_id)),
        ("Steps 5-8: Prerequisites → DAG → Viz", lambda: steps5_to_8(video_id)),
    ]

    for label, func in steps:
        print("\n" + "-" * 70)
        print(f"▶ {label}")
        print("-" * 70)
        ok = func()
        if not ok:
            print(f"\n❌ Pipeline stopped at: {label}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f"✅ PIPELINE COMPLETED FOR {video_id} (method1)")
    print("=" * 70)


if __name__ == "__main__":
    main()
