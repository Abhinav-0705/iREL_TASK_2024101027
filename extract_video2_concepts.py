#!/usr/bin/env python3
"""
Extract and save concepts for video_2
"""
import sys
import json
from pathlib import Path

# Add source paths
sys.path.insert(0, 'src/3_concept_extraction')
from llm_concept_extractor import LLMConceptExtractor

def extract_and_save_video2_concepts():
    """Extract and save concepts for video_2"""
    video_id = "video_2"
    
    print(f"🧠 Extracting and saving concepts for {video_id}...")
    
    # Check if segments exist
    segments_file = f"data/processed/{video_id}_segments.json"
    if not Path(segments_file).exists():
        print(f"❌ Segments file not found: {segments_file}")
        return False
    
    # Initialize extractor
    extractor = LLMConceptExtractor()
    
    try:
        # Extract concepts
        print("🤖 Running concept extraction...")
        result = extractor.extract_concepts_from_video(segments_file, video_id)
        
        if result and result.get('total_concepts', 0) > 0:
            # Save concepts
            extractor.save_concepts(result, video_id)
            print(f"✅ Successfully extracted and saved {result['total_concepts']} concepts!")
            
            # Show first few concepts
            concepts = result.get('concepts', [])
            if concepts:
                print("\n📋 First 5 concepts:")
                for i, concept in enumerate(concepts[:5], 1):
                    concept_name = concept.get('concept', 'N/A')
                    confidence = concept.get('llm_confidence', 0)
                    print(f"   {i}. {concept_name} (confidence: {confidence:.2f})")
            
            return True
        else:
            print("❌ No concepts extracted")
            return False
            
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = extract_and_save_video2_concepts()
    if success:
        print("\n🎉 Concept extraction completed successfully!")
    else:
        print("\n💥 Concept extraction failed!")