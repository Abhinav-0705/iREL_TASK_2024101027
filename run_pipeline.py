"""
Simplified Pipeline Runner
Run individual pipeline steps
"""
import sys
import json
from pathlib import Path

def run_step_1():
    """Step 1: Data Collection"""
    print("=" * 60)
    print("STEP 1: DATA COLLECTION - Extract Transcripts")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / "src" / "1_data_collection"))
    from transcript_extractor import TranscriptExtractor
    from src.utils.config_loader import ConfigLoader
    
    config = ConfigLoader()
    videos = config.load_videos()
    
    extractor = TranscriptExtractor()
    extractor.process_videos(videos['videos'], prefer_whisper=False)
    
    print("\n✓ Step 1 completed!")

def run_step_1_5():
    """Step 1.5: Transcript Translation (Hinglish/Telugu → English)"""
    print("=" * 60)
    print("STEP 1.5: TRANSCRIPT TRANSLATION")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / "src" / "1_data_collection"))
    from transcript_translator import TranscriptTranslator
    
    translator = TranscriptTranslator()
    translator.process_all_videos()
    
    print("\n✓ Step 1.5 completed!")

def run_step_2():
    """Step 2: Sentence Segmentation"""
    print("=" * 60)
    print("STEP 2: SENTENCE SEGMENTATION")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / "src" / "2_segmentation"))
    from sentence_segmenter import SentenceSegmenter
    
    segmenter = SentenceSegmenter()
    segmenter.process_all_transcripts()
    
    print("\n✓ Step 2 completed!")

def run_step_3():
    """Step 3: Concept Extraction (LLM or Classical)"""
    print("=" * 60)
    print("STEP 3: CONCEPT EXTRACTION")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / "src" / "3_concept_extraction"))
    from src.utils.config_loader import ConfigLoader
    
    config = ConfigLoader()
    videos = config.load_videos()
    settings = config.load_settings()
    
    # Check config for extraction method
    ce_config = settings.get('concept_extraction', {})
    method = ce_config.get('method', 'classical')
    
    if method == 'llm':
        print("Using LLM-based concept extraction (Pipeline 3)")
        from llm_concept_extractor import LLMConceptExtractor
        llm_config = ce_config.get('llm', {})
        extractor = LLMConceptExtractor(
            model_name=llm_config.get('model', 'gemini-2.0-flash'),
            temperature=llm_config.get('temperature', 0.3),
            max_concepts=llm_config.get('max_concepts', 20),
            chunk_size=llm_config.get('chunk_size', 500),
            chunk_overlap=llm_config.get('chunk_overlap', 50),
            groq_model=llm_config.get('groq_model', 'llama-3.3-70b-versatile'),
        )
    else:
        print("Using classical concept extraction (KeyBERT/TF-IDF/NER)")
        from concept_extractor import ConceptExtractor
        extractor = ConceptExtractor()
    
    extractor.process_all_segmented(videos['videos'])
    
    print("\n✓ Step 3 completed!")

def run_step_4():
    """Step 4: Linguistic Normalization"""
    print("=" * 60)
    print("STEP 4: LINGUISTIC NORMALIZATION")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / "src" / "4_normalization"))
    from linguistic_normalizer import LinguisticNormalizer
    from src.utils.config_loader import ConfigLoader
    
    config = ConfigLoader()
    videos = config.load_videos()
    
    normalizer = LinguisticNormalizer()
    normalizer.process_all_concepts(videos['videos'])
    
    print("\n✓ Step 4 completed!")

def run_step_5():
    """Step 5: Window-based Prerequisite Detection (±30 sec + teaching signals)"""
    print("=" * 60)
    print("STEP 5: PREREQUISITE DETECTION (Window-based)")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / "src" / "5_prerequisite_detection"))
    from window_based_detector import WindowBasedPrerequisiteDetector
    
    detector = WindowBasedPrerequisiteDetector(window_seconds=30)
    detector.process_all_videos()
    
    print("\n✓ Step 5 completed!")

def run_step_6():
    """Step 6: LLM-based Prerequisite Verification"""
    print("=" * 60)
    print("STEP 6: LLM PREREQUISITE VERIFICATION")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / "src" / "6_relation_classification"))
    from llm_verifier import LLMPrerequisiteVerifier
    
    verifier = LLMPrerequisiteVerifier()
    verifier.process_all_videos()
    
    print("\n✓ Step 6 completed!")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / "src" / "5_prerequisite_detection"))
    from prerequisite_detector import PrerequisiteDetector
    
    detector = PrerequisiteDetector()
    detector.process_all_normalized()
    
    print("\n✓ Step 5 completed!")

def run_step_6():
    """Step 6: Relation Classification"""
    print("=" * 60)
    print("STEP 6: RELATION CLASSIFICATION")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / "src" / "6_relation_classification"))
    from relation_classifier import RelationClassifier
    
    classifier = RelationClassifier()
    classifier.process_all_prerequisites()
    
    print("\n✓ Step 6 completed!")

def run_step_7():
    """Step 7: Build Concept DAG"""
    print("=" * 60)
    print("STEP 7: BUILD CONCEPT DAG")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / "src" / "7_hypergraph_construction"))
    from dag_builder import ConceptDAGBuilder
    
    builder = ConceptDAGBuilder()
    builder.process_all_videos()
    
    print("\n✓ Step 7 completed!")

def run_step_8():
    """Step 8: DAG Visualization"""
    print("=" * 60)
    print("STEP 8: DAG VISUALIZATION")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / "src" / "8_visualization"))
    from dag_visualizer import DAGVisualizer
    
    visualizer = DAGVisualizer()
    visualizer.process_all_videos()
    
    print("\n✓ Step 8 completed!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Code-Mixed Concept Extraction Pipeline"
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Run a specific step (1, 1.5, 2-8)"
    )
    parser.add_argument(
        "--from-step",
        type=str,
        help="Run from specific step onwards (1, 1.5, 2-8)"
    )
    
    args = parser.parse_args()
    
    # Map step names to functions
    all_steps = [
        ("1", run_step_1, "Transcript Extraction"),
        ("1.5", run_step_1_5, "Transcript Translation"),
        ("2", run_step_2, "Sentence Segmentation"),
        ("3", run_step_3, "Concept Extraction"),
        ("4", run_step_4, "Linguistic Normalization"),
        ("5", run_step_5, "Prerequisite Detection"),
        ("6", run_step_6, "Relation Classification"),
        ("7", run_step_7, "DAG Construction"),
        ("8", run_step_8, "Visualization")
    ]
    
    if args.step:
        # Run single step
        for step_id, step_func, step_name in all_steps:
            if step_id == args.step:
                step_func()
                return
        print(f"Invalid step: {args.step}")
    elif args.from_step:
        # Run from step onwards
        start_index = None
        for i, (step_id, _, _) in enumerate(all_steps):
            if step_id == args.from_step:
                start_index = i
                break
        
        if start_index is None:
            print(f"Invalid step: {args.from_step}")
            return
        
        for step_id, step_func, step_name in all_steps[start_index:]:
            try:
                step_func()
                print()
            except Exception as e:
                print(f"\n✗ Error in step {step_id}: {e}")
                import traceback
                traceback.print_exc()
                print("Pipeline stopped.")
                return
    else:
        # Run all steps
        print("\n" + "=" * 60)
        print("STARTING FULL PIPELINE")
        print("=" * 60 + "\n")
        
        for step_id, step_func, step_name in all_steps:
            try:
                step_func()
                print()
            except Exception as e:
                print(f"\n✗ Error in step {step_id}: {e}")
                import traceback
                traceback.print_exc()
                print("Pipeline stopped.")
                return
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

if __name__ == "__main__":
    main()
