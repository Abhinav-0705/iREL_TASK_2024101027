"""
Main Pipeline Runner - Execute the full pipeline
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
# Import using importlib to handle directories with special names
import importlib

# Import modules dynamically
transcript_module = importlib.import_module('1_data_collection.transcript_extractor', 'src')
TranscriptExtractor = transcript_module.TranscriptExtractor

segmentation_module = importlib.import_module('2_segmentation.sentence_segmenter', 'src')
SentenceSegmenter = segmentation_module.SentenceSegmenter

concept_module = importlib.import_module('3_concept_extraction.concept_extractor', 'src')
ConceptExtractor = concept_module.ConceptExtractor

llm_concept_module = importlib.import_module('3_concept_extraction.llm_concept_extractor', 'src')
LLMConceptExtractor = llm_concept_module.LLMConceptExtractor

normalization_module = importlib.import_module('4_normalization.linguistic_normalizer', 'src')
LinguisticNormalizer = normalization_module.LinguisticNormalizer

prerequisite_module = importlib.import_module('5_prerequisite_detection.prerequisite_detector', 'src')
PrerequisiteDetector = prerequisite_module.PrerequisiteDetector

relation_module = importlib.import_module('6_relation_classification.relation_classifier', 'src')
RelationClassifier = relation_module.RelationClassifier

hypergraph_module = importlib.import_module('7_hypergraph_builder.hypergraph_builder', 'src')
HypergraphBuilder = hypergraph_module.HypergraphBuilder

visualization_module = importlib.import_module('8_visualization.hypergraph_visualizer', 'src')
HypergraphVisualizer = visualization_module.HypergraphVisualizer


class Pipeline:
    """Main pipeline for code-mixed concept extraction"""
    
    def __init__(self, config_dir: str = "config"):
        self.logger = setup_logger("pipeline")
        self.config = ConfigLoader(config_dir)
        
        # Load configurations
        self.settings = self.config.load_settings()
        self.videos = self.config.load_videos()
        
        self.logger.info("Pipeline initialized")
    
    def run_step_1_data_collection(self):
        """Step 1: Extract transcripts from videos"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: DATA COLLECTION")
        self.logger.info("=" * 60)
        
        extractor = TranscriptExtractor()
        extractor.process_videos(
            self.videos['videos'],
            prefer_whisper=False
        )
        
        self.logger.info("Step 1 completed!")
    
    def run_step_2_segmentation(self):
        """Step 2: Segment transcripts into sentences"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: SENTENCE SEGMENTATION")
        self.logger.info("=" * 60)
        
        segmenter = SentenceSegmenter()
        segmenter.process_all_transcripts()
        
        self.logger.info("Step 2 completed!")
    
    def run_step_3_concept_extraction(self):
        """Step 3: Extract concepts from sentences (LLM or Classical)"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: CONCEPT EXTRACTION")
        self.logger.info("=" * 60)
        
        # Check config for extraction method
        ce_config = self.settings.get('concept_extraction', {})
        method = ce_config.get('method', 'classical')
        
        if method == 'llm':
            self.logger.info("Using LLM-based concept extraction (Pipeline 3)")
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
            self.logger.info("Using classical concept extraction (KeyBERT/TF-IDF/NER)")
            extractor = ConceptExtractor()
        
        extractor.process_all_segmented(self.videos['videos'])
        
        self.logger.info("Step 3 completed!")
    
    def run_step_4_normalization(self):
        """Step 4: Normalize code-mixed concepts"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: LINGUISTIC NORMALIZATION")
        self.logger.info("=" * 60)
        
        normalizer = LinguisticNormalizer()
        normalizer.process_all_concepts(self.videos['videos'])
        
        self.logger.info("Step 4 completed!")
    
    def run_step_5_prerequisite_detection(self):
        """Step 5: Detect prerequisite relationships"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: PREREQUISITE DETECTION")
        self.logger.info("=" * 60)
        
        detector = PrerequisiteDetector()
        detector.process_all_normalized()
        
        self.logger.info("Step 5 completed!")
    
    def run_step_6_relation_classification(self):
        """Step 6: Classify relation types"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 6: RELATION CLASSIFICATION")
        self.logger.info("=" * 60)
        
        classifier = RelationClassifier()
        classifier.process_all_prerequisites()
        
        self.logger.info("Step 6 completed!")
    
    def run_step_7_hypergraph_builder(self):
        """Step 7: Build concept hypergraphs"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 7: HYPERGRAPH CONSTRUCTION")
        self.logger.info("=" * 60)
        
        builder = HypergraphBuilder()
        builder.process_all_hypergraphs()
        
        self.logger.info("Step 7 completed!")
    
    def run_step_8_visualization(self):
        """Step 8: Visualize hypergraphs"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 8: VISUALIZATION")
        self.logger.info("=" * 60)
        
        visualizer = HypergraphVisualizer()
        visualizer.visualize_all()
        
        self.logger.info("Step 8 completed!")
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STARTING FULL PIPELINE")
        self.logger.info("=" * 60 + "\n")
        
        try:
            self.run_step_1_data_collection()
            self.run_step_2_segmentation()
            self.run_step_3_concept_extraction()
            self.run_step_4_normalization()
            self.run_step_5_prerequisite_detection()
            self.run_step_6_relation_classification()
            self.run_step_7_hypergraph_builder()
            self.run_step_8_visualization()
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def run_from_step(self, step: int):
        """Run pipeline from a specific step"""
        steps = {
            1: self.run_step_1_data_collection,
            2: self.run_step_2_segmentation,
            3: self.run_step_3_concept_extraction,
            4: self.run_step_4_normalization,
            5: self.run_step_5_prerequisite_detection,
            6: self.run_step_6_relation_classification,
            7: self.run_step_7_hypergraph_builder,
            8: self.run_step_8_visualization
        }
        
        for s in range(step, 9):
            if s in steps:
                steps[s]()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Code-Mixed Educational Video Concept Extraction Pipeline"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Run a specific step (1-8)"
    )
    parser.add_argument(
        "--from-step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Run pipeline from a specific step"
    )
    
    args = parser.parse_args()
    
    pipeline = Pipeline()
    
    if args.step:
        # Run single step
        steps = {
            1: pipeline.run_step_1_data_collection,
            2: pipeline.run_step_2_segmentation,
            3: pipeline.run_step_3_concept_extraction,
            4: pipeline.run_step_4_normalization,
            5: pipeline.run_step_5_prerequisite_detection,
            6: pipeline.run_step_6_relation_classification,
            7: pipeline.run_step_7_hypergraph_builder,
            8: pipeline.run_step_8_visualization
        }
        steps[args.step]()
        
    elif args.from_step:
        # Run from specific step
        pipeline.run_from_step(args.from_step)
        
    else:
        # Run full pipeline
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
