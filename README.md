
## DEMO VIDEO LINK - https://drive.google.com/file/d/16AAOSZuTXAXWubsjm2ARGb3lQQsm5q6k/view?usp=drive_link 


# Code-Mixed Educational Video Concept & Prerequisite Pipeline

## 🎯 Project Goal

Model pedagogical dependencies from code-mixed educational lectures (Hindi–English and Telugu–English) by building directed prerequisite graphs over extracted concepts. The final goal is to support curriculum design and learner modeling by turning unstructured lecture videos into structured concept maps.

---

## 🧩 Architectural Choices

### 1. Two-level pipeline: explicit vs. hybrid implicit

The project separates prerequisite modeling into two complementary pipelines:

- **Method 1 – Explicit prerequisites (method1.py)**  
  Operates directly on the teacher’s explanations to detect *explicit* prerequisite mentions.
  - Steps 1–4: Transcript extraction → segmentation (via YouTube chunks) → 2-pass LLM concept extraction → (optional) normalization.
  - Step 1.5: Translation to English while preserving technical vocabulary.
  - Steps 5–8: Window-based prerequisite candidate detection, LLM verification, and DAG visualization.

- **Method 2 – Hybrid implicit prerequisites (method2.py)**  
  Starts from the same concepts and translated transcript, but uses additional video-level signals (co-occurrence, temporal proximity, explanation overlap) to propose *implicit* prerequisite edges that may never be stated in a single sentence.
  - Uses `hybrid_run_detector.py` to compute candidate implicit edges.
  - Applies per-method confidence thresholds and cycle-breaking to build a clean DAG.
  - Produces a filtered implicit prerequisite graph (JSON + PNG visualization).

This split makes the design easier to explain and evaluate:

- **Explicit pipeline** is close to how a human annotator would mark “A is needed for B” in the transcript.
- **Implicit pipeline** recovers deeper structure (e.g., concepts that always appear earlier and are used in explanations) even when the teacher doesn’t say “X is a prerequisite of Y”.

### 2. Config-driven per-video orchestration

Videos are described in `config/videos.yaml` with fields like:

- `id` (e.g., `video_1`, `video_2`, ...)
- `title`
- `domain` (CS, Physics, Chemistry, etc.)
- `language` (Hinglish, Telugu-English)
- `url` (YouTube link)

`method1.py` and the original `run_single_video.py` both read this config so the pipeline can be run on *any* configured video without code changes. This decouples the core algorithms from specific URLs or domains.

### 3. Transcript-first design with translation as a separate step

Because the lectures are code-mixed and partly non-English, the architecture is:

1. **Always work from the original-language transcript** (`*_transcript.json`).
2. **Translate to English later** (`*_transcript_en.json`) for tasks where English NLP/LLM tools are strongest (concept extraction, prerequisite prompts, etc.).

This preserves original structure and timing while allowing high-quality English reasoning in downstream LLM calls.

### 4. File-based modular stages

Each major stage writes its outputs to `data/` so that steps are reusable and easy to debug:

- Raw transcripts → `data/raw/<video_id>_transcript.json`
- Translated transcripts → `data/raw/<video_id>_transcript_en.json`
- Concepts → `data/processed/<video_id>_concepts.json`
- Explicit prerequisite candidates → `data/processed/<video_id>_prerequisite_candidates.json`
- Verified explicit prerequisites → `data/processed/<video_id>_prerequisites.json`
- Hybrid implicit prerequisites → `data/processed/<video_id>_implicit_prerequisites.json`
- Graph JSONs & PNGs → `data/output/` and `data/visualizations/`

This explicit on-disk structure makes the pipeline transparent for grading and experimentation: instructors can inspect intermediate JSON files to see what each stage did.

---

## � Output Structure & Rationale

The output is intentionally split into several levels to separate concerns:

1. **Concept inventory** – `data/processed/<video_id>_concepts.json`

   ```json
   {
     "video_id": "video_2",
     "total_concepts": 20,
     "concepts": [
       {"concept": "Heat Capacity", "llm_category": "core"},
       {"concept": "Specific Heat Capacity", "llm_category": "core"},
       {"concept": "Ideal Gas", "llm_category": "supporting"}
     ]
   }
   ```

   - **Reasoning:** This file can be reused by any future prerequisite model, or even a curriculum designer, without re-running LLM extraction.

2. **Explicit prerequisites (verified)** – `data/processed/<video_id>_prerequisites.json`

   ```json
   {
     "video_id": "video_5",
     "total_candidates": 17,
     "all_edges": [
       {
         "source": "Heat Capacity",
         "target": "First Law Of Thermodynamics",
         "llm_verified": true,
         "confidence": 0.9,
         "reason": "...short LLM justification..."
       }
     ]
   }
   ```

   - **Reasoning:** Separate the raw candidates from LLM-verified ones, and store justifications for interpretability.

3. **Hybrid implicit prerequisites (filtered)** – `data/output/<video_id>_implicit_filtered_dag.json`

   ```json
   {
     "video_id": "video_5",
     "method": "hybrid_implicit_filtered",
     "num_nodes": 20,
     "num_edges": 15,
     "nodes": [
       {"id": 0, "label": "Heat Capacity", "concept": "Heat Capacity", "llm_category": "core"}
     ],
     "edges": [
       {
         "source": 0,
         "target": 3,
         "prerequisite": "Heat Capacity",
         "target_concept": "Constant Pressure",
         "confidence": 0.88,
         "edge_type": "combined"
       }
     ]
   }
   ```

   - **Reasoning:** Node/edge indexing is convenient for front-ends or graph tools, while still preserving human-readable labels.

4. **Visualization artifacts** – `data/visualizations/<video_id>_dag.png`, `data/visualizations/<video_id>_implicit_filtered_dag.png`

   - **Reasoning:** Separate visual outputs from JSON so that the same data can be rendered in multiple styles if needed; PNGs are easiest to include in the written report.

---


## 🔄 Pipeline Overview

High-level flow (both methods share early stages):

```text
YouTube Video
   ↓
Transcript Extraction (YouTube API / manual)
   ↓
Concept Extraction (LLM, 2-pass)
   ↓
Translation to English (Groq LLM)
   ↓
Explicit Prerequisites (window-based + LLM verification)  →  Explicit DAG
   ↓
Hybrid Implicit Prerequisites (temporal/semantic/co-occurrence)  →  Implicit DAG
```

### Method 1 (explicit) – implemented in `method1.py`

1. **Step 1 – Data collection:**  
   `TranscriptExtractor` downloads or fetches the transcript from YouTube and saves it as JSON.

2. **Step 2 – Segmentation:**  
   If the YouTube transcript is already chunked, those segments are used directly; no extra sentence splitter is needed.

3. **Step 3 – Concept extraction (2-pass LLM):**  
   `LLMConceptExtractor`:
   - Pass 1: Extract raw concept candidates from each chunk.
   - Pass 2: Deduplicate, merge synonyms, and score concepts.

4. **Step 4 – Normalization:**  
   Skipped in the final version, because concepts were already enforced to be in “final form” during Pass 2.

5. **Step 1.5 – Translation to English:**  
   `TranscriptTranslator` (Groq LLM) converts the transcript to English while preserving math notation and technical words.

6. **Steps 5–8 – Explicit prerequisites and DAG:**  
   - `WindowBasedPrerequisiteDetector` scans the translated transcript for teaching patterns (e.g., “before we do B, we must understand A”).
   - `LLMPrerequisiteVerifier` re-checks each candidate edge with an LLM and assigns `llm_verified` + confidence.
   - `create_dag_graph.py` turns the verified edges into a directed acyclic graph and renders a PNG.

You can run this method for any video as:

```bash
GROQ_API_KEY="<your_key>" python method1.py video_5
```

### Method 2 (hybrid implicit) – implemented in `method2.py`

1. **Prerequisites:**  
   Requires that `method1.py` has already produced concepts and translated transcript.

2. **Hybrid detector:**  
   `hybrid_run_detector.py` uses multiple signals:
   - Semantic similarity between concept definitions.
   - Temporal ordering in the lecture.
   - Co-occurrence patterns and explanation overlaps.

3. **Filtering & cycle removal:**  
   `method2.py` applies the same thresholds as `filter_and_visualize_video2_implicit.py` and removes weakest edges from any cycles to force a DAG.

4. **Visualization & JSON:**  
   - Reuses `filter_and_visualize_video2_implicit.py` to draw a layered left→right graph with edge colors per method.  
   - Saves a machine-readable DAG JSON for downstream analysis.

Run this method for any video as:

```bash
python method2.py video_5
```

---

## 🚀 Setup & How to Run

### Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

Set your Groq API key (required for concept extraction + translation in method1):

```bash
export GROQ_API_KEY="<your_groq_key>"
```

### Running the pipelines

1. **Run explicit pipeline (method 1) for a video**

```bash
GROQ_API_KEY="$GROQ_API_KEY" python method1.py video_2
```

2. **Run hybrid implicit pipeline (method 2) for the same video**

```bash
python method2.py video_2
```

Outputs will appear under `data/raw/`, `data/processed/`, `data/output/`, and `data/visualizations/` as described above.

---

## 🛠️ Core Technologies

- **Transcript Extraction:** `youtube-transcript-api`, `yt-dlp`, manual transcript for Telugu where needed.
- **Translation & Concept LLMs:** Groq (`llama-3.3-70b-versatile`).
- **NLP / Utilities:** spaCy, sentence transformers, KeyBERT, scikit-learn.
- **Graphs & Visualization:** NetworkX, Matplotlib.

---

## 👤 Author

Abhinav Chatrathi

