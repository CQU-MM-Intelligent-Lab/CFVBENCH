# CFVBench: A Comprehensive Video Benchmark for Fine-grained Multimodal Retrieval-Augmented Generation

## 📖 Overview

CFVBENCH is a a large-scale, manually verified benchmark constructed from 599 publicly available videos, yielding 5,360 open-ended QA pairs. CFVBench spans high-density domains such as chart-heavy reports, news broadcasts, and software tutorials, requiring models to retrieve and reason over long temporal video spans while maintaining fine-grained multimodal information.

Adaptive Visual Refinement (AVR) is a plug-and-play framework that adaptively increases frame sampling density and selectively invokes external tools when necessary. 

## 🗂️ Repository Structure

```
CFVBENCH/
├── avr/                          # Automatic Video RAG pipeline
│   ├── test.py                   # Main inference script
│   ├── api_config.py             # API configuration for LLM providers
│   ├── question_processing.py    # Question generation pipeline
│   └── ...
├── videorag/                     # Video RAG core modules
│   ├── videorag.py               # Main VideoRAG implementation
│   ├── evaluate/                 # Evaluation toolkit
│   │   ├── evaluate.py           # Multi-metric evaluation
│   │   └── keypoint_match.py     # Keypoint matching utilities
│   └── _videoutil/               # Video processing utilities
│       ├── asr.py                # Automatic Speech Recognition
│       ├── caption.py            # Frame captioning
│       └── split.py              # Video segmentation
├── Bench/                        # Dataset directory
│   └── a9vPm615xnY/              # Example video with QA pairs
│       ├── a9vPm615xnY.mp4
│       └── multiQA.json
└── prompts/                      # Prompt templates
    ├── Caption Generation.md                                           # Frame caption synthesis
    ├── DET word.md                                                     # Word detection
    ├── Evaluation.md                                                   # Evaluation prompts
    ├── Final Generation.md                                             # Final answer generation
    ├── News Video keypoint extraction Protocol.md                      # News video keypoint extraction
    ├── News-type and Structured-Data-type Multi-Hop Question-Answer Generation Protocol.md
    ├── QA Pair Filter.md                                               # QA pair filtering
    ├── Refine planner.md                                               # Refinement planning
    ├── Single-Hop Question-Answer Generation Protocol.md               # Single-hop QA generation
    ├── Structured-Data Video KeyPoint Extraction Protocol.md           # Structured data extraction
    ├── Text KeyPoint Extraction Protocol.md                            # Text keypoint extraction
    ├── Tutorial-type Multi-Hop Question-Answer Generation Protocol.md  # Tutorial multi-hop QA
    ├── Tutorial-type Video Keypoints Extraction Protocol.md            # Tutorial keypoint extraction
    └── Video Timestamp Extraction Protocol.md                          # Timestamp extraction
```

## 📊 Dataset

You can find our dataset examples in the `Bench/` directory. The complete dataset will be released upon paper acceptance.