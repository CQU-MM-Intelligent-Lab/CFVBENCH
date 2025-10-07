# CFVBENCH: Complex Fact-Centric Video Question Answering Benchmark

## 📖 Overview

CFVBENCH is a comprehensive benchmark designed to evaluate Large Vision-Language Models (LVLMs) on complex, fact-centric video question answering tasks. The benchmark focuses on:

- **Multi-hop reasoning** across temporal video segments
- **Fact-based questions** requiring precise information extraction
- **Diverse video types**: News, Tutorial, and Structured-Data videos
- **Automated evaluation** with multi-metric assessment

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
└── prompts/                      # Prompt templates
    ├── Caption Generation.md
    ├── QA Pair Filter.md
    └── ...
```

## 📊 Benchmark Features

### Question Types

- **Single-hop**: Direct factual questions
- **Multi-hop**: Questions requiring reasoning across multiple segments
- **Type-specific**: Tailored for News, Tutorial, and Structured-Data videos

### Evaluation Metrics

- **BERTScore**: Semantic similarity measurement
- **LLM-based**: GPT/Claude-powered evaluation
- **Exact Match**: Strict string matching
- **F1 Score**: Token overlap measurement

### Video Processing Pipeline

1. **ASR**: Automatic speech recognition using Faster-Whisper
2. **Frame Captioning**: Dense frame-level visual descriptions
3. **Segmentation**: Intelligent video segment chunking
4. **Retrieval**: Vector-based segment retrieval for QA

### Evaluation Framework

Multi-metric evaluation supporting:
- Automated scoring with multiple metrics
- Keypoint-based factual accuracy assessment
- Comprehensive result analysis