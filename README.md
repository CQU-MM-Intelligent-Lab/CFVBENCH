# CFVBench: A Comprehensive Video Benchmark for Fine-grained Multimodal Retrieval-Augmented Generation

## Overview

CFVBENCH is a a large-scale, manually verified benchmark constructed from 599 publicly available videos, yielding 5,363 open-ended QA pairs. CFVBench spans high-density domains such as chart-heavy reports, news broadcasts, and software tutorials, requiring models to retrieve and reason over long temporal video spans while maintaining fine-grained multimodal information.

Adaptive Visual Refinement (AVR) is a plug-and-play framework that adaptively increases frame sampling density and selectively invokes external tools when necessary. 

## Repository Structure

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

## License
This dataset is released under the [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

:warning: Terms of Use
By accessing or using this dataset, you acknowledge and agree to the following terms:

This dataset is strictly intended for academic and research purposes. Any commercial use, redistribution for profit, or utilization beyond the scope of research is strictly prohibited. The user assumes full responsibility for any consequences, legal or otherwise, arising from the use, dissemination, or modification of this dataset. The authors and their affiliated institutions bear no liability for misuse.

The raw video data contained in this dataset is collected from public sources. The content, views, and opinions expressed within these videos (including but not limited to political, military, religious, or social commentary) belong solely to the original creators and do not reflect the views, positions, or ideologies of the dataset authors. The inclusion of any specific video is solely for the purpose of technical research (e.g., algorithm training, visual analysis) and does not imply endorsement or agreement with the content by the dataset authors.

We do not claim ownership of the copyright for the raw video files. Access to this data is provided to researchers under the principles of fair use for academic study. 
If you are the copyright holder of any work included in this dataset and believe that the removal of specific content is warranted, please raise an issue or contact us directly. We are committed to respecting intellectual property rights and will address valid takedown requests promptly.

> **Note:** Please read and understand the license and disclaimers outlined above thoroughly. If you do not agree to these terms, you must refrain from downloading or using this dataset.

## Dataset

You can find our dataset in the `Bench/` directory.

## Quick Start

To set up the environment, you can use the provided `environment.yml` file.

1.  Create the conda environment:
    ```bash
    conda env create -f environment.yml
    ```

2.  Activate the environment:
    ```bash
    conda activate cfv
    ```

3.  **Download Dataset**:
    Download the dataset zip file from the following link:
    [CFVBench Dataset](https://drive.google.com/file/d/1UVc0MQaCT1YS89VlPMcL9fY7av-9Di8D/view?usp=sharing)
    
    After downloading, unzip the contents into the `Bench/` directory. The structure should look like this:
    ```
    Bench/
    ├── video_id_1.mp4
    ├── video_id_2.mp4
    ├── ...
    ├── video_id_1/
    │   └── multiQA.json
    ├── video_id_2/
    │   └── multiQA.json
    └── ...
    ```