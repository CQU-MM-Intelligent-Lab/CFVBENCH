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

## License & Terms of Use

The CFVBench dataset is distributed with a strict distinction between the annotations provided by the authors and the raw video content collected from public sources.

1\. Annotations License

The annotations, question-answer pairs, and structured metadata created by the authors are licensed under the **[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**.

2\. Video Data Usage

The raw video files are collected from public sources (YouTube). The authors **do not** own the copyright to these videos. Access to this data is provided strictly for **non-commercial academic research** under the legal principles of **Fair Use** (e.g., for technical algorithm training and visual analysis).

By downloading or using this dataset, you acknowledge and agree to the following terms:

  * **Academic Use Only:** This dataset is strictly intended for research and educational purposes.
  * **Non-Commercial:** Any commercial use, redistribution for profit, or utilization beyond the scope of research is strictly prohibited.
  * **No Redistribution:** You are prohibited from re-hosting or redistributing the raw video files publicly.
  * **Liability:** The user assumes full responsibility for any consequences arising from the use of this dataset. The authors and their affiliated institutions bear no liability for misuse.

> **Disclaimer:** The content, views, and opinions expressed within the raw videos belong solely to the original creators and do not reflect the views, positions, or ideologies of the CFVBench authors.

### :warning: Notice and Takedown Policy

We are committed to respecting intellectual property rights. **If you are a copyright holder** and believe your content has been included in a way that violates your rights, please contact us directly via email or by raising a GitHub Issue. Upon receipt of a valid request, we will **immediately remove** the specific content from the dataset.

-----

## Dataset

You can find our dataset in the `Bench/` directory.

## Quick Start

To set up the environment, you can use the provided `environment.yml` file.

1.  Create the conda environment:

    ```bash
    conda env create -f environment.yml
    conda activate cfv
    pip install httpx[socks]
    conda install -c conda-forge libiconv ffmpeg cudnn=8 -y
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    ```

2.  **Download Dataset**:

    > **Note:** By downloading the file below, you agree to the Terms of Use and License outlined above.

    Download the dataset zip file from the following link:
    [CFVBench Video Dataset](https://drive.google.com/file/d/1UVc0MQaCT1YS89VlPMcL9fY7av-9Di8D/view?usp=sharing)

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

3. Model Setup
   Open `videorag/_config.py` to review the required model paths and download targets, then update the configuration to match your local environment.

4. Run Tests
- Run the full benchmark, preprocess all videos, and generate retrieval-based answers for all questions:
  `python avr/avr.py`
- Run a custom question on a single video and preprocess only that video:
  `python avr/avr.py --question "YOUR QUESTION" --video_path "Bench/VIDEO_ID.mp4"`
  example: 
  `python avr/avr.py --question "What is the main topic of the video?" --video_path "Bench/_Dsu07-VKRw.mp4"`
- Run the benchmark for a single video and preprocess only that video:
  `python avr/avr.py --video_path "Bench/VIDEO_ID.mp4"`
  example: 
  `python avr/avr.py --video_path "Bench/_Dsu07-VKRw.mp4"`
  This will evaluate all questions under `Bench/_Dsu07-VKRw`.
