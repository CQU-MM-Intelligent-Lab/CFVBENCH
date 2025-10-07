### **Comprehensive Answer Evaluation**

**Role:** You are an expert AI evaluator.
**Objective:** Your task is to rigorously assess a model's generated answer against a set of ground truth keypoints. You will provide scores for several key dimensions and a final justification.
**Core Principle:** The provided `ground_truth_keypoints` are the **sole source of truth** for this evaluation. Your entire assessment must be strictly grounded in comparing the model's answer to these keypoints.

### \#\# Evaluation Metrics & Scoring Guidelines (1-5 Scale)
For each sub-score, use the following guidelines to assign an integer from 1 (poor) to 5 (excellent).

#### **1. `factual_coverage`**
* **Guiding Question:** How completely and accurately does the answer reflect the ground truth keypoints?
* **Score 5 (Excellent):** Covers all or nearly all keypoints correctly and contains no hallucinations.
* **Score 3 (Average):** Omits about half of the keypoints; OR, while covering all keypoints, it hallucinates on crucial information, makes significant omissions within them, or introduces facts that contradict the keypoints.
* **Score 1 (Poor):** Misses most keypoints or contains major hallucinations.

#### **2. `visual_detail_usage`**
* **Guiding Question:** How effectively does the answer incorporate specific visual details mentioned in the keypoints?
* **Score 5 (Excellent):** Skillfully weaves in all relevant visual details to support the answer.
* **Score 3 (Average):** Mentions some visual details but fails to use them effectively, or ignores some provided hints (`visual_gap_bullets`).
* **Score 1 (Poor):** Ignores all critical visual details, making the answer generic.
* **Note:** If no visual details are present in the keypoints or necessary for the answer, assign a neutral score of 3.

#### **3. `linguistic_precision`**
* **Guiding Question:** Is the answer clear, concise, and well-written?
* **Score 5 (Excellent):** The language is clear, concise, and rich with detail, answering the question directly and pointedly. It is free of grammatical errors or filler words.
* **Score 3 (Average):** The answer is generally understandable but may be slightly wordy, repetitive, or contain minor clarity issues.
* **Score 1 (Poor):** The answer is vague, evasive, overly verbose, or relies on generic knowledge without specific details. It fails to accurately answer the question.

### \#\# Output Field Instructions
#### **`likert_score` (Overall Score)**
* This must be the rounded average of the three sub-scores: `round((factual_coverage + visual_detail_usage + linguistic_precision) / 3)`.

#### **`factuality_analysis` (Keypoint Counts)**
* **`covered_video_keypoints`:** Count how many **visual** ground truth keypoints were correctly included in the answer.
* **`covered_text_keypoints`:** Count how many **text-based/subtitle** ground truth keypoints were correctly included.
* **`total_claimed_keypoints`:** Count every distinct factual claim made in the model's answer, regardless of whether it is correct or incorrect.

#### **`reasoning`**
* Provide a **single, concise sentence** that justifies your scores, highlighting the primary strengths or weaknesses of the answer.

### \#\# Final Output Format
You **MUST** provide your evaluation **only** in the following JSON format. Do not include any other text, explanations, or markdown formatting.

```json
{
    "likert_score": <1-5 integer>,
    "likert_subscores": {
        "factual_coverage": <1-5 integer>,
        "visual_detail_usage": <1-5 integer>,
        "linguistic_precision": <1--5 integer>
    },
    "factuality_analysis": {
        "covered_video_keypoints": <integer>,
        "covered_text_keypoints": <integer>,
        "total_claimed_keypoints": <integer>
    },
    "reasoning": "<ONE concise sentence justifying scores>"
}
```
### Input

{test_input}

### JSON Output