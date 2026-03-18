# Single-Hop Question-Answer Generation

### Objective

Generate a corresponding Q\&A pair for each key point. The goal is to build a practical knowledge base that enables users to understand important or complex data, specialized information.You will process one key point at a time. Each generated Q&A pair must be derivable from a single key point.

### 2. Question Generation
For each key point, first identify the single most important piece of information it contains (e.g., a specific number, a definition, a reason, a name, an outcome).
Form a Direct \& Natural Question: Create a question that a real user, seeking that core information, would genuinely ask. The question must be specific, clear, and easy to understand.
The questions include, but are not limited to, the following patterns:
What: To ask for definitions, names, or specific details.
Why: To ask for reasons or justifications.
How: To ask about processes or methods.
How much/many: To ask about specific quantities or data.
What was the result/outcome of...?: To ask about consequences.

- Specific objects or entities must be explicitly identified. When referring to specific entities (e.g., people, organizations, locations), their full names or affiliations must be specified as clearly as possible (e.g., use “U.S. Congress” instead of “Congress”).

- Avoid trivial or universal questions. The questions should be relevant.

### 3. QA Generation
3 core principles:
Retrievability: Does the question contain enough context to be found precisely?
Human-like quality: Do the questions make sense and resemble questions that a real person would ask?
Answer Quality: Is the answer accurate, comprehensive and helpful?

- Before generating a Q\&A pair, verify that the selected video keypoint provides unique, essential information that the text keypoint lacks, and which is critical for forming the answer. If the video key point is merely illustrative or decorative (e.g., "a person is talking," "an object is on a table") and does not add a factual cornerstone to the answer, you must discard this pairing and select a different set of keypoints.
- - A QA pair should be independent and understandable without any context.
- The Answer must directly and completely address the question, using the factual information contained within the source keypoint.
- Generate specific, closed-ended QA pairs.
- Avoid source attribution. Do not mention "the video", "the narrator", or "the keypoint". Treat the information as factual knowledge.
Prohibited phrases:  
    Camera type: “The close-up shows...,” “In the frame...”  
    Production type: “Video summary...,” “The narrator mentions...”  
    Explanatory type: “This indicates...”(Unless the key point contains explanatory information)
- Ensure the language is natural, as if a user is asking for help and an expert is providing a direct answer.
- Generate as many as question-answer pairs from the provided keypoints.
- The output must be a JSON file and formatted to match the example without any interpretation or code markup.

## Output Example
[
    {{
        "question": "What were the key interest rate targets set by the Bank of Japan during its policy meeting on July 1, 2025?",
        "answer": "The Bank of Japan's Policy Board decided to maintain the short-term interest rate target at -0.1% and set the 10-year government bond yield target to be around 0%.",
        "keypoints": [
            "On July 1, 2025, the Bank of Japan's Policy Board announced its decision to keep the short-term interest rate target at -0.1% and the 10-year government bond yield target around 0%."
        ]
    }},
    {{
        "question": "What were the main reasons cited by the Bank of Japan for continuing its ultra-loose monetary policy?",
        "answer": "The primary reasons cited for the policy were persistent uncertainties in the global economic outlook and sluggish domestic wage growth.",
        "keypoints": [
            "The Bank of Japan attributed its continuation of an ultra-loose monetary policy to uncertainties in the global economy and slow domestic wage growth."
        ]
    }},
    {{
        "question": "What specific revision did the Bank of Japan make to its inflation forecast for the 2025 fiscal year?",
        "answer": "The Bank of Japan slightly revised its inflation forecast for the 2025 fiscal year upwards, from 1.8\% to 1.9\%, acknowledging the impact of rising import costs.",
        "keypoints": [
            "The bank slightly revised its inflation forecast for the fiscal year 2025 upwards from 1.8\% to 1.9\%, acknowledging rising import costs."
        ]
    }}
]
### Test

Input:

video keypoints: {video_keypoints_str}
(text keypoints: {text_keypoints_str})