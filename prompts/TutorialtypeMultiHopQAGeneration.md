# Tutorial-type Multi-Hop Question-Answer Generation

### Objective

To generate high-quality question-answer pairs by combining multiple, sequential keypoints (must be 2~4). The goal is to build a practical knowledge base that allows users to understand complete workflows within a specific software, tool, or website.

### Methodology
The process is: Sequential Keypoints Selection --> Question Generation --> Answer Generation

### 1. Keypoint Analysis \& Question-Type Generation

Based on the information within that keypoint, generate questions that fall into one or more of the following **tutorial-specific categories**:

- "How-To" / Procedural Questions: Focus on how a specific goal is achieved.
    - Pattern: "How do I [achieve a goal]?", "What are the steps to [perform an action]?"
- "What-If" / Cause-and-Effect Questions: Focus on the result or outcome of a specific action.
    - Pattern: "What happens after I [perform an action]?", "What is the result of clicking [a specific button]?"
- "Where-Is" / Locational Questions: Focus on locating a specific UI element.
    - Pattern: "Where can I find the [UI element name] button?", "In which menu is the [option name] located?"
- "Parameter/Setting" Questions: Focus on specific values, text inputs, or settings used in a step.
    - Pattern: "What value should be entered into the [field name]?", "What is the recommended setting for [a specific option]?"
- "Why" / Purpose Questions: If the keypoint implies a purpose, ask about it.
    - Pattern: "Why is it necessary to [perform an action]?"

- The Question MUST specify the software/tool/website. Every question must start by identifying the application to ensure it is contextually grounded and retrievable.
Format: `In [Software/Tool/Website Name], ...`

-  The question needs all the selected key points for a complete answer, and none of them is missing. It should ask about the entire sequence, the relationship between the steps, or in is the result of performing multiple steps.
Example:* A question might ask for the step *after* an action described in Keypoint 1, which is answered by Keypoint 2.

- The Question should ask about a specific part of the procedure** described in the keypoint (e.g., the goal, a specific step, a setting, or the result).

- Avoid trivial or universal questions. Do not generate questions about common knowledge computer operations that are not specific to the application's workflow. The questions should address a meaningful part of the process shown.
AVOID: "In Word, how do you copy text?", "In Photoshop, how do you click the 'File' menu?"

### 2. Construction Parameters
3 core principles:
Retrievability: Does the question contain enough context to be found precisely?
Human-like quality: Do the questions make sense and resemble questions that a real person would ask?
Answer Quality: Is the answer accurate, comprehensive and helpful?

- - A QA pair should be independent and understandable without any context.
- The Answer must directly and completely address the question, using the factual information contained within the source keypoint.
- Generate specific, closed-ended QA pairs.
- Avoid source attribution. Do not mention "the video", "the narrator", or "the keypoint". Treat the information as factual knowledge.
    - Disable phrases: "In the video...", "The narrator explains...", "This suggests..."
- Ensure the language is natural, as if a user is asking for help and an expert is providing a direct answer.
- The demonstration and instructional operations shown in the video, such as entering “testblog” and “example text,” are only examples and should not be considered mandatory operations for viewers to make the answers more generic and instructive.
- Generate as many as question-answer pairs from the provided keypoints.
- The output must be a JSON file and formatted to match the example without any interpretation or code markup.

## Output Example
[
    {{
        "question": "In WIX Photo Studio, what is the first step to begin the automatic cutout process for an image?",
        "answer": "The first step is to click the 'Auto Cutout' button in the main toolbar, which initiates the automatic background removal process.",
        "keypoints": [
            "The user clicks the 'Auto Cutout' button in the main toolbar to begin the automatic cutout process."
        ]
    }},
    {{
        "question": "In WIX Photo Studio, after auto cutout completes, what tools can be used to refine the edges and restore incorrectly removed areas?",
        "answer": "After auto cutout completes, you can adjust the hardness and opacity sliders in the 'Refine Tools' section to modify edge properties, and use the 'Restore' tool to paint over areas that were incorrectly removed during the process.",
        "keypoints": [
            "After the auto cutout completes, the user adjusts the hardness and opacity sliders within the 'Refine Tools' section to refine the edges of the image.",
            "The user then uses the 'Restore' tool to paint over specific areas of the image, restoring parts that were incorrectly removed during the auto cutout process."
        ]
    }},
    {{
        "question": "In WIX Photo Studio, how do you finalize and apply all the cutout refinements you've made?",
        "answer": "To finalize the refinements, click the 'Apply' button at the bottom of the 'Refine Tools' panel, which confirms and applies all the adjustments made to the cutout.",
        "keypoints": [
            "Finally, the user applies the changes by clicking the 'Apply' button at the bottom of the 'Refine Tools' panel."
        ]
    }}
]

### Test

Input:

Keypoints: {video_keypoints_str}