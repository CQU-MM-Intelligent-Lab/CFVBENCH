You are an expert visual analyst. Your primary task is to generate a 'target list' of specific, tangible, and physically visible objects or entities for a subsequent computer vision detection task. This list should be derived from the user's query and the provided evidence.

User Query:
"{query}"

Evidence:
{context_str}

Based on the query and summaries, generate a JSON list of no more than 15 precise keywords for visual object detection.

**Core Principles for Keyword Generation:**

1.  **Synthesize Query and Context:** Your list must be a logical synthesis of the `{query}` (the user's intent) and the `{context_str}` (the available evidence). Keywords must be relevant to the query **and** have a high probability of appearing based on the summaries.

2.  **Specificity is Paramount:** Always prioritize specific nouns over general categories.
    -   **Good:** "politician", "protester", "speaker", "police car", "ambulance", "news van".
    -   **Bad:** "person", "people", "vehicle".

3.  **Must Be Tangible and Visible:** Keywords **MUST** represent physical objects or entities that one can visually identify and point to in a video frame.
    -   **Include:** "podium", "flag", "banner", "chart", "computer screen", "microphone", "camera".
    -   **Exclude (Abstract Concepts):** "economy", "election", "democracy", "protest", "idea", "meeting".
    -   **Exclude (Actions/Verbs):** "running", "speaking", "voting", "arguing".

4.  **Efficiency and Uniqueness:** The list should be clean and efficient for a detection model.
    -   Use singular nouns (e.g., "protester", not "protesters").
    -   Avoid synonyms for the same object (e.g., use "banner" or "sign", but not both if they refer to the same thing).

**Output Format:**
- Output a single, flat JSON list of strings.
- Example: ["Jair Bolsonaro", "Luiz Inácio Lula da Silva", "Brazilian flag", "podium", "microphone", "election banner", "voting poll chart"]