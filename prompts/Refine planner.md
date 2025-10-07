# refine planner
You are an expert video-content analyst. Evaluate the sufficiency and information density of retrieved evidence for question answering.

Question:
"{query}"

Evidence:
{context_str}

Example (output a single JSON object (no extra text/markdown) ):
{{
    "overall_answerability_score": <0-5 integer>,
    "information_density_score": <0-5 integer>,
    "refinement_targets": [
    {{
        "clip_id": "<id>",
        "reasoning": "<short reason>",
        "checks": {{
            "temporal_coherence_needed": <true|false>,
            "ocr_needed": <true|false>,
            "det_needed": <true|false>
        }}
    }}
],
    "numeric_evidence_required": <true|false>, 
        # if numeric_evidence_required true
        "numeric_focus_clip_id": <clip id or empty string>, 
    "temporal_sequence_incomplete": <true|false>,
        # if temporal_sequence_incomplete true
        "temporal_focus_clip_id": "<clip id or empty string>", 
}}
**Scoring rules:**
- overall_answerability_score (0-5): Confidence in answering the query based *only* on the current `context_str`. 0 = missing key facts or irrelevant; 5 = clear answer with complete details and relations.
- information_density_score (0-5): An assessment of the visual complexity in the clips. 5 = clips contain only a few generic objects (low density); 0 = clips contain many diverse objects (high density, details likely under-sampled).
- Populate `refinement_targets` if the total score (`overall_answerability_score` + `information_density_score`) is <= 5, indicating that the current information of the target clips(1~5) is insufficient.
**Numeric / Structured Evidence Flag (HIGH PRIORITY):**
- Set `numeric_evidence_required` to true if answering the query likely requires reading numerical values, units, ranges, coordinates, chart/graph axes or legends, percentages, counts, scores, dates/timestamps, or any other structured numeric/textual data from on-screen visuals (e.g., line graphs, tables, scoreboards, temperature curves).
- When true, down-stream logic of numeric_focus_clip_id will prefer refinement with denser frames and OCR regardless of the sufficiency scores.
**Temporal Sequence Incomplete (HIGH PRIORITY):**
- Set `temporal_sequence_incomplete` true if the question needs a multi-step ordered operation but current clips do not cover the full process.
- When true provide exactly one `temporal_focus_clip_id` (existing clip id) that partially shows the process. If false set it to an empty string.

STRICT OUTPUT RULES (MANDATORY):
1. Output ONLY one JSON object. No explanations, no markdown fences.
2. All keys and all string values MUST be enclosed in double quotes.
3. Do NOT escape underscore '_' (never produce \_).
4. If `temporal_sequence_incomplete` is true, `temporal_focus_clip_id` MUST be a valid existing clip id; if `numeric_evidence_required` is true, `numeric_focus_clip_id` MUST be a valid existing clip id; otherwise it MUST be an empty string.
5. Strictly follow the fields output in the example.
6. No extra fields.
7. Booleans in lowercase, integers for scores.
8. If no refinement needed, still output the JSON with an empty refinement_targets list.

**Guidance for Refinement Targets:**
- ocr_needed:
  Set to `true` when the answer to the query depends on reading and understanding symbolic, textual, or structured data presented visually in the clip, and this information is missing from evidence. This applies to cases like:
  - Reading specific text/numbers: e.g., needing to know player scores, product prices, on-screen instructions, or code.
  - Understanding user interface (UI) elements: e.g., identifying a clicked button, reading a menu option.
  - Extracting data from structured formats: e.g., getting values from tables, charts, or graphs.
- det_needed:
  Set to `true` when the answer to the query requires identifying, locating, or understanding specific physical objects, entities, their attributes, or spatial relationships, and the evidence is too generic or lacks this detail. This applies to cases like:
  - Specific identification/classification: e.g., `context_str` mentions "a car," but the query asks if it's "a red Tesla Model 3."
  - Determining object state or attributes: e.g., needing to know "if the traffic light is green" or "if the laptop is open."
  - Understanding spatial relationships and interactions: e.g., needing to know "what the person is holding" or "which object is to the left of the table."