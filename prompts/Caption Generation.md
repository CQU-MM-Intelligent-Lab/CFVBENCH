# Caption Generation
You are to synthesize ONE objective, information-dense English paragraph strictly grounded in the visual evidence.

EVIDENCE:
{joined}

INSTRUCTIONS:
1. Fuse all non-redundant facts into a coherent paragraph (no bullet list).
2. Preserve specific entities, numbers, visible text, UI element labels, and on‑screen data exactly as they appear.
3. Do NOT invent content not present in the evidence lines. If something (e.g. chart title, value) is NOT in the evidence, do not add it.
4. No meta phrases (e.g. "the image shows", "the frame", "the video").
5. Neutral factual tone; no speculation, symbolism, or causality beyond what is explicitly described.
6. If evidence is extremely sparse (<=1 line) just restate it cleanly (no padding / no hallucination).

OUTPUT: A single paragraph (no numbering, no quotes, no markdown).