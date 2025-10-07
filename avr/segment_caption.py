from typing import List

from test_env_utils import image_to_base64

async def generate_segment_caption(frames: List[str], llm_cfg) -> str:
    """Generate an English multi-frame grounded caption.

    Pipeline:
      1) Per-frame objective sentence (no speculation) so model attends to every frame.
      2) Synthesize all unique sentences into ONE paragraph that integrates ALL visual evidence.
    """
    if not frames:
        return ""

    # --- Stage 1: per-frame objective description (English) ---
    async def _describe_single_frame(img_path: str, idx: int, total: int) -> str:
        b64 = image_to_base64(img_path)
        if not b64:
            return ""
        prompt = (
            f"Frame {idx+1}/{total}: Provide ONE objective English sentence (<=25 words) describing ONLY visible content: main scene, salient objects, on-screen text, charts, logos. "
            "NO speculation, NO inferred causes, NO symbolism. Forbidden words: maybe, might, seems, appear, appears, appear to, suggests, symbolize, symbolizes, represent, represents, possibly."
        )
        try:
            resp = await llm_cfg.cheap_model_func(
                prompt=prompt,
                system_prompt=(
                    "Return ONLY one concise objective English sentence. Do NOT speculate."
                ),
                images_base64=[b64],
                max_new_tokens=80,
                temperature=0.10,
                top_p=0.9
            )
        except Exception as e:
            print(f"[Caption][WARN] frame describe fail {img_path}: {e}")
            resp = ""

        # Normalize response: support object with .choices, dicts, or plain string
        def _extract_text_from_resp(r):
            try:
                if r is None:
                    return ""
                if hasattr(r, 'choices'):
                    # e.g., OpenAI-like object
                    try:
                        return r.choices[0].message.content
                    except Exception:
                        return str(r)
                if isinstance(r, dict):
                    if 'return' in r:
                        return r['return'] or ""
                    if 'choices' in r and isinstance(r['choices'], list) and r['choices']:
                        c = r['choices'][0]
                        if isinstance(c, dict) and 'message' in c:
                            return c['message'].get('content','') or c.get('text','')
                        return c.get('text','') if isinstance(c, dict) else str(c)
                    if 'message' in r and isinstance(r['message'], dict):
                        return r['message'].get('content','')
                    # fallback
                    return str(r)
                # fallback to string
                return str(r)
            except Exception:
                return str(r)

        full = (_extract_text_from_resp(resp) or "").strip()
        # Safely get first non-empty line
        lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
        txt = lines[0] if lines else ""
        if txt.startswith('```'):
            txt = txt.strip('`')
        if (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'")):
            txt = txt[1:-1].strip()
        # Remove forbidden speculative words & trailing punctuation spaces
        forbidden = [
            "maybe", "might", "seems", "seem", "appears", "appear", "appear to",
            "suggests", "suggest", "symbolize", "symbolizes", "symbolise", "symbolises",
            "represent", "represents", "possibly", "perhaps"
        ]
        low = txt.lower()
        for w in forbidden:
            if w in low:
                # crude removal; could be improved by regex word boundaries
                txt = ' '.join([t for t in txt.split() if t.lower() != w])
                low = txt.lower()
        return txt.strip()

    frame_descs: List[str] = []
    for i, fp in enumerate(frames):
        try:
            d = await _describe_single_frame(fp, i, len(frames))
        except Exception as e:
            print(f"[Caption][ERR] single frame caption error: {e}")
            d = ""
        if d:
            frame_descs.append(d)
    # Deduplicate but preserve order
    seen = set(); uniq_descs = []
    for d in frame_descs:
        key = d.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq_descs.append(d)
    if not uniq_descs:
        return ""

    # --- Stage 2: Synthesis paragraph (English) ---
    # Build bullet evidence list
    joined = "\n".join(f"- {s}" for s in uniq_descs)
    # NOTE: Previous version BUG: the bullet list (joined) was never injected into the prompt, causing the model
    # to rely only on a static instruction block (with an example about Excel / charts / EV specs), leading to
    # hallucinated / cross-video contamination. We now explicitly include ONLY the extracted per-frame sentences
    # as the sole evidence, and we REMOVED the concrete example to avoid lexical anchoring.
    synth_prompt = f"""You are to synthesize ONE objective, information-dense English paragraph strictly grounded in the visual evidence.

EVIDENCE (each line is an objective sentence from a different frame; treat them as unordered but from the same short clip):
{joined}

INSTRUCTIONS:
1. Fuse all non-redundant facts into a coherent paragraph (no bullet list).
2. Preserve specific entities, numbers, visible text, UI element labels, and on‑screen data exactly as they appear.
3. Do NOT invent content not present in the evidence lines. If something (e.g. chart title, value) is NOT in the evidence, do not add it.
4. No meta phrases (e.g. "the image shows", "the frame", "the video").
5. Neutral factual tone; no speculation, symbolism, or causality beyond what is explicitly described.
6. If evidence is extremely sparse (<=1 line) just restate it cleanly (no padding / no hallucination).

OUTPUT: A single paragraph (no numbering, no quotes, no markdown)."""
    try:
        resp2 = await llm_cfg.cheap_model_func(
            prompt=synth_prompt,
            system_prompt="Produce ONE objective English paragraph. Do not speculate.",
            images_base64=None,
            max_new_tokens=400,
            temperature=0.2,
            top_p=0.9
        )
    except Exception as e:
        print(f"[Caption][ERR] synthesis failed: {e}")
        resp2 = ""
    # Normalize response similar to single-frame handling
    def _extract_text_from_resp_top(r):
        try:
            if r is None:
                return ""
            if hasattr(r, 'choices'):
                try:
                    return r.choices[0].message.content
                except Exception:
                    return str(r)
            if isinstance(r, dict):
                if 'return' in r:
                    return r['return'] or ""
                if 'choices' in r and isinstance(r['choices'], list) and r['choices']:
                    c = r['choices'][0]
                    if isinstance(c, dict) and 'message' in c:
                        return c['message'].get('content','') or c.get('text','')
                    return c.get('text','') if isinstance(c, dict) else str(c)
                if 'message' in r and isinstance(r['message'], dict):
                    return r['message'].get('content','')
                return str(r)
            return str(r)
        except Exception:
            return str(r)

    para = (_extract_text_from_resp_top(resp2) or "").strip()
    if para.startswith('```'):
        parts = [ln for ln in para.splitlines() if not ln.strip().startswith('```')]
        para = " ".join(parts).strip()
    if (para.startswith('"') and para.endswith('"')) or (para.startswith("'") and para.endswith("'")):
        para = para[1:-1].strip()
    # Remove speculative words again defensively
    for w in [
        "maybe", "might", "seems", "seem", "appears", "appear", "appear to",
        "suggests", "suggest", "symbolize", "symbolizes", "symbolise", "symbolises",
        "represent", "represents", "possibly", "perhaps"
    ]:
        if w in para.lower():
            tokens = []
            for t in para.split():
                if t.lower() != w:
                    tokens.append(t)
            para = " ".join(tokens)
    if len(para) < 20:  # fallback if model produced too little
        para = " ".join(uniq_descs)
    return para.strip()
