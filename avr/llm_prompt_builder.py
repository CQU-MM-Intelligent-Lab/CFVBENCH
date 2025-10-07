import json
import os
import glob

from test_env_utils import extract_final_answer

async def build_and_call_llm(query, all_segment_data, llm_cfg, base_mode: bool = False):
    print("\n--- [Step 3] Constructing multimodal prompt ---")
    llm_input_chunks = []
    all_images_b64 = []
    image_counter = 0
    images_disabled = True
    # If running in base_mode, send compressed images (the frames extracted earlier are already compressed)
    if base_mode:
        images_disabled = False
    # --- 新增: token 统计辅助 ---
    def _approx_token_count(text: str) -> int:
        try:
            import tiktoken  # 若存在更精确  # type: ignore
            enc = tiktoken.get_encoding('cl100k_base')
            return len(enc.encode(text))
        except Exception:
            # 简单启发: 按空格分 + 标点近似
            if not text:
                return 0
            return max(1, len(text.strip().split()))

    total_tokens_text = 0

    # 按需关闭摘要：最终输入仅使用重采样帧 + 原始字幕 + OCR/DET
    use_diff_flag = False
    per_clip_logs = []

    # 保留上游已经排好（含邻居 prev-base-next）顺序，不再按 id 重新排序
    # caption 统计
    _cap_total_chars = 0
    _cap_non_empty = 0
    _cap_clips = 0

    for data in all_segment_data:
        clip_id = data['id']
        chunk_lines = [f"[Chunk: {clip_id}]"]
        # 新增：段级 caption 先于字幕，便于模型形成视觉先验
        seg_cap = (data.get('segment_caption') or '').strip()
        if seg_cap:
            chunk_lines.append(f"Caption: \"{seg_cap}\"")
        # caption 统计累积
        _cap_clips += 1
        if seg_cap:
            _cap_non_empty += 1
            _cap_total_chars += len(seg_cap)
        # 1. 视觉差异摘要 (若可用)
        diff_text = data.get('diff_summary') if use_diff_flag else None
        diff_text = (diff_text or '').strip()
        # 2. OCR / DET
        ocr_text = (data.get('ocr_text') or '').strip()
        det_text = (data.get('det_text') or '').strip()
        # 3. 原字幕（始终使用完整字幕，不截断）
        raw_sub = (data.get('transcript') or '').strip()
        subtitle_snippet = raw_sub

        # 组装文本块（顺序: caption -> diff -> OCR -> DET -> Subtitles）
        if diff_text:
            chunk_lines.append(f"RefinedVisualSummary: \"{diff_text}\"")
        if ocr_text:
            chunk_lines.append(f"On-Screen Text: \"{ocr_text}\"")
        if det_text:
            chunk_lines.append(f"Detected Objects: \"{det_text}\"")
        chunk_lines.append(f"Subtitles: \"{subtitle_snippet}\"")

        # 统计 token
        diff_tokens = _approx_token_count(diff_text) if diff_text else 0
        ocr_tokens = _approx_token_count(ocr_text) if ocr_text else 0
        det_tokens = _approx_token_count(det_text) if det_text else 0
        sub_tokens = _approx_token_count(subtitle_snippet)
        clip_total = diff_tokens + ocr_tokens + det_tokens + sub_tokens
        total_tokens_text += clip_total
        per_clip_logs.append(
            f"[PromptBuild][{clip_id}] diff={diff_tokens} ocr={ocr_tokens} det={det_tokens} sub={sub_tokens} total={clip_total}"
        )

        # Keyframes (保持原逻辑)
        frames_with_ts = data.get("frames_with_ts", [])
        if frames_with_ts:
            # In base_mode we will attach images; otherwise only report counts
            chunk_lines.append(f"Keyframes: {len(frames_with_ts)} frames ({'images attached' if base_mode else 'images disabled'})")
        else:
            chunk_lines.append(f"Keyframes: 0 frames ({'images attached' if base_mode else 'images disabled'})")

        llm_input_chunks.append("\n".join(chunk_lines))

    # 输出 token 总览日志
    print("[PromptBuild] Per-clip token usage (approx):")
    for line in per_clip_logs:
        print(line)
    print(f"[PromptBuild] Aggregate text tokens (approx): {total_tokens_text}")
    # If base_mode is enabled, collect exactly 5 images per clip (no compression) and encode as base64
    images_to_send = None
    if base_mode:
        max_per_clip = 5  # 固定每个clip保留5张图片
        import base64

        def _resolve_local_path(p: str) -> str:
            """Try to resolve possibly-remote or moved frame path to a local path.
            Strategies:
            - return as-is if exists
            - map common remote prefix to current cwd
            - search by basename under likely folders
            Returns original path if nothing found.
            """
            try:
                if p and os.path.exists(p):
                    return p
            except Exception:
                pass
            # map common remote prefix used in logs to current cwd
            remote_prefix = "/work/Vimo/VideoRAG-algorithm"
            try:
                cwd = os.getcwd()
            except Exception:
                cwd = None
            if cwd and isinstance(p, str) and p.startswith(remote_prefix):
                candidate = p.replace(remote_prefix, cwd.replace('\\', '/'))
                candidate = os.path.normpath(candidate)
                if os.path.exists(candidate):
                    return candidate
            # fallback: search by basename in a few likely dirs
            try:
                base = os.path.basename(p)
                search_roots = [cwd or '.', os.path.join(cwd, 'group') if cwd else 'group', os.path.join(cwd, 'workdir') if cwd else 'workdir', os.path.join(cwd, 'videorag-workdir') if cwd else 'videorag-workdir']
                for root in [r for r in search_roots if r]:
                    pattern = os.path.join(root, '**', base)
                    for match in glob.glob(pattern, recursive=True):
                        if os.path.exists(match):
                            return match
            except Exception:
                pass
            return p

        for data in all_segment_data:
            frames_with_ts = data.get('frames_with_ts', [])
            # take up to max_per_clip frames
            for fp, _ in frames_with_ts[:max_per_clip]:
                try:
                    if not fp:
                        continue
                    resolved = _resolve_local_path(fp)
                    if not os.path.exists(resolved):
                        # debug: show attempted resolution
                        print(f"[PromptBuild][WARN] image file not found: {fp} -> tried: {resolved}")
                        continue
                    with open(resolved, 'rb') as f:
                        b = f.read()
                    all_images_b64.append(base64.b64encode(b).decode('ascii'))
                    image_counter += 1
                except Exception as e:
                    print(f"[PromptBuild][WARN] failed to read image {fp}: {e}")
                    continue
        images_to_send = all_images_b64 if all_images_b64 else None
        print(f"[PromptBuild] base_mode -> {image_counter} images will be sent (no compression, 5 frames per clip).")
    else:
        print(f"[PromptBuild] Images disabled -> 0 images will be sent (captions used instead).")
    # 输出 caption 汇总日志
    try:
        _avg_cap = (_cap_total_chars // _cap_non_empty) if _cap_non_empty else 0
        print(f"[Answer-Caption] clips={_cap_clips} non_empty={_cap_non_empty} total_chars={_cap_total_chars} avg_non_empty_chars={_avg_cap}")
    except Exception:
        pass

    context_str = "\n\n".join(llm_input_chunks)
    
    final_prompt_template = """
### **Multimodal Evidence Synthesis for Question Answering**

**Role:**
You are an expert AI assistant specializing in answering questions based on a provided set of video evidence.

**Objective:**
Your goal is to construct a detailed and accurate answer by carefully analyzing, integrating, and synthesizing all relevant information from a collection of video segments.

**Context Description:**
You will be provided with keyframes and corresponding subtitles from 5 video segments. Each segment is identified by a unique ID (e.g., `EKPFZPyQurA_3`). To answer the question thoroughly, you must combine multiple pieces of information. These details may be found within a single segment's text and images, or they may be spread across several different segments. You must act as an analyst, gathering all necessary evidence before constructing your final response.

**Step-by-Step Task Instructions:**

1.  **Analyze All Evidence:** Meticulously review all subtitles and visually examine the content of all keyframes from every provided video segment.
2.  **Extract Relevant Details:** Identify and extract every piece of information—both textual and visual—that directly contributes to answering the user's question.
3.  **Synthesize a Coherent Answer:** Integrate all extracted details into a single, comprehensive, and logically structured answer. Do not simply list facts; explain how they connect to fully address the question.

**Critical Constraints:**

  * **Absolute Grounding in Evidence:** Your answer **MUST** be derived **exclusively** from the information contained within the provided subtitles and keyframes. This is your only source of truth.
  * **No External Knowledge:** You **MUST NOT** use any pre-existing knowledge, make assumptions, or infer information that is not explicitly presented in the provided evidence. The provided context is your entire world.
  * **Insufficiency Clause:** If, after careful analysis, the combined evidence from all segments is still insufficient to answer the question, you **MUST** state: "The provided information is insufficient to answer the question."
  * **Output Format:** The final output **MUST** be a single, valid JSON object. Do not include any additional text, explanations, or markdown formatting outside of the specified JSON structure.

-----

**Input Structure:**

You will receive the input in the following format:

```
User Question: {user_question}

{context_str}
```

-----

**Output Specification:**

Your response must be a single JSON object in the following format:

```json
{{
  "answer": "..."
}}
```"""
    final_prompt = final_prompt_template.format(user_question=query, context_str=context_str)

    print("\n--- [Step 4] Sending prompt to LLM ---")
    print(f"Prompt text length: {len(final_prompt)}")
    print(f"Number of images: {image_counter if image_counter else 0} {'(disabled)' if not base_mode else ''}")
    print("----------------------------------\n")

    param_response_type = 'The final output MUST be a single, valid JSON object. Do not include any additional text, explanations, apologies, or markdown formatting outside of the JSON structure.'

    response = await llm_cfg.cheap_model_func(
        prompt=final_prompt,
        system_prompt=param_response_type,
        images_base64=images_to_send,  # 在 base_mode 下传递压缩图像 base64
        max_new_tokens=512,
        temperature=0.25,
        top_p=0.9
    )

    print("\n--- [Step 5] Processing LLM response ---")
    answer_obj = None
    try:
        start_index = response.find('{')
        end_index = response.rfind('}') + 1
        if start_index != -1 and end_index != -1:
            json_str = response[start_index:end_index]
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "answer" in parsed:
                answer_obj = {"answer": str(parsed.get("answer", ""))}
            else:
                answer_obj = {"answer": str(response).strip()}
        else:
            answer_obj = {"answer": str(response).strip()}
    except Exception:
        answer_obj = {"answer": str(response or "").strip()}

    # --- 新增：统一抽取最终干净答案 ---
    answer_obj["answer"] = extract_final_answer(answer_obj.get("answer", ""))
    return answer_obj
