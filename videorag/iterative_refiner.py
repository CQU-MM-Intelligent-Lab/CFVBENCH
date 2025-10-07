import json
import os
import time
import asyncio
from typing import List, Dict, Any
from ._config import (
    DEDUP_PHASH_THRESHOLD_DEFAULT,
    DEDUP_DEBUG,
    FRAME_COUNT_MAPPING_EXTENDED,
)
from ._llm import (
    ollama_refiner_func,
    get_default_ollama_chat_model,
    internvl_refiner_func,
    custom_openai_complete_if_cache,
    openai_complete_if_cache,
)
from ._llm_common import is_ollama_model_warmed, mark_ollama_model_warmed, get_ollama_async_client_instance
from .refine_frames_utils import (
    _map_score_to_frames,
    _ir_average_hash,
    _ir_hamming_int,
    _ir_frame_hash,
    _ir_hash_distance,
    _dedup_frames,
    _ensure_dir,
    _uniform_subsample,
    _extract_frame_opencv,
    _interpolate_and_fill_frames,
)

class IterativeRefiner:
    """
    Encapsulates the logic for iterative retrieval and refinement.
    """

    def __init__(self, config: Dict[str, Any], llm_api_func: callable = None):
        self.config = config
        self.llm_api_func = llm_api_func
        # Warm-up state: to avoid first-call model load latency we warm the model once
        self._warmed = False
        # Try to warm up model at startup in background (non-blocking)
        try:
            do_warm = os.environ.get("REFINER_WARMUP_ON_FIRST_CALL", "1").strip() in {"1", "true", "True"}
        except Exception:
            do_warm = True
        if do_warm:
            model_name = os.environ.get("REFINER_OLLAMA_MODEL", "").strip() or get_default_ollama_chat_model()
            try:
                loop = asyncio.get_running_loop()
                # If this model has been warmed by another module/instance, avoid scheduling duplicate warmup
                try:
                    if is_ollama_model_warmed(model_name):
                        self._warmed = True
                    else:
                        # schedule warmup in running loop
                        loop.create_task(self._warmup(model_name))
                except Exception:
                    # fallback: schedule warmup
                    loop.create_task(self._warmup(model_name))
            except RuntimeError:
                # no running loop: start a daemon thread to run warmup
                import threading
                def _bg():
                    try:
                        asyncio.run(self._warmup(model_name))
                    except Exception:
                        pass
                t = threading.Thread(target=_bg, daemon=True)
                t.start()

    def _infer_modalities_from_query(self, query: str) -> Dict[str, bool]:
        """Heuristic fallback: infer OCR/DET needs from query text when evaluator didn't mark them.
        Only used when total score indicates refinement but checks are all false.
        """
        q = (query or "").lower()
        # OCR-related hints
        ocr_keywords = {
            # 1. 明确的读取/识别动作 (Direct Actions)
            "read", "what does it say", "text", "word", "character", "symbol", "sign",
            "transcribe", "identify the text", "name of",
            # 2. UI 界面元素 (UI Elements)
            "ui", "menu", "button", "icon", "slider", "dropdown", "checkbox",
            "input field", "text box", "dialog", "window", "panel", "tab", "option", "toolbar",
            # 3. 结构化数据 (Structured Data)
            "chart", "graph", "axis", "axes", "table", "data", "list", "form", "figure",
            "diagram", "spreadsheet", "code", "log",
            # 4. 数字与标识信息 (Numeric & Identifier Info)
            "number", "score", "price", "date", "time", "version", "percentage",
            "amount", "value", "coordinate", "id", "serial number", "statistics",
            # 5. 文档与内容 (Document & Content)
            "label", "title", "headline", "caption", "subtitle", "paragraph", "document", "article"
        }
        # DET-related hints
        det_keywords = {
            # 1. 明确的检测/定位动作 (Direct Actions)
            "object", "detect", "find", "locate", "point to", "identify the object",
            "count", "how many", # 'how many' 通常需要先检测到物体再计数
            # 2. 空间关系 (Spatial Relationships) - 非常重要
            "position", "location", "where is",
            "left", "right", "top", "bottom", "center", "middle", "corner",
            "above", "below", "under", "over", "on top of",
            "beside", "next to", "between",
            "in front of", "behind", "inside", "outside",
            # 3. 物体属性 (Object Attributes)
            "color", "shape", "size", "texture", "pattern", "style", "material",
            # 4. 指代与选择 (Selection & Reference)
            "which", "what object", "which one", "the one that is",
            # 5. 交互与状态 (Interaction & State)
            "hold", "holding", "wearing", "using", "carrying",
            # 6. 通用物体名词 (General Object Nouns)
            "item", "thing", "person", "car", "animal", "building", "tool", "element", "component"
        }
        need_ocr = any(k in q for k in ocr_keywords)
        need_det = any(k in q for k in det_keywords)
        return {"ocr": need_ocr, "det": need_det}

    def _is_numeric_question(self, query: str) -> bool:
        """Heuristic: detect if the query likely requires reading numeric/structured values from visuals."""
        q = (query or "").lower()
        import re
        # 更保守的检测规则，减少泛化与单个数字误触发：
        # 1) 排除明显为解释/定性的问题
        exclusion_tokens = {
            "why", "how does", "how to", "what is the purpose of", "explain", "describe",
            "summarize", "meaning", "purpose", "method", "cause", "reason", "consequence",
            "who", "which person", "which company", "what is the name", "what color", "what kind", "what type"
        }
        if any(t in q for t in exclusion_tokens):
            return False

        # 2) 极强信号（短语匹配）——出现这些短语之一即可认为需要数值读取
        very_strong = {
            "how many", "how much", "count", "what is the percentage", "what is the rate",
            "what is the frequency", "what is the value", "what is the number", "what is the total",
            "percentage", "percent", "%", "rate", "frequency"
        }
        if any(phrase in q for phrase in very_strong):
            return True

        # 3) 支撑性信号（图表/表格/轴/图例等）必须与多个数值或单位一起出现才触发
        support_tokens = {
            "graph", "chart", "plot", "diagram", "figure", "table", "data",
            "axis", "axes", "legend", "scale", "range", "column", "row",
            "statistics", "stats", "average", "mean", "median", "distribution",
            "correlation", "ratio"
        }

        # detect numbers and units
        numbers = re.findall(r"\b\d+(?:[\.,]\d+)?\b", q)
        unit_pairs = re.findall(r"-?\d+\s*(°c|℃|c|%|percent|年|years?|月|小时|h|km|km/h|m/s|kg|kgf)", q)
        has_support = any(tok in q for tok in support_tokens)

        # 如果出现图表/表格相关词且同时出现多个数值或单位 -> 触发
        if has_support and (len(numbers) >= 2 or len(unit_pairs) >= 1):
            return True

        # 4) 货币/百分号等符号单独出现时也可视为数值需求，但要求与上下文词汇配合
        if re.search(r"[\$€£¥]", q) and (has_support or len(numbers) >= 1):
            return True

        # 5) 范围表达（如 "from 10 to 20" / "10-20"）与单位结合时触发
        if re.search(r"\b\d+\s*(to|-)\s*\d+\b", q) and has_support:
            return True

        return False

    async def _generate_visual_keywords(self, query: str, context: List[Dict[str, Any]]) -> List[str]:
        """Generates a targeted list of visual keywords using an LLM."""
        context_str = "\n".join([
            f"- (Time: {c.get('start_time', 'N/A')}-{c.get('end_time', 'N/A')}s): {c.get('summary', 'No summary available.')}"
            for c in context
        ])
        prompt = f"""
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
"""
        refiner_model = os.environ.get("REFINER_OLLAMA_MODEL", "").strip() or get_default_ollama_chat_model()

        # 并发与超时控制：避免 keyword 生成长时间阻塞主流程
        timeout_sec = float(os.environ.get("REFINER_KEYWORD_TIMEOUT", "30"))
        try:
            async def _call_refiner_by_config():
                # choose backend based on SKIP_OLLAMA and model name
                skip = os.environ.get("SKIP_OLLAMA", "").strip().lower() in {"1", "true", "yes"}
                if "internvl" in (refiner_model or "").lower():
                    return await internvl_refiner_func(model_name=refiner_model, prompt=prompt)
                if skip:
                    base = os.environ.get("CUSTOM_OPENAI_BASE_URL", "").strip()
                    key = os.environ.get("CUSTOM_OPENAI_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
                    model = os.environ.get("CUSTOM_OPENAI_MODEL", "").strip() or os.environ.get("OPENAI_MODEL", "").strip() or refiner_model
                    if base and key:
                        return await custom_openai_complete_if_cache(model, prompt, base_url=base, api_key=key)
                    if os.environ.get("OPENAI_API_KEY", ""):
                        return await openai_complete_if_cache(refiner_model, prompt)
                # fallback to Ollama
                return await ollama_refiner_func(model_name=refiner_model, prompt=prompt)

            response_str = await asyncio.wait_for(_call_refiner_by_config(), timeout=timeout_sec)

            # Robustly parse the JSON list from the response
            start = response_str.find('[')
            end = response_str.rfind(']')
            if start != -1 and end != -1 and end > start:
                keywords = json.loads(response_str[start:end+1])
                if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
                    return keywords[:15]
        except asyncio.TimeoutError:
            print(f"[Refine-KeywordGen][Timeout] > {timeout_sec}s, fallback to heuristic extraction.")
        except Exception as e:
            print(f"[Refine-KeywordGen] Failed to generate or parse keywords: {e}")

        # Fallback to simple extraction if LLM fails
        from ._videoutil.refinement_utils import extract_keyword_queries_from_query
        return extract_keyword_queries_from_query(query)

    async def _warmup(self, model_name: str) -> None:
        """Perform a short LLM call to warm the model (preload weights). Non-fatal."""
        try:
            probe_prompt = "Please respond with a short JSON: {\"warm\": true}."
            # Use a small timeout for warmup but tolerate failure
            try:
                # Decide whether to skip Ollama before initializing any Ollama client
                skip = os.environ.get("SKIP_OLLAMA", "").strip().lower() in {"1", "true", "yes"}

                # Only initialize the Ollama async client when not skipping Ollama
                if not skip:
                    try:
                        # ensure underlying Ollama client is initialized (no-op if already created)
                        get_ollama_async_client_instance()
                    except Exception:
                        pass

                # Warmup the appropriate backend: internvl, custom/openai (when skip), or Ollama
                if "internvl" in (model_name or "").lower():
                    await asyncio.wait_for(internvl_refiner_func(model_name=model_name, prompt=probe_prompt), timeout=10)
                elif skip:
                    base = os.environ.get("CUSTOM_OPENAI_BASE_URL", "").strip()
                    key = os.environ.get("CUSTOM_OPENAI_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
                    model = os.environ.get("CUSTOM_OPENAI_MODEL", "").strip() or os.environ.get("OPENAI_MODEL", "").strip() or model_name
                    if base and key:
                        await asyncio.wait_for(custom_openai_complete_if_cache(model, probe_prompt, base_url=base, api_key=key), timeout=10)
                    elif os.environ.get("OPENAI_API_KEY", ""):
                        await asyncio.wait_for(openai_complete_if_cache(model_name, probe_prompt), timeout=10)
                    else:
                        # No API configured; skip warmup instead of forcing Ollama
                        print(f"[Refiner][Warmup] SKIP_OLLAMA is set but no API credentials found; skipping warmup for model={model_name}")
                else:
                    await asyncio.wait_for(ollama_refiner_func(model_name=model_name, prompt=probe_prompt), timeout=10)
                self._warmed = True
                try:
                    mark_ollama_model_warmed(model_name)
                except Exception:
                    pass
                print(f"[Refiner][Warmup] model={model_name} warmup succeeded")
            except asyncio.TimeoutError:
                print(f"[Refiner][Warmup] model={model_name} warmup timed out (10s)")
            except Exception as e:
                print(f"[Refiner][Warmup] model={model_name} warmup error: {e}")
        except Exception:
            # Ensure warmup never throws
            pass

    def _build_evaluation_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        # 使用段级 caption + 字幕；caption 来自 5 帧聚合，可为空。
        # 简洁日志: 统计当前用于评估的 caption 使用情况（最小侵入）
        try:
            _caps = [(c.get('id'), (c.get('caption') or '').strip()) for c in (context or [])]
            _non_empty = [c for c, txt in _caps if txt]
            _total_chars = sum(len(txt) for _, txt in _caps if txt)
            _avg = (_total_chars // len(_non_empty)) if _non_empty else 0
            print(f"[Refiner-Caption] clips={len(_caps)} non_empty={len(_non_empty)} total_chars={_total_chars} avg_non_empty_chars={_avg}")
        except Exception:
            pass
        lines = []
        # 去除对 caption 的裁剪; 仅可对字幕做可选裁剪
        max_field = int(os.environ.get("REFINER_EVAL_MAX_FIELD_CHARS", "280") or 280)
        def _trunc_sub(txt: str) -> str:
            if not txt: return ""
            return (txt[:max_field] + "…") if len(txt) > max_field else txt
        for c in context:
            lines.append(
                f"- Clip (ID: {c.get('id','N/A')}, Time: {c.get('start_time','N/A')}-{c.get('end_time','N/A')}s)\n  Caption: {(c.get('caption','') or '').strip()}\n  Subtitles: {_trunc_sub(c.get('summary','No transcript available.'))}"
            )
        context_str = "\n".join(lines)
        prompt = f"""
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
"""
        return prompt


    # 全局并发限制（默认单并发）
    _EVAL_SEM = asyncio.Semaphore(int(os.environ.get("REFINER_MAX_PARALLEL", "1")))

    async def _evaluate_context(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calls the LLM to evaluate the context and returns the parsed JSON response."""
        prompt = self._build_evaluation_prompt(query, context)
        timeout_sec = float(self.config.get("refiner_timeout_seconds") or os.environ.get("REFINER_TIMEOUT_SECONDS", "90"))
        fallback_mode = os.environ.get("REFINER_TIMEOUT_FALLBACK", "final").lower()
        refiner_model = os.environ.get("REFINER_OLLAMA_MODEL", "").strip() or get_default_ollama_chat_model()

        # Warm-up on first real evaluation to preload model weights (avoids first-call stall)
        try:
            do_warm = os.environ.get("REFINER_WARMUP_ON_FIRST_CALL", "1").strip() in {"1", "true", "True"}
        except Exception:
            do_warm = True
        if do_warm and not getattr(self, "_warmed", False):
            # If another instance already warmed this model, mark warmed and continue
            try:
                if is_ollama_model_warmed(refiner_model):
                    self._warmed = True
                else:
                    # schedule warmup in background but do not block evaluation
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._warmup(refiner_model))
                    except RuntimeError:
                        # no running loop: run warmup synchronously but with timeout
                        try:
                            asyncio.run(asyncio.wait_for(self._warmup(refiner_model), timeout=12))
                        except Exception:
                            pass
            except Exception as e:
                print(f"[Refiner][Warmup] failed: {e}")

        async def _call_model():
            skip = os.environ.get("SKIP_OLLAMA", "").strip().lower() in {"1", "true", "yes"}
            if "internvl" in (refiner_model or "").lower():
                return await internvl_refiner_func(model_name=refiner_model, prompt=prompt, images_base64=None)
            if skip:
                base = os.environ.get("CUSTOM_OPENAI_BASE_URL", "").strip()
                key = os.environ.get("CUSTOM_OPENAI_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
                model = os.environ.get("CUSTOM_OPENAI_MODEL", "").strip() or os.environ.get("OPENAI_MODEL", "").strip() or refiner_model
                if base and key:
                    return await custom_openai_complete_if_cache(model, prompt, base_url=base, api_key=key)
                if os.environ.get("OPENAI_API_KEY", ""):
                    return await openai_complete_if_cache(refiner_model, prompt)
                # fallback to ollama if nothing else configured
            return await ollama_refiner_func(model_name=refiner_model, prompt=prompt, images_base64=None)

        t0 = time.time()
        try:
            # Debug: surface model and prompt size before calling LLM
            try:
                print(f"[Refiner][EvalCall] model={refiner_model} prompt_len={len(prompt)}")
            except Exception:
                pass
            async with self._EVAL_SEM:
                # Retry loop: wait up to per_attempt_timeout for each attempt, then retry
                attempts = int(os.environ.get("REFINER_RETRY_ATTEMPTS", "5"))
                per_attempt_timeout = float(os.environ.get("REFINER_RETRY_TIMEOUT", "30"))
                llm_response_str = None
                for attempt in range(1, attempts + 1):
                    try:
                        print(f"[Refiner][Retry] attempt={attempt}/{attempts} timeout_per_attempt={per_attempt_timeout}s")
                        llm_response_str = await asyncio.wait_for(_call_model(), timeout=per_attempt_timeout)
                        # got response
                        break
                    except asyncio.TimeoutError:
                        print(f"[Refiner][Retry] attempt {attempt} timed out after {per_attempt_timeout}s")
                        if attempt == attempts:
                            # exhausted attempts -> treat as timeout
                            elapsed = time.time() - t0
                            print(f"[Refiner][Timeout] All {attempts} attempts timed out (elapsed {elapsed:.1f}s). Fallback -> {fallback_mode}.")
                            # 超时：默认使用 env 可配置的上限分数 (默认 5)
                            try:
                                ans_score_on_timeout = int(os.environ.get("REFINER_TIMEOUT_SCORE_ON_TIMEOUT", "5"))
                            except Exception:
                                ans_score_on_timeout = 5
                            print(f"[Refiner][Timeout] ans_score_on_timeout={ans_score_on_timeout}")
                            return {
                                "overall_answerability_score": ans_score_on_timeout,
                                "information_density_score": 0,
                                "numeric_evidence_required": False,
                                "temporal_sequence_incomplete": False,
                                "temporal_focus_clip_id": "",
                                "refinement_targets": [],
                                "parse_error": True,
                                "timeout": True,
                                "fallback_mode": fallback_mode
                            }
                        # else continue to next attempt
                    except Exception as e:
                        # Other errors -> fail fast to outer handler
                        raise
        except Exception as e:
            print(f"[Refiner][EvalError] {e}. Fallback refine-all.")
            return {
                "overall_answerability_score": 0,
                "information_density_score": 0,
                "numeric_evidence_required": False,
                "temporal_sequence_incomplete": False,
                "temporal_focus_clip_id": "",
                "refinement_targets": [],
                "parse_error": True,
                "error": str(e)
            }
        else:
            elapsed = time.time() - t0
            print(f"[Refiner][EvalDone] model={refiner_model} elapsed={elapsed:.2f}s prompt_tokens~{len(prompt)//4}")

        # Early sanity checks: if the model returned an obvious error string
        # or an empty response, short-circuit to a consistent fallback so
        # downstream logic doesn't attempt to parse invalid text.
        if not llm_response_str or (isinstance(llm_response_str, str) and llm_response_str.strip() == ""):
            print(f"[Refiner][EvalError] Empty LLM response. Raw head: '{(llm_response_str or '')[:200]}'")
            return {
                "overall_answerability_score": 0,
                "information_density_score": 0,
                "numeric_evidence_required": False,
                "temporal_sequence_incomplete": False,
                "temporal_focus_clip_id": "",
                "refinement_targets": [],
                "parse_error": True,
                "error": "Empty LLM response"
            }

        if isinstance(llm_response_str, str) and llm_response_str.strip().lower().startswith(("internvl generate error", "internvl generate error:", "error:", "internvl error")):
            head = llm_response_str.strip()[:300].replace("\n", "\\n")
            print(f"[Refiner][EvalError] LLM returned error-like response. Raw head: {head}")
            return {
                "overall_answerability_score": 0,
                "information_density_score": 0,
                "numeric_evidence_required": False,
                "temporal_sequence_incomplete": False,
                "temporal_focus_clip_id": "",
                "refinement_targets": [],
                "parse_error": True,
                "error": f"LLM error response: {head}"
            }

        def _extract_json(s: str | None) -> Dict[str, Any]:
            if not s:
                raise ValueError("Empty LLM response")
            import re
            raw = s.strip()
            if raw.startswith("```"):
                lines = [ln for ln in raw.splitlines() if ln.strip() not in ("```", "```json", "```JSON")]
                raw = "\n".join(lines).strip()
            if raw.lower().startswith("json"):
                brace_pos = raw.find("{")
                if brace_pos != -1:
                    raw = raw[brace_pos:]
            def _sanitize_invalid_escapes(txt: str) -> str:
                return re.sub(r'\\([^"\\/bfnrtu])', r'\1', txt)
            def _sanitize_bad_keys(txt: str) -> str:
                return re.sub(r'([\{\s,])\s*-\s*(")', r'\1\2', txt)
            sanitized_text = _sanitize_bad_keys(_sanitize_invalid_escapes(raw))
            start = sanitized_text.find("{")
            end = sanitized_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = sanitized_text[start:end+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
            return json.loads(sanitized_text)

        def _tolerant_parse(raw: str) -> Dict[str, Any]:
            import re
            txt = (raw or "").strip()
            # remove code fences
            if txt.startswith("```"):
                lines = [ln for ln in txt.splitlines() if not ln.strip().startswith("```")]
                txt = "\n".join(lines)

            # remove common model thought/analysis markers and leading angle-bracket tags
            txt = re.sub(r"<\/?think[^>]*>", "", txt, flags=re.IGNORECASE)
            txt = re.sub(r"^<[^\n>]+>\s*", "", txt, flags=re.MULTILINE)

            # quick helper: try to find a balanced JSON object and parse it
            def _find_json_object(s: str):
                depth = 0
                start = None
                for i, ch in enumerate(s):
                    if ch == '{':
                        if start is None:
                            start = i
                        depth += 1
                    elif ch == '}':
                        if depth > 0:
                            depth -= 1
                            if depth == 0 and start is not None:
                                cand = s[start:i+1]
                                # try direct parse
                                try:
                                    return json.loads(cand)
                                except Exception:
                                    # gentle sanitize: convert single quotes to double quotes if appears
                                    try:
                                        cand2 = cand.replace("\"'", '"')
                                        cand2 = cand2.replace("'", '"')
                                        return json.loads(cand2)
                                    except Exception:
                                        pass
                                start = None
                return None

            try:
                parsed = _find_json_object(txt)
                if isinstance(parsed, dict):
                    # map common alternative keys to canonical fields
                    def _int_of(keys, default=0):
                        for k in keys:
                            v = parsed.get(k)
                            if v is None:
                                continue
                            try:
                                return int(v)
                            except Exception:
                                try:
                                    return int(str(v).strip())
                                except Exception:
                                    continue
                        return default

                    def _bool_of(keys):
                        for k in keys:
                            if k in parsed:
                                return bool(parsed.get(k))
                        return False

                    res = {
                        'overall_answerability_score': _int_of(['overall_answerability_score', 'overall_answerability', 'answerability_score', 'overall_answerability']),
                        'information_density_score': _int_of(['information_density_score', 'information_density', 'density_score']),
                        'numeric_evidence_required': _bool_of(['numeric_evidence_required', 'numeric_required']),
                        'temporal_sequence_incomplete': _bool_of(['temporal_sequence_incomplete', 'temporal_incomplete']),
                        'temporal_focus_clip_id': parsed.get('temporal_focus_clip_id') or parsed.get('temporal_focus') or "",
                        'refinement_targets': []
                    }

                    # extract refinement targets if present
                    rt = parsed.get('refinement_targets') or parsed.get('targets') or []
                    if isinstance(rt, list):
                        for t in rt:
                            if not isinstance(t, dict):
                                continue
                            cid = t.get('clip_id') or t.get('id') or ''
                            reasoning = t.get('reasoning') or t.get('reason') or ''
                            checks = t.get('checks') or {}
                            res['refinement_targets'].append({
                                'clip_id': str(cid),
                                'reasoning': reasoning,
                                'checks': {
                                    'temporal_coherence_needed': bool(checks.get('temporal_coherence_needed') or checks.get('temporal') or False),
                                    'ocr_needed': bool(checks.get('ocr_needed') or checks.get('ocr') or False),
                                    'det_needed': bool(checks.get('det_needed') or checks.get('det') or False),
                                }
                            })

                    # numeric_focus_clip_id
                    res['numeric_focus_clip_id'] = parsed.get('numeric_focus_clip_id') or parsed.get('numeric_focus') or ""

                    # accept parsed response if it contains any meaningful info
                    if (res.get('overall_answerability_score', 0) + res.get('information_density_score', 0)) > 0 or res.get('refinement_targets'):
                        return res
            except Exception:
                pass

            # Fallback: original regex-based extraction (cleaned)
            txt = re.sub(r'\\([^"\\/bfnrtu])', r'\1', txt)
            txt = re.sub(r'([{"\s,])\-\s*"', r'\1"', txt)
            txt = re.sub(r'([{"\s,])\-\s*([a-zA-Z_])', r'\1"\2', txt)
            def _fix_clip_id(m):
                val = m.group(1) or m.group(2)
                return f'"clip_id": "{val}"'
            txt = re.sub(r'"clip_id"\s*:\s*(?:"([^"]+)"|([A-Za-z0-9_\-.]+))', _fix_clip_id, txt)

            def _find_int(pat: str) -> int:
                m = re.search(pat, txt)
                if m:
                    try:
                        return int(m.group(1))
                    except:
                        return 0
                return 0

            ans_score = _find_int(r'"overall[_\\]?answerability[_\\]?score"\s*:\s*(\d)')
            den_score = _find_int(r'"information[_\\]?density[_\\]?score"\s*:\s*(\d)')
            targets = []
            for block in re.findall(r'\{[^{}]*"clip_id"[^{}]*\}', txt):
                cid_m = re.search(r'"clip_id"\s*:\s*"([^"]+)"', block)
                if not cid_m:
                    continue
                cid = cid_m.group(1).strip()
                reasoning_m = re.search(r'"reasoning"\s*:\s*"([^"]+)"', block)
                reasoning = reasoning_m.group(1).strip() if reasoning_m else ""
                def _bool(key: str) -> bool:
                    return bool(re.search(rf'"{key}"\s*:\s*true', block, re.IGNORECASE))
                targets.append({
                    "clip_id": cid,
                    "reasoning": reasoning,
                    "checks": {
                        "temporal_coherence_needed": _bool("temporal_coherence_needed"),
                        "ocr_needed": _bool("ocr_needed"),
                        "det_needed": _bool("det_needed")
                    }
                })

            def _find_bool_flag(name: str) -> bool:
                return bool(re.search(rf'"{name}"\s*:\s*true', txt, re.IGNORECASE))

            temporal_flag = _find_bool_flag("temporal_sequence_incomplete")
            focus_id = ""
            m_focus = re.search(r'"temporal_focus_clip_id"\s*:\s*"([^\"]*)"', txt)
            if m_focus:
                focus_id = m_focus.group(1).strip()
            if not temporal_flag:
                focus_id = ""

            result = {
                "overall_answerability_score": ans_score,
                "information_density_score": den_score,
                "numeric_evidence_required": _find_bool_flag("numeric_evidence_required"),
                "temporal_sequence_incomplete": temporal_flag,
                "temporal_focus_clip_id": focus_id,
                "refinement_targets": targets
            }

            # Attempt to extract optional top-level numeric_focus_clip_id as a string
            try:
                m = re.search(r'"numeric_focus_clip_id"\s*:\s*"([^\"]*)"', txt, re.IGNORECASE)
                if m:
                    result["numeric_focus_clip_id"] = m.group(1).strip()
                else:
                    m2 = re.search(r'"numeric_focus_clip_id"\s*:\s*([A-Za-z0-9_\-\.]+)', txt, re.IGNORECASE)
                    if m2:
                        result["numeric_focus_clip_id"] = m2.group(1).strip()
            except Exception:
                pass

            if (ans_score + den_score) == 0 and not targets and not temporal_flag:
                raise ValueError("Tolerant parse extracted nothing meaningful.")
            return result

        try:
            evaluation_result = _extract_json(llm_response_str)
            evaluation_result.setdefault("temporal_sequence_incomplete", False)
            evaluation_result.setdefault("temporal_focus_clip_id", "")
            evaluation_result.setdefault("numeric_evidence_required", False)
            evaluation_result.setdefault("refinement_targets", [])
            evaluation_result.setdefault("numeric_focus_clip_id", "")
            evaluation_result["parse_error"] = False
            evaluation_result["tolerant"] = False
        except Exception as strict_err:
            try:
                evaluation_result = _tolerant_parse(llm_response_str)
                evaluation_result["parse_error"] = False
                evaluation_result["tolerant"] = True
                try:
                    raw_head = (llm_response_str or "")[:400].replace("\n", "\\n")
                    print(f"[Refiner][ParseOK/Tolerant] ans={evaluation_result.get('overall_answerability_score')} den={evaluation_result.get('information_density_score')} targets={len(evaluation_result.get('refinement_targets') or [])} temporal={evaluation_result.get('temporal_sequence_incomplete')}\n  Raw head: {raw_head}")
                except Exception:
                    print(f"[Refiner][ParseOK/Tolerant] ans={evaluation_result.get('overall_answerability_score')} den={evaluation_result.get('information_density_score')} targets={len(evaluation_result.get('refinement_targets') or [])} temporal={evaluation_result.get('temporal_sequence_incomplete')}")
                evaluation_result.setdefault("numeric_focus_clip_id", "")
            except Exception as tol_err:
                raw_head = (llm_response_str or "")[:500].replace("\n", "\\n")
                print(f"[Refiner][ParseError] strict={strict_err}; tolerant={tol_err}. Raw head: {raw_head}")
                return {
                    "sufficient": False,
                    "refinement_targets": [],
                    "reasoning": "Fallback due to LLM response parsing error.",
                    "parse_error": True,
                    "raw_head": raw_head,
                    "strict_exception": str(strict_err),
                    "tolerant_exception": str(tol_err)
                }
            # Log numeric_focus_clip_id for observability
            try:
                _nf = evaluation_result.get("numeric_focus_clip_id", "")
                print(f"[Refiner][ParsedNumericFocus] numeric_focus_clip_id='{_nf}' numeric_required={evaluation_result.get('numeric_evidence_required', False)} tolerant={evaluation_result.get('tolerant', False)}")
            except Exception:
                pass

            # Merge optional numeric_focus_clip_id into refinement_targets for downstream convenience
        try:
            nfid = str(evaluation_result.get("numeric_focus_clip_id") or "").strip()
            if nfid:
                existing_ids = {t.get('clip_id') for t in (evaluation_result.get('refinement_targets') or []) if t.get('clip_id')}
                if nfid not in existing_ids:
                    evaluation_result.setdefault('refinement_targets', []).append({
                        'clip_id': nfid,
                        'reasoning': 'numeric_focus_requested',
                        'checks': {'temporal_coherence_needed': False, 'ocr_needed': True, 'det_needed': False}
                    })
        except Exception:
            pass
        return evaluation_result

    def _decide(self, evaluation: Dict[str, Any], query: str, initial_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        def frames_for_total(t: int) -> int:
            mapping = {5: 20, 4: 24, 3: 28, 2: 32, 1: 36, 0: 40}
            return mapping.get(max(0, min(5, t)), 40)

        # 统一计算得分
        try:
            s_ans = int(evaluation.get("overall_answerability_score", 0))
        except:
            s_ans = 0
        try:
            s_den = int(evaluation.get("information_density_score", 0))
        except:
            s_den = 0
        s_ans = max(0, min(5, s_ans))
        s_den = max(0, min(5, s_den))
        has_scores = not evaluation.get("parse_error", False)

        # 超时回退处理
        if evaluation.get("timeout") and (evaluation.get("fallback_mode") or os.environ.get("REFINER_TIMEOUT_FALLBACK", "final")).lower() == "final":
            fallback_total = s_ans + s_den
            if fallback_total > 5:
                return {"status": "final", "targets": [], "scores": {"answerability": s_ans, "density": s_den, "total": min(10, fallback_total)}, "temporal_sequence_incomplete": False, "temporal_focus_clip_id": ""}
            evaluation["parse_error"] = False
            has_scores = True

        total = (s_ans + s_den) if has_scores else 0
        scores = {"answerability": s_ans, "density": s_den, "total": total}
        FORCE_REFINE = os.environ.get("FORCE_REFINE_ALL", "0").strip() in {"1", "true", "True"}
        temporal_flag = bool(evaluation.get("temporal_sequence_incomplete"))
        temporal_focus = evaluation.get("temporal_focus_clip_id") or ""
        targets_eval = evaluation.get("refinement_targets", []) or []

        # ---------- Temporal-focused logic ----------
        if temporal_flag and temporal_focus:
            # 三种组合:
            # A: temporal + numeric + (OCR/DET) -> 按映射 frames_for_total(total) 对所有 clip
            # B: temporal + numeric (无 OCR/DET) -> 使用默认 fine_num_frames (或 15) 对所有 clip
            # C: temporal only -> 对显式 target refine，否则不 refine（保留 5 帧）
            numeric_trigger = bool(evaluation.get("numeric_evidence_required")) or self._is_numeric_question(query)

            # Collect per-target checks map
            checks_map: Dict[str, Dict[str, Any]] = {}
            any_ocr_det_flag = False
            for t in targets_eval:
                cid = t.get("clip_id")
                ch = (t or {}).get("checks", {}) or {}
                if cid:
                    checks_map[cid] = ch
                if ch.get("ocr_needed") or ch.get("det_needed"):
                    any_ocr_det_flag = True

            # A: numeric + any ocr/det -> refine ONLY the explicit targets returned by evaluator (do not force all clips)
            if numeric_trigger and any_ocr_det_flag:
                if targets_eval:
                    new_frames = frames_for_total(total)
                    decision_targets = []
                    for t in targets_eval:
                        cid = t.get("clip_id")
                        if not cid:
                            continue
                        ch = (t or {}).get("checks", {}) or {}
                        run_ocr = bool(ch.get("ocr_needed")) or True
                        run_det = bool(ch.get("det_needed")) or False
                        decision_targets.append({"clip_id": cid, "refinement_params": {"new_sampling_rate_per_30s": new_frames, "run_ocr": run_ocr, "run_det": run_det}})
                    print(f"[Refiner] Temporal+Numeric+OCR/DET -> mapped frames {new_frames} for selected targets ({len(decision_targets)}).")
                    return {"status": "refine", "targets": decision_targets, "scores": scores, "temporal_sequence_incomplete": True, "temporal_focus_clip_id": temporal_focus}
                else:
                    print("[Refiner] Temporal+Numeric+OCR/DET signaled but no explicit targets in plan -> skipping global OCR refine.")
                    return {"status": "final", "targets": [], "scores": scores, "temporal_sequence_incomplete": True, "temporal_focus_clip_id": temporal_focus}

            # B: numeric but no explicit ocr/det checks -> refine ONLY if evaluator provided explicit targets; otherwise skip
            if numeric_trigger:
                if targets_eval:
                    default_frames = int(self.config.get("fine_num_frames_per_segment", 15) or 15)
                    decision_targets = []
                    for t in targets_eval:
                        cid = t.get("clip_id")
                        if not cid:
                            continue
                        ch = (t or {}).get("checks", {}) or {}
                        run_ocr = bool(ch.get("ocr_needed")) or True
                        run_det = bool(ch.get("det_needed")) or False
                        decision_targets.append({"clip_id": cid, "refinement_params": {"new_sampling_rate_per_30s": default_frames, "run_ocr": run_ocr, "run_det": run_det}})
                    print(f"[Refiner] Temporal+Numeric -> default frames {default_frames} for selected targets ({len(decision_targets)}).")
                    return {"status": "refine", "targets": decision_targets, "scores": scores, "temporal_sequence_incomplete": True, "temporal_focus_clip_id": temporal_focus}
                else:
                    # If numeric+temporal was signaled but no explicit targets were provided,
                    # fallback to refining the temporal_focus clip with denser OCR sampling
                    print("[Refiner] Temporal+Numeric signaled but no explicit targets in plan -> fallback to temporal_focus clip OCR refine.")
                    default_frames = int(self.config.get("fine_num_frames_per_segment", 15) or 15)
                    return {"status": "refine", "targets": [{"clip_id": temporal_focus, "refinement_params": {"new_sampling_rate_per_30s": default_frames, "run_ocr": True, "run_det": False}}], "scores": scores, "temporal_sequence_incomplete": True, "temporal_focus_clip_id": temporal_focus}

            # C: temporal only with explicit targets -> refine those targets with frames_for_total(total)
            if targets_eval:
                decision_targets = []
                for t in targets_eval:
                    cid = t.get("clip_id")
                    if not cid:
                        continue
                    ch = (t or {}).get("checks", {}) or {}
                    run_ocr = bool(ch.get("ocr_needed"))
                    run_det = bool(ch.get("det_needed"))
                    decision_targets.append({"clip_id": cid, "refinement_params": {"new_sampling_rate_per_30s": frames_for_total(total), "run_ocr": run_ocr, "run_det": run_det}})
                print(f"[Refiner] Temporal-only with explicit targets -> refine specified targets with frames_for_total({total}).")
                return {"status": "refine", "targets": decision_targets, "scores": scores, "temporal_sequence_incomplete": True, "temporal_focus_clip_id": temporal_focus}

            # temporal only -> 不 refine 保持 5 帧, 交由下游添加邻居
            print("[Refiner] Temporal only -> no extra refine (keep 5 frames).")
            return {"status": "final", "targets": [], "scores": scores, "temporal_sequence_incomplete": True, "temporal_focus_clip_id": temporal_focus}

        # ---------- Non-temporal or fallback logic ----------
        if FORCE_REFINE:
            print("[Refiner] FORCE_REFINE_ALL active.")

        # 数值证据且非时序问题 -> 全段 OCR refine
        if (bool(evaluation.get("numeric_evidence_required")) or self._is_numeric_question(query)) and not FORCE_REFINE and not temporal_flag:
            # 对非时序的数值请求：仅在 evaluator 返回显式 targets 时对这些目标执行 OCR refine；否则不对所有 clip 强制 OCR
            new_frames = frames_for_total(total)
            if targets_eval:
                decision_targets = []
                for t in (targets_eval or []):
                    cid = t.get("clip_id")
                    if not cid:
                        continue
                    ch = (t or {}).get("checks", {}) or {}
                    run_ocr = bool(ch.get("ocr_needed")) or True
                    run_det = bool(ch.get("det_needed")) or False
                    decision_targets.append({"clip_id": cid, "refinement_params": {"new_sampling_rate_per_30s": new_frames, "run_ocr": run_ocr, "run_det": run_det}})
                print(f"[Refiner] Numeric evidence path -> refine selected targets ({len(decision_targets)}) with frames_for_total({total}).")
                return {"status": "refine", "targets": decision_targets, "scores": scores, "temporal_sequence_incomplete": False, "temporal_focus_clip_id": ""}
            else:
                # No explicit targets: instead of skipping, perform a conservative global OCR refine
                # over all clips in initial_context to avoid missing numeric evidence due to parsing issues.
                print("[Refiner] Numeric evidence requested but no explicit targets provided by plan -> performing global OCR refine over all clips.")
                decision_targets = []
                new_frames = frames_for_total(total)
                for c in (initial_context or []):
                    cid = c.get('id')
                    if not cid:
                        continue
                    decision_targets.append({"clip_id": cid, "refinement_params": {"new_sampling_rate_per_30s": new_frames, "run_ocr": True, "run_det": False}})
                return {"status": "refine", "targets": decision_targets, "scores": scores, "temporal_sequence_incomplete": False, "temporal_focus_clip_id": ""}

        # If numeric evidence was signalled but we have a temporal focus in play, prefer temporal handling.
        if (bool(evaluation.get("numeric_evidence_required")) or self._is_numeric_question(query)) and not FORCE_REFINE and temporal_flag:
            print("[Refiner] Numeric evidence requested but temporal_sequence_incomplete=True -> defer numeric-all refine to temporal path.")

        # 当 total > 5 且没有显式需要（ocr/det/numeric/temporal）时，直接 final
        if not FORCE_REFINE and total > 5:
            explicit_need = False
            if evaluation.get("numeric_evidence_required") or self._is_numeric_question(query):
                explicit_need = True
            for t in (evaluation.get("refinement_targets") or []):
                ch = (t or {}).get("checks", {}) or {}
                if ch.get("ocr_needed") or ch.get("det_needed"):
                    explicit_need = True
                    break
            if not explicit_need:
                hints_tmp = self._infer_modalities_from_query(query)
                if hints_tmp.get("ocr") or hints_tmp.get("det"):
                    explicit_need = True
            if not explicit_need:
                return {"status": "final", "targets": [], "scores": scores, "temporal_sequence_incomplete": False, "temporal_focus_clip_id": ""}

        # 构建决策Targets（默认或显式targets）
        targets_eval = evaluation.get("refinement_targets", []) or []
        decision_targets: List[Dict[str, Any]] = []
        if not targets_eval:
            hints = self._infer_modalities_from_query(query)
            run_ocr = bool(hints.get("ocr")) or FORCE_REFINE
            run_det = bool(hints.get("det")) or FORCE_REFINE
            for c in initial_context or []:
                cid = c.get("id")
                if cid:
                    decision_targets.append({"clip_id": cid, "refinement_params": {"new_sampling_rate_per_30s": frames_for_total(total), "run_ocr": run_ocr, "run_det": run_det}})
        else:
            for t in targets_eval:
                cid = t.get("clip_id")
                if not cid:
                    continue
                checks = t.get("checks", {}) or {}
                hints = self._infer_modalities_from_query(query)
                run_ocr = bool(checks.get("ocr_needed")) or (FORCE_REFINE or hints.get("ocr", False))
                run_det = bool(checks.get("det_needed")) or (FORCE_REFINE or hints.get("det", False))
                decision_targets.append({"clip_id": cid, "refinement_params": {"new_sampling_rate_per_30s": frames_for_total(total), "run_ocr": run_ocr, "run_det": run_det}})

        return {"status": "refine", "targets": decision_targets, "scores": scores, "temporal_sequence_incomplete": False, "temporal_focus_clip_id": ""}

    async def plan(self, query: str, initial_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 再次显式打印 caption 统计（与 _build_evaluation_prompt 内的统计并存，确保用户可见）
        try:
            _caps = [(c.get('id'), (c.get('caption') or '').strip()) for c in (initial_context or [])]
            _non_empty = [cid for cid, txt in _caps if txt]
            _total_chars = sum(len(txt) for _, txt in _caps if txt)
            _avg = (_total_chars // len(_non_empty)) if _non_empty else 0
            print(f"[Refiner-Caption][PreEval] clips={len(_caps)} non_empty={len(_non_empty)} total_chars={_total_chars} avg_non_empty_chars={_avg}")
        except Exception:
            pass
        evaluation = await self._evaluate_context(query, initial_context)
        decision = self._decide(evaluation, query, initial_context)
        # Observability: concise log snapshot for analysis
        try:
            # Always print evaluation scores for debugging/observability
            _sc = decision.get('scores', {}) or {}
            print(
                f"[Refiner] Scores: answerability={_sc.get('answerability', 0)} "
                f"density={_sc.get('density', 0)} total={_sc.get('total', 0)} (final if total > 5; threshold=5)"
            )
            print(
                f"[Refiner] Decision: status={decision.get('status')}. Num targets: {len(decision.get('targets', []))}. Parse error: {evaluation.get('parse_error', False)}"
            )
            if decision.get('status') == 'refine':
                for target in decision.get('targets', []):
                    params = target.get('refinement_params', {})
                    print(
                        f"  - Target: {target.get('clip_id')}, New FPS/30s: {params.get('new_sampling_rate_per_30s')}, OCR: {params.get('run_ocr')}, DET: {params.get('run_det')}"
                    )
        except Exception:
            pass
        # 简洁单行日志（便于统计与快速过滤）
        try:
            numeric_flag = int(bool(evaluation.get('numeric_evidence_required')) or bool(self._is_numeric_question(query)))
        except Exception:
            numeric_flag = int(bool(evaluation.get('numeric_evidence_required')))
        nf = evaluation.get('numeric_focus_clip_id') or ""
        concise = (
            f"temporal={int(decision.get('temporal_sequence_incomplete', False))} "
            f"numeric={numeric_flag} "
            f"numeric_focus={nf} "
            f"status={decision.get('status')} "
            f"targets={len(decision.get('targets', []))} "
            f"total={int((decision.get('scores') or {}).get('total', 0))}"
        )
        print(concise)
        return decision
