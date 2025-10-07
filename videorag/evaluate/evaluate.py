import os
import json
import glob
import subprocess
import re
import difflib
import time
from typing import Any, Dict, List, Optional, Tuple
import argparse
import logging

# Suppress noisy warnings from sentence_transformers
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)

try:
    # Suppress excessive warnings from transformers
    from transformers.utils.logging import set_verbosity_error
    set_verbosity_error()
    # Use the same default model resolver as generation
    from .._llm import get_default_ollama_chat_model
except Exception:  # pragma: no cover
    def get_default_ollama_chat_model() -> str:  # type: ignore
        return os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5vl:7b")


try:
    from .config import (
        INPUT_BASE_DIR,
        OUTPUT_BASE_DIR,
        OUTPUT_ROOT_DIR,
        DATA_ROOT_DIR,
        DEFAULT_OLLAMA_DIR,
        DEFAULT_HF_HOME,
        EVAL_LLM_MODEL,
        OLLAMA_MODEL,
        MODELSCOPE_BASE_DIR,
        MODELSCOPE_ROBERTA_DIR,
        _resolve_default_bertscore_model,
        FORCED_ST_MODEL,
        DEFAULT_BERTSCORE_MODEL,
        DEFAULT_BERTSCORE_BATCH,
    )
    from .prompts import (
        EVAL_PROMPT_TEMPLATE,
    )
except Exception:
    # Allow running as a standalone script: import from the same folder
    try:
        from config import (
            INPUT_BASE_DIR,
            OUTPUT_BASE_DIR,
            OUTPUT_ROOT_DIR,
            DATA_ROOT_DIR,
            DEFAULT_OLLAMA_DIR,
            DEFAULT_HF_HOME,
            EVAL_LLM_MODEL,
            OLLAMA_MODEL,
            MODELSCOPE_BASE_DIR,
            MODELSCOPE_ROBERTA_DIR,
            _resolve_default_bertscore_model,
            FORCED_ST_MODEL,
            DEFAULT_BERTSCORE_MODEL,
            DEFAULT_BERTSCORE_BATCH,
        )
        from prompts import (
            EVAL_PROMPT_TEMPLATE,
        )
    except Exception:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from config import (
            INPUT_BASE_DIR,
            OUTPUT_BASE_DIR,
            OUTPUT_ROOT_DIR,
            DATA_ROOT_DIR,
            DEFAULT_OLLAMA_DIR,
            DEFAULT_HF_HOME,
            EVAL_LLM_MODEL,
            OLLAMA_MODEL,
            MODELSCOPE_BASE_DIR,
            MODELSCOPE_ROBERTA_DIR,
            _resolve_default_bertscore_model,
            FORCED_ST_MODEL,
            DEFAULT_BERTSCORE_MODEL,
            DEFAULT_BERTSCORE_BATCH,
        )
        from prompts import (
            EVAL_PROMPT_TEMPLATE,
        )

# Paths are provided strictly by evaluate.config (which reads videorag._config)


# Prompts are now imported from .prompts without any content changes


# ---------------- Preflight & Utilities ----------------
def _looks_like_sentence_transformer(path_or_name: str) -> bool:
    low = (path_or_name or "").lower()
    if "sentence-transformers" in low or "minilm" in low or "mpnet" in low:
        return True
    # local dir with modules.json is a strong hint of ST
    try:
        return os.path.isdir(path_or_name) and os.path.exists(os.path.join(path_or_name, "modules.json"))
    except Exception:
        return False


def preflight_checks() -> None:
    """Lightweight dependency and model sanity checks; prints clear guidance.
    - Ensures ollama is present (best-effort)
    - Ensures at least one ROUGE backend exists
    - Ensures bert_score is importable and the chosen model is supported (non ST)
    - If an unsupported BERTScore model is detected, override to a safe default
    """
    import shutil

    if shutil.which("ollama") is None:
        print("[Preflight Warn] 'ollama' binary not found in PATH. LLM evaluation may fail.")

    # ROUGE availability
    rouge_ok = False
    try:
        from rouge_score.rouge_scorer import RougeScorer  # type: ignore
        rouge_ok = True
    except Exception:
        try:
            from rouge import Rouge  # type: ignore
            rouge_ok = True
        except Exception:
            pass
    if not rouge_ok:
        print("[Preflight Warn] Neither 'rouge-score' nor 'rouge' is available. ROUGE-L will be skipped.")

    # BERTScore import and model sanity
    try:
        import bert_score  # type: ignore
    except Exception as e:
        print(f"[Preflight Error] 'bert_score' is not installed: {e}. BERTScore will be skipped.")

    eff_model = os.environ.get("BERTSCORE_MODEL", DEFAULT_BERTSCORE_MODEL)
    if _looks_like_sentence_transformer(eff_model):
        # Validate local sentence-transformers model is loadable
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _ = SentenceTransformer(eff_model)
            print(f"[Preflight OK] sentence-transformers model ready: {eff_model}")
        except Exception as e:
            print(f"[Preflight Error] Failed to load sentence-transformers model at '{eff_model}': {e}")


# ---------------- Utilities ----------------
def safe_len(obj: Optional[List[Any]]) -> int:
    return len(obj) if isinstance(obj, list) else 0


def build_eval_input(question: str, keypoints: Dict[str, List[str]], model_answer: str) -> Dict[str, Any]:
    return {
        "question": question or "",
        "ground_truth_keypoints": {
            "video": keypoints.get("video", []) if isinstance(keypoints, dict) else [],
            "text": keypoints.get("text", []) if isinstance(keypoints, dict) else [],
        },
        "model_answer": model_answer or "",
    }


def normalize_keypoints(raw_keypoints: Any, gt_path: str) -> Dict[str, List[str]]:
    """Normalize keypoints to a dict with 'video' and 'text' lists.

    - For files like tsingleQA.json: treat list as text keypoints
    - For files like vsingleQA.json: treat list as video keypoints
    - For standard multiQA: expect {'video': [...], 'text': [...]} dict
    """
    video_list: List[str] = []
    text_list: List[str] = []
    if isinstance(raw_keypoints, dict):
        v = raw_keypoints.get("video", [])
        t = raw_keypoints.get("text", [])
        video_list = v if isinstance(v, list) else []
        text_list = t if isinstance(t, list) else []
    elif isinstance(raw_keypoints, list):
        base = os.path.basename(gt_path).lower()
        if "tsingleqa" in base:
            text_list = raw_keypoints
        elif "vsingleqa" in base:
            video_list = raw_keypoints
        else:
            # Default to text-only if unknown single format
            text_list = raw_keypoints
    return {"video": video_list, "text": text_list}


def call_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout_sec: int = 240, retries: int = 2) -> str:
    env = os.environ.copy()
    env.setdefault("OLLAMA_MODELS", DEFAULT_OLLAMA_DIR)
    env.setdefault("OLLAMA_NUM_GPU", "1")
    env.setdefault("OLLAMA_KEEP_ALIVE", "5m")

    for attempt in range(retries + 1):
        try:
            proc = subprocess.run(
                ["ollama", "run", model],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
                env=env,
            )
            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()

            if proc.returncode == 0 and stdout:
                return stdout
            if proc.returncode == 0 and not stdout and stderr:
                return stderr

            error_message = (
                f"Attempt {attempt + 1}/{retries + 1} failed. RC={proc.returncode}. "
                f"stderr: '{stderr[:200]}...'"
            )
            print(f"[Ollama Call Warn] {error_message}")
            if attempt < retries:
                time.sleep(5)
                continue
            else:
                return f"[ollama_error]{error_message}"
        except subprocess.TimeoutExpired:
            error_message = f"Attempt {attempt + 1}/{retries + 1} timed out after {timeout_sec}s."
            print(f"[Ollama Call Warn] {error_message}")
            if attempt < retries:
                time.sleep(5)
                continue
            else:
                return f"[ollama_error]{error_message}"
        except Exception as e:
            error_message = f"Attempt {attempt + 1}/{retries + 1} failed with an unexpected error: {e}"
            # Don't retry on unexpected errors
            return f"[ollama_error]{error_message}"
    return "[ollama_error]Exhausted all retries."


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts a JSON object from a string, tolerating markdown code fences
    and other surrounding text. Also attempts to fix common JSON errors.
    """
    if not text:
        return None

    # Remove DeepSeek R1 thinking tags if present
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    
    # Strip common thinking/noise lines
    text = re.sub(r"^(?:thinking\.|done thinking\.|\s)+$", "", text, flags=re.IGNORECASE | re.MULTILINE)

    # 1) Prefer markdown-fenced JSON
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try cleanup then continue to general search
            candidate = re.sub(r",\s*([\}\]])", r"\1", candidate)
            try:
                return json.loads(candidate)
            except Exception:
                pass

    # 2) General scan: find all balanced top-level {...} blocks, try those that look like our schema
    def _balanced_objects(s: str) -> List[str]:
        objs: List[str] = []
        stack = 0
        start = -1
        for i, ch in enumerate(s):
            if ch == '{':
                if stack == 0:
                    start = i
                stack += 1
            elif ch == '}':
                if stack > 0:
                    stack -= 1
                    if stack == 0 and start != -1:
                        objs.append(s[start:i+1])
                        start = -1
        return objs

    candidates = _balanced_objects(text)
    # Heuristic: prioritize candidates that contain our expected keys
    def _score(c: str) -> int:
        score = 0
        for key in ("\"likert_score\"", "\"factuality_analysis\"", "\"likert_subscores\""):
            if key in c:
                score += 1
        return score
    candidates.sort(key=_score, reverse=True)

    for cand in candidates:
        # Skip obvious LaTeX or non-JSON like {Likert Score} formula
        if re.search(r"\{\s*[A-Za-z][^:\}\n]*\}\s*=", cand):
            continue
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # Clean common issues and retry
            cleaned = re.sub(r",\s*([\}\]])", r"\1", cand)
            # Replace fancy quotes with normal quotes
            cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    # 3) No valid JSON found
    print(f"[JSON Warn] Could not parse any JSON object from text: {text[:300]}...")
    return None


def compute_recall_precision_f1(
    covered_video: int,
    covered_text: int,
    total_claims: int,
    gt_video_count: int,
    gt_text_count: int,
) -> Tuple[float, float, float, float, float]:
    m_total = max(0, int(covered_video)) + max(0, int(covered_text))
    n_total = max(0, int(gt_video_count)) + max(0, int(gt_text_count))

    recall_video = (covered_video / gt_video_count) if gt_video_count > 0 else 0.0
    recall_text = (covered_text / gt_text_count) if gt_text_count > 0 else 0.0
    recall_overall = (m_total / n_total) if n_total > 0 else 0.0

    precision_overall = (m_total / total_claims) if total_claims > 0 else 0.0
    if (precision_overall + recall_overall) > 0:
        f1_overall = 2 * precision_overall * recall_overall / (precision_overall + recall_overall)
    else:
        f1_overall = 0.0

    return recall_video, recall_text, recall_overall, precision_overall, f1_overall


def try_rouge_l_f(reference: str, candidate: str) -> Optional[float]:
    try:
        # Prefer rouge-score
        from rouge_score.rouge_scorer import RougeScorer  # type: ignore

        scorer = RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference or "", candidate or "")
        return float(scores["rougeL"].fmeasure)
    except Exception:
        try:
            # Fallback to "rouge" package if available
            from rouge import Rouge  # type: ignore

            r = Rouge()
            out = r.get_scores(candidate or "", reference or "", avg=True)
            return float(out.get("rouge-l", {}).get("f", 0.0))
        except Exception:
            return None


def try_bertscore_f1(reference: str, candidate: str) -> Optional[float]:
    """Compute text similarity score.
    - If BERTSCORE_MODEL points to a sentence-transformers model (local path or name), use ST cosine mapped to [0,1].
    - Else, use official bert_score with the provided model.
    """
    if os.environ.get("DISABLE_BERTSCORE", "").lower() in ("1", "true", "yes"):
        return None
    ref = (reference or "").strip()
    cand = (candidate or "").strip()
    if not ref or not cand:
        return 0.0

    model_type = os.environ.get("BERTSCORE_MODEL", DEFAULT_BERTSCORE_MODEL)
    if _looks_like_sentence_transformer(model_type):
        # SentenceTransformer branch
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            # cache model per path
            cache_key = f"st::{model_type}"
            if not hasattr(try_bertscore_f1, "_st_cache"):
                try_bertscore_f1._st_cache = {}  # type: ignore[attr-defined]
            _cache = getattr(try_bertscore_f1, "_st_cache")  # type: ignore[attr-defined]
            model = _cache.get(cache_key)
            if model is None:
                print(f"[BERTScore-ST] Loading SentenceTransformer: {model_type}")
                model = SentenceTransformer(model_type)
                _cache[cache_key] = model
            embs = model.encode([cand, ref], normalize_embeddings=True, show_progress_bar=False)
            sim = float((embs[0] * embs[1]).sum())  # cosine since normalized
            score = (sim + 1.0) / 2.0
            return score
        except Exception as e:
            print(f"[BERTScore-ST Error] {e}")
            return None

    try:
        from bert_score import score  # type: ignore
        import torch  # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[BERTScore] Using model_type: {model_type} on device {device}")
        _, _, f1 = score(
            [cand], [ref],
            model_type=model_type,
            verbose=False,
            device=device,
            batch_size=DEFAULT_BERTSCORE_BATCH,
            idf=False,
        )
        return float(f1.mean().item())
    except Exception as e:
        print(f"[BERTScore Error] Failed with model '{model_type}': {e}")
        return None


# 兼容包内调用与脚本直跑：优先相对导入，失败时回退为同目录绝对导入
try:
    from .keypoint_match import heuristic_match_keypoints  # 拆分后的启发式匹配模块
except Exception:  # pragma: no cover
    try:
        # 当以 `python evaluate.py` 在本目录直跑时
        from keypoint_match import heuristic_match_keypoints  # type: ignore
    except Exception:
        # 兜底：显式将当前文件目录加入 sys.path 再导入
        import sys as _sys, os as _os  # type: ignore
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from keypoint_match import heuristic_match_keypoints  # type: ignore


def evaluate_one(
    question: str,
    keypoints: Dict[str, List[str]],
    model_answer: str,
    gt_answer: str,
) -> Dict[str, Any]:
    eval_input = build_eval_input(question, keypoints, model_answer)

    # 视觉差异 bullet 聚合：在普通模式下（无外部差异数据可引用）尝试生成“未利用”的 video keypoint 提示，
    # 供打分模型感知视觉信息缺失，提高 visual_detail_usage 判别灵敏度。
    video_kps = keypoints.get("video", []) if isinstance(keypoints, dict) else []
    lower_answer = (model_answer or "").lower()
    gap_bullets: List[str] = []
    for kp in video_kps:
        # 简单启发：若视频关键点中的任一连续4+字符子串不在答案中，则视为缺失。
        token = str(kp).strip()
        if not token:
            continue
        core = token[:40].lower()
        # 使用一个代表性片段（去掉数字标号等）
        core = re.sub(r"^[\d\-\.\)\(\s]+", "", core)
        core = re.sub(r"[^a-z0-9\u4e00-\u9fa5 ]+", " ", core)
        core = re.sub(r"\s+", " ", core).strip()
        if len(core) >= 4 and core not in lower_answer:
            gap_bullets.append(f"Missing visual detail: {core}")
        if len(gap_bullets) >= 6:  # 限制长度，避免 prompt 膨胀
            break
    if gap_bullets:
        eval_input["visual_gap_bullets"] = gap_bullets

    prompt = EVAL_PROMPT_TEMPLATE.replace("{test_input}", json.dumps(eval_input, ensure_ascii=False, indent=2))

    raw = call_ollama(prompt)
    parsed = extract_json_object(raw)

    # Retry once with the same evaluation prompt if JSON parsing failed
    if not isinstance(parsed, dict):
        print(f"[Warn] Initial JSON parsing failed. Retrying once with the evaluation prompt. Raw output: {raw[:500]}...")
        prompt = EVAL_PROMPT_TEMPLATE.replace("{test_input}", json.dumps(eval_input, ensure_ascii=False, indent=2))
        raw = call_ollama(prompt)
        parsed = extract_json_object(raw)
        if not isinstance(parsed, dict):
            print(f"[Error] JSON parsing failed on second attempt. Raw output: {raw[:500]}...")


    covered_video = 0
    covered_text = 0
    total_claims = 0
    likert = None
    likert_subscores = {}
    reasoning = ""

    if isinstance(parsed, dict):
        likert = parsed.get("likert_score")
        reasoning = parsed.get("reasoning", "")
        analysis = parsed.get("factuality_analysis", {})
        if isinstance(analysis, dict):
            covered_video = analysis.get("covered_video_keypoints", 0)
            covered_text = analysis.get("covered_text_keypoints", 0)
            total_claims = analysis.get("total_claimed_keypoints", 0)
        # 解析子评分；若缺失则回退为空
        subs = parsed.get("likert_subscores", {})
        if isinstance(subs, dict):
            for k in ("factual_coverage", "visual_detail_usage", "linguistic_precision"):
                v = subs.get(k)
                if isinstance(v, int):
                    likert_subscores[k] = v
        # 若主 likert 缺失而子项存在，用平均生成
        if (likert is None or not isinstance(likert, int)) and likert_subscores:
            vals = [v for v in likert_subscores.values() if isinstance(v, int)]
            if vals:
                likert = int(round(sum(vals) / len(vals)))
    else:
        # This branch is taken if JSON parsing fails completely.
        # All metrics will be 0 or default, which is a strong penalty.
        print(f"[Error] Could not parse LLM evaluation output. Treating as zero score.")

    gt_video_n = safe_len(keypoints.get("video")) if isinstance(keypoints, dict) else 0
    gt_text_n = safe_len(keypoints.get("text")) if isinstance(keypoints, dict) else 0

    rouge_l_f = try_rouge_l_f(gt_answer or "", model_answer or "")
    sim_score = try_bertscore_f1(gt_answer or "", model_answer or "")
    # 仅当选择了 sentence-transformers 模型时，输出 st_cosine_score；否则置为 None
    _eff_model = os.environ.get("BERTSCORE_MODEL", DEFAULT_BERTSCORE_MODEL)
    st_cosine_score = sim_score if _looks_like_sentence_transformer(_eff_model) else None

    # 采用本地启发式匹配修正 keypoint 覆盖数
    h_video, h_text, kp_debug = heuristic_match_keypoints(keypoints, model_answer)

    return {
        # 使用启发式覆盖（保留 LLM 原结果供对比）
        "covered_video_keypoints": int(h_video),
        "covered_text_keypoints": int(h_text),
        "llm_covered_video_keypoints": int(covered_video),
        "llm_covered_text_keypoints": int(covered_text),
        "gt_video_n": int(gt_video_n),
        "gt_text_n": int(gt_text_n),
        "total_claimed_keypoints": int(total_claims),
    "likert_score": likert if likert is not None else 0,
    "likert_subscores": likert_subscores,
    "rouge_l_f": rouge_l_f if rouge_l_f is not None else None,
    "st_cosine_score": st_cosine_score if st_cosine_score is not None else None,
        "eval_reasoning": reasoning,
        "keypoint_unmatched_debug": kp_debug,  # 未匹配原因 debug
    }

def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def index_data_by_question(data_root: str) -> Dict[str, Dict[str, Any]]:
    """Scan all JSON files under data_root and index GT by exact question string.

    Returns a mapping: question -> {"answer": str, "keypoints": dict, "source_path": str}
    """
    index: Dict[str, Dict[str, Any]] = {}

    # Prepare candidate roots to scan: primary data_root, then common fallbacks
    roots_to_scan: List[str] = []
    if data_root:
        roots_to_scan.append(data_root)
    # Add DATA_ROOT_DIR (strict) as a candidate GT root
    try:
        bench_path = DATA_ROOT_DIR
        if bench_path not in roots_to_scan:
            roots_to_scan.append(bench_path)
    except Exception:
        pass
    # Also try INPUT_BASE_DIR as another possible source
    try:
        if INPUT_BASE_DIR and INPUT_BASE_DIR not in roots_to_scan:
            roots_to_scan.append(INPUT_BASE_DIR)
    except Exception:
        pass

    all_gt_files: List[str] = []
    scanned_roots: List[str] = []
    for root in roots_to_scan:
        if not root:
            continue
        if os.path.isdir(root):
            scanned_roots.append(root)
            found = glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
            all_gt_files.extend(found)

    # Deduplicate
    gt_files = sorted(set(all_gt_files))
    if scanned_roots:
        print(f"[Evaluate] Data JSONs discovered: {len(gt_files)} from roots: {scanned_roots}")
    else:
        print(f"[Evaluate] No data roots found among: {roots_to_scan}")

    for path in gt_files:
        try:
            data = load_json_file(path)
        except Exception as e:
            print(f"[Evaluate] Fail(read Data): {path}: {e}")
            continue

        items = data if isinstance(data, list) else [data]
        for entry in items:
            if not isinstance(entry, dict):
                continue
            # Some GT files might use 'question' at top-level or nested fields
            q = str(entry.get("question", "")).strip()
            if not q:
                continue
            # Prefer 'answer' but fall back to other keys if present
            ans = entry.get("answer", entry.get("ans", ""))
            ans = str(ans) if ans is not None else ""
            # Accept both 'keypoints' and 'keypoint' and normalize
            raw_keypoints = entry.get("keypoints", entry.get("keypoint", {}))
            kp = normalize_keypoints(raw_keypoints, path)
            index[q] = {"answer": ans, "keypoints": kp, "source_path": path}

    return index


def _normalize_text_for_match(s: str) -> str:
    if not s:
        return ""
    # Lowercase, remove punctuation (keep CJK chars and word chars), collapse spaces
    t = s.lower()
    t = re.sub(r"[^\\w\u4e00-\u9fff]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main(args: argparse.Namespace) -> None:
    # Preflight dependency/model checks
    preflight_checks()
    # 递归扫描 OUTPUT_ROOT_DIR 下所有子目录，收集所有 .json 文件（不再根据文件名做严格过滤）
    # Use OUTPUT_ROOT_DIR (strict) as results root
    results_root = OUTPUT_ROOT_DIR
    if not results_root:
        raise RuntimeError("OUTPUT_ROOT_DIR (results root) is not configured")
    candidate_files = glob.glob(os.path.join(results_root, "**", "*.json"), recursive=True)
    out_files = list(candidate_files)
    if not out_files:
        print(f"[Evaluate] No result files found under {results_root}")
        return

    total_files = len(out_files)
    files_updated = 0
    files_skipped = 0
    files_failed = 0
    print(f"[Evaluate] Files discovered: {total_files}")

    for out_path in out_files:
        # 1) Read result first and decide whether to skip (already evaluated)
        try:
            res_data = load_json_file(out_path)
        except Exception as e:
            print(f"[Evaluate] Fail(read): {out_path}: {e}")
            files_failed += 1
            continue

        if not args.force:
            # Skip only if all items in the file are fully and successfully evaluated
            is_fully_evaluated = True
            res_list_preview = res_data if isinstance(res_data, list) else [res_data]

            if not res_list_preview or not any(isinstance(i, dict) for i in res_list_preview):
                is_fully_evaluated = False
            else:
                for item in res_list_preview:
                    if not isinstance(item, dict):
                        continue  # Ignore non-item entries like a potential summary

                    # An item is considered faulty and needs re-evaluation if:
                    # - missing core evaluation fields (we ignore BERTScore entirely), or
                    # - eval_reasoning indicates an error
                    reasoning = str(item.get("eval_reasoning", "")).strip()
                    required_fields = ["likert_score", "eval_reasoning"]
                    missing_required = any(k not in item for k in required_fields)
                    is_faulty = missing_required or reasoning.startswith(("[no_json]", "[ollama_error]"))

                    if is_faulty:
                        is_fully_evaluated = False
                        print(f"[Evaluate] Re-evaluating {os.path.basename(out_path)} due to faulty item (index: {item.get('index', 'N/A')})")
                        break

            if is_fully_evaluated:
                print(f"[Evaluate] Skip (fully evaluated): {os.path.relpath(out_path, OUTPUT_ROOT_DIR)}")
                files_skipped += 1
                continue

        # 2) Evaluate by matching questions against Data index
        res_list = res_data if isinstance(res_data, list) else [res_data]

        # Build index mapping for safety
        cnt_items = 0
        cnt_rouge = 0

        for i, item in enumerate(res_list):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", ""))
            if not question:
                print(f"[Evaluate] Skip item without question in {os.path.basename(out_path)} (i={i})")
                continue

            # Lookup GT by exact question match
            # Build the index once outside the loop for efficiency
            if i == 0:
                # Lazy init: build once and attach to function state to avoid rebuilding per file
                pass
            # Access global-like cache via function attribute to avoid recomputation
            if not hasattr(main, "_gt_index"):
                main._gt_index = index_data_by_question(DATA_ROOT_DIR)  # type: ignore[attr-defined]
            gt_index = getattr(main, "_gt_index")  # type: ignore[attr-defined]

            gt_entry = gt_index.get(question)
            if not gt_entry:
                # Attempt fuzzy matching against indexed GT questions
                norm_q = _normalize_text_for_match(question)
                candidates = list(gt_index.keys())
                # Build normalized mapping lazily
                if not hasattr(main, "_norm_q_map"):
                    main._norm_q_map = {k: _normalize_text_for_match(k) for k in candidates}  # type: ignore[attr-defined]
                norm_map = getattr(main, "_norm_q_map")  # type: ignore[attr-defined]
                # Use difflib to find the closest normalized matches
                choices = list(norm_map.values())
                if choices:
                    close = difflib.get_close_matches(norm_q, choices, n=1, cutoff=0.75)
                    if close:
                        # find original question key
                        matched_norm = close[0]
                        orig_q = next((k for k, v in norm_map.items() if v == matched_norm), None)
                        if orig_q:
                            gt_entry = gt_index.get(orig_q)
                if not gt_entry:
                    print(f"[Evaluate] GT not found for question (file={os.path.basename(out_path)}): {question[:80]}...")
                    continue

            # If not forced, skip items already evaluated successfully
            if not args.force:
                reasoning_existing = str(item.get("eval_reasoning", "")).strip()
                required_fields_item = [
                    "covered_video_keypoints",
                    "covered_text_keypoints",
                    "gt_video_n",
                    "gt_text_n",
                    "total_claimed_keypoints",
                    "likert_score",
                    "eval_reasoning",
                ]
                missing_required_item = any(k not in item for k in required_fields_item)
                has_error_flag = reasoning_existing.startswith(("[no_json]", "[ollama_error]"))
                if not missing_required_item and not has_error_flag:
                    # Already has evaluation, keep as-is
                    continue

            keypoints = gt_entry.get("keypoints", {"video": [], "text": []})
            gt_answer = str(gt_entry.get("answer", ""))
            model_answer = str(item.get("answer", ""))

            metrics = evaluate_one(question, keypoints, model_answer, gt_answer)

            # Write back normalized counts and metrics for downstream aggregation
            item.update({
                "covered_video_keypoints": int(metrics["covered_video_keypoints"]),
                "covered_text_keypoints": int(metrics["covered_text_keypoints"]),
                "gt_video_n": int(metrics["gt_video_n"]),
                "gt_text_n": int(metrics["gt_text_n"]),
                "total_claimed_keypoints": int(metrics["total_claimed_keypoints"]),
                "likert_score": int(metrics["likert_score"]),
                "likert_subscores": metrics.get("likert_subscores", {}),
                "rouge_l_f": (round(metrics["rouge_l_f"], 6) if isinstance(metrics["rouge_l_f"], float) else None),
                "st_cosine_score": (round(metrics["st_cosine_score"], 6) if isinstance(metrics.get("st_cosine_score"), float) else None),
                "eval_reasoning": metrics["eval_reasoning"],
            })

            if isinstance(metrics["rouge_l_f"], float):
                cnt_rouge += 1
            cnt_items += 1

        save_json_file(out_path, res_list)
        print(
            f"[Evaluate] Done: {os.path.relpath(out_path, OUTPUT_ROOT_DIR)} "
            f"items={cnt_items} rouge={cnt_rouge}"
        )
        files_updated += 1

    print(
        f"[Evaluate] Summary: total={total_files} updated={files_updated} "
        f"skipped={files_skipped} failed={files_failed}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on model outputs.")
    parser.add_argument(
        "-f", "--f", "--force",
        dest="force",
        action="store_true",
        help="Force re-evaluation of all result files, overwriting existing evaluations."
    )
    args = parser.parse_args()
    main(args)


