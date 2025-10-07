import os
import re
from typing import Any, Dict, List, Optional, Tuple

_MONTH_MAP = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april", "may": "may", "jun": "june",
    "jul": "july", "aug": "august", "sep": "september", "sept": "september", "oct": "october",
    "nov": "november", "dec": "december"
}

_PUNCT_RE = re.compile(r"[\p{Punct}]+", re.UNICODE) if hasattr(re, "_pattern_type") else re.compile(r"[\W_]+", re.UNICODE)
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b")

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    # 去除标点与符号(保留数字/英文/常用中文与空格) —— Python re 不支持 \p{..}，使用兼容写法
    s = re.sub(r"[^0-9a-z\u4e00-\u9fa5\s]", " ", s)
    # 数字序数 -> 数字
    s = _ORDINAL_RE.sub(lambda m: m.group(1), s)
    # 月份统一
    def _month_replace(match: re.Match) -> str:
        w = match.group(0)
        return _MONTH_MAP.get(w[:3], w)
    s = re.sub(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", _month_replace, s)
    # 多空格折叠
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize(s: str) -> List[str]:
    return [t for t in s.split() if t]

def _token_sets(text: str) -> Tuple[List[str], set]:
    norm = _normalize_text(text)
    toks = _tokenize(norm)
    return toks, set(toks)

def _try_sentence_embedding_similarity(kp: str, answer: str) -> Optional[float]:
    """可选：若已安装 sentence_transformers，快速计算余弦。失败返回 None。"""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np  # type: ignore
        # 优先使用 central config 中指定的 sentence-transformers 本地路径或名称
        try:
            from videorag._config import SENT_TRANSFORMER_MODEL_PATH
        except Exception:
            SENT_TRANSFORMER_MODEL_PATH = None
        model_name = os.environ.get("KP_EMBED_MODEL", SENT_TRANSFORMER_MODEL_PATH or "all-MiniLM-L6-v2")
        if not hasattr(_try_sentence_embedding_similarity, "_model") or getattr(_try_sentence_embedding_similarity, "_loaded_name", "") != model_name:  # type: ignore[attr-defined]
            _try_sentence_embedding_similarity._model = SentenceTransformer(model_name)  # type: ignore[attr-defined]
            _try_sentence_embedding_similarity._loaded_name = model_name  # type: ignore[attr-defined]
        model = _try_sentence_embedding_similarity._model  # type: ignore[attr-defined]
        embs = model.encode([kp, answer], normalize_embeddings=True, show_progress_bar=False)
        return float((embs[0] * embs[1]).sum())
    except Exception:
        return None

def heuristic_match_keypoints(keypoints: Dict[str, List[str]], answer: str) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    依据模糊规则匹配 keypoints：
      - 统一归一化（lower/去标点/序数化简/月名标准化）
      - Jaccard >= 0.5 或 重叠 token 数 >= ceil(min(len(kp), len(ans)) * 0.6)
      - 备选：若嵌入相似度 >= 0.8 视为匹配（仅在可用时）
      - 每个 keypoint 只计一次
      - 返回未匹配 debug 原因
    返回: (video_matched, text_matched, debug_list)
    """
    ans_tokens, ans_set = _token_sets(answer)
    ans_len = len(ans_tokens)
    debug: List[Dict[str, Any]] = []
    video_matched = 0
    text_matched = 0

    def _check(kp: str) -> Tuple[bool, Dict[str, Any]]:
        kp_tokens, kp_set = _token_sets(kp)
        if not kp_tokens:
            return False, {"kp": kp, "reason": "empty_after_normalize"}
        inter = kp_set & ans_set
        union = kp_set | ans_set if ans_set else kp_set
        jacc = (len(inter) / len(union)) if union else 0.0
        overlap_need = int(__import__("math").ceil(min(len(kp_tokens), ans_len) * 0.6)) if (len(kp_tokens) and ans_len) else 1
        overlap_pass = len(inter) >= overlap_need
        matched = (jacc >= 0.5) or overlap_pass
        emb_sim = None
        if not matched:
            emb_sim = _try_sentence_embedding_similarity(" ".join(kp_tokens), " ".join(ans_tokens))
            if emb_sim is not None and emb_sim >= 0.8:
                matched = True
        if matched:
            return True, {"kp": kp, "jaccard": round(jacc, 4), "overlap": len(inter), "need": overlap_need, "emb": (round(emb_sim,4) if emb_sim is not None else None)}
        else:
            reason = "overlap_low" if ans_len and len(kp_tokens) else "empty"  # 默认
            len_ratio = (min(len(kp_tokens), ans_len) / max(len(kp_tokens), ans_len)) if (kp_tokens and ans_len) else 0.0
            if len_ratio < 0.2 and ans_len and len(kp_tokens):
                reason = "length_diff"
            return False, {
                "kp": kp,
                "reason": reason,
                "jaccard": round(jacc,4),
                "overlap": len(inter),
                "need": overlap_need,
                "len_kp": len(kp_tokens),
                "len_ans": ans_len,
                "len_ratio": round(len_ratio,4),
                "emb": (round(emb_sim,4) if emb_sim is not None else None)
            }

    for domain, lst in ("video", keypoints.get("video", [])), ("text", keypoints.get("text", [])):
        for kp in lst:
            ok, info = _check(kp)
            if ok:
                if domain == "video":
                    video_matched += 1
                else:
                    text_matched += 1
            else:
                debug.append(info)
    return video_matched, text_matched, debug
