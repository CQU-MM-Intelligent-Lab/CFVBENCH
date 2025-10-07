"""Utility functions extracted from `iterative_refinement.py` for frame去重/插值等。
"""

import os
import statistics
from typing import Dict, Any
from PIL import Image
import numpy as np

try:
    import imagehash  # optional
    _IR_HAS_IMAGEHASH = True
except Exception:
    _IR_HAS_IMAGEHASH = False

def _map_score_to_frames(score: float, mapping: dict[int,int]) -> int:
    """
    将打分(可为浮点)映射到目标帧数；score 会 round 到最近整数并 clamp 在映射键范围内。
    """
    if not mapping:
        return 0
    keys_sorted = sorted(mapping.keys())
    s = int(round(score))
    if s < keys_sorted[0]:
        s = keys_sorted[0]
    elif s > keys_sorted[-1]:
        s = keys_sorted[-1]
    return mapping[s]

def _ir_average_hash(img, hash_size: int = 8) -> int:
    """
    简易均值哈希（无 imagehash 时使用）返回 64bit int。
    """
    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = Image.LANCZOS
    g = img.convert("L").resize((hash_size, hash_size), resample)
    arr = np.asarray(g, dtype=np.uint8)
    avg = float(arr.mean())
    bits = (arr > avg).astype(np.uint8).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v

def _ir_hamming_int(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def _ir_frame_hash(path: str):
    """
    返回 (hash_obj_or_int, ok)；失败时 (None, False)
    """
    try:
        with Image.open(path) as im:
            if _IR_HAS_IMAGEHASH:
                return imagehash.phash(im), True
            return _ir_average_hash(im), True
    except Exception:
        return None, False

def _ir_hash_distance(h1, h2) -> int:
    if h1 is None or h2 is None:
        return 9999
    if _IR_HAS_IMAGEHASH:
        try:
            return int(h1 - h2)
        except Exception:
            return 9999
    return _ir_hamming_int(h1, h2)

def _dedup_frames(frame_paths: list[str], target_count: int | None = None, threshold: int = 5, debug: bool = False) -> list[str]:
    """
    帧去重 + (可选)补齐:
    - 仅使用阈值 < threshold 判为重复（后者丢弃）
    - target_count=None 时不补齐；若需要补齐由外层插值逻辑负责
    - debug=True 输出距离统计帮助调参
    """
    if not frame_paths or len(frame_paths) == 1:
        return frame_paths

    kept: list[str] = []
    kept_hashes: list = []
    dropped: list[tuple[str, object]] = []
    dist_records: list[int] = []  # 每个非首帧对已保留集合的最小距离

    for p in frame_paths:
        h, ok = _ir_frame_hash(p)
        if not kept:
            kept.append(p)
            kept_hashes.append(h)
            continue
        min_dist = min((_ir_hash_distance(h, kh) for kh in kept_hashes), default=9999) if ok else 9999
        dist_records.append(min_dist)
        if min_dist < threshold:
            dropped.append((p, h))
        else:
            kept.append(p)
            kept_hashes.append(h)

    if debug and dist_records:
        try:
            removed = len(dropped)
            total = len(frame_paths)
            ratio = removed / total
            q = lambda a, p: a[int((len(a)-1)*p)]
            d_sorted = sorted(dist_records)
            stats_line = (
                f"[DeDup-Stats] total={total} kept={len(kept)} removed={removed} "
                f"ratio={ratio:.2%} thr={threshold} "
                f"min/median/max={d_sorted[0]}/{statistics.median(d_sorted)}/{d_sorted[-1]} "
                f"q25={q(d_sorted,0.25)} q75={q(d_sorted,0.75)}"
            )
            print(stats_line)
        except Exception:
            pass

    if target_count is None or len(kept) >= target_count:
        return kept

    if dropped:
        scored = []
        for p, h in dropped:
            if h is None:
                scored.append((0, p))
            else:
                dmin = min((_ir_hash_distance(h, kh) for kh in kept_hashes), default=0)
                scored.append((dmin, p))
        scored.sort(reverse=True, key=lambda x: x[0])
        for _, p in scored:
            if len(kept) >= target_count:
                break
            kept.append(p)
        if len(kept) < target_count:
            for _, p in scored:
                if p in kept:
                    continue
                kept.append(p)
                if len(kept) >= target_count:
                    break

    return kept[:target_count] if target_count else kept

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def _uniform_subsample(paths: list[str], k: int) -> list[str]:
    if k >= len(paths):
        return paths
    if k <= 0:
        return []
    import math
    out = []
    for i in range(k):
        idx = round(i * (len(paths) - 1) / (k - 1)) if k > 1 else 0
        out.append(paths[idx])
    # 去重保持顺序
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    # 若因去重少了再补前面
    i = 0
    while len(uniq) < k and i < len(paths):
        if paths[i] not in seen:
            uniq.append(paths[i]); seen.add(paths[i])
        i += 1
    return uniq[:k]

def _extract_frame_opencv(video_path: str, timestamp: float, out_dir: str, clip_id: str, idx: int) -> str | None:
    try:
        import cv2  # optional
    except Exception:
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, timestamp) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        _ensure_dir(out_dir)
        fname = f"{clip_id}_interp_{idx:04d}.jpg"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, frame)
        return fpath
    except Exception:
        return None

def _interpolate_and_fill_frames(
    clip_id: str,
    dedup_paths: list[str],
    orig_frames: list[tuple[str, float]],
    target_cnt: int,
    segment_meta: dict | None,
    video_path_db: dict | None
) -> list[tuple[str, float]]:
    """
    保证最终等于 target_cnt:
    1. dedup_paths 为去重后路径（仅去重不补齐）
    2. 若数量足够且可能>target -> 下采样
    3. 若不足 -> 尝试时间均匀采样补帧 (OpenCV), 失败回退重复补齐
    """
    if target_cnt is None or target_cnt <= 0:
        # 无目标，直接返回去重保持原时间戳
        ts_map = {}
        for p, ts in orig_frames:
            if p not in ts_map:
                ts_map[p] = ts
        return [(p, ts_map.get(p, 0.0)) for p in dedup_paths]

    # 建立原时间戳映射
    ts_map: dict[str, float] = {}
    for p, ts in orig_frames:
        if p not in ts_map:
            ts_map[p] = ts

    # 如果去重后超过目标数，均匀下采样
    if len(dedup_paths) > target_cnt:
        sampled = _uniform_subsample(dedup_paths, target_cnt)
        return [(p, ts_map.get(p, 0.0)) for p in sampled]

    # 去重后数量正好
    if len(dedup_paths) == target_cnt:
        return [(p, ts_map.get(p, 0.0)) for p in dedup_paths]

    # 去重后不足 -> 时间插值补帧
    need = target_cnt - len(dedup_paths)
    # 估计时间范围
    if segment_meta:
        st = segment_meta.get("start_time")
        et = segment_meta.get("end_time")
    else:
        # fallback: 用已有帧时间范围
        times_existing = [ts_map.get(p, 0.0) for p in dedup_paths]
        if times_existing:
            st, et = min(times_existing), max(times_existing)
        else:
            st, et = 0.0, float(need)  # 伪造
    if not isinstance(st, (int, float)) or not isinstance(et, (int, float)) or et <= st:
        # 注意：此处不能在右值中引用可能为 None 的 st
        # 之前写法 st, et = 0.0, st + max(need, 1) 会在 st 为 None 时触发类型错误
        st, et = 0.0, float(max(need, 1))

    # 已有帧时间戳集合
    exist_time_pairs = [(p, ts_map.get(p, 0.0)) for p in dedup_paths]
    existing_times = [ts for _p, ts in exist_time_pairs]

    # 均匀期望时间点（总 target_cnt 个），选缺失的点去尝试补
    import math
    desired_times = []
    for i in range(target_cnt):
        if target_cnt == 1:
            t = (st + et) / 2.0
        else:
            t = st + (et - st) * i / (target_cnt - 1)
        desired_times.append(t)

    # 判定一个时间是否已有（近邻阈值 = (et-st)/ (target_cnt*4)）
    tol = (et - st) / (target_cnt * 4 + 1e-6)
    missing_times = [t for t in desired_times if all(abs(t - ets) > tol for ets in existing_times)]

    # 仅取需要数
    to_generate = missing_times[:need]

    # 寻找视频路径
    video_path = None
    if segment_meta:
        video_path = segment_meta.get("video_path")
    if (not video_path) and isinstance(video_path_db, dict):
        video_path = video_path_db.get(clip_id)
    # 输出目录：沿用首帧目录或当前工作目录
    if orig_frames:
        base_dir = os.path.dirname(orig_frames[0][0])
    else:
        base_dir = _ensure_dir(os.path.join(os.getcwd(), "refine_frames"))

    generated: list[tuple[str, float]] = []
    if video_path and os.path.exists(video_path):
        gi = 0
        for t in to_generate:
            path_new = _extract_frame_opencv(video_path, t, base_dir, clip_id, gi)
            gi += 1
            if path_new and os.path.exists(path_new):
                generated.append((path_new, t))
            if len(generated) >= need:
                break

    # 如果生成不足，最后回退：再用原（含重复）的 frame_paths 均匀抽样到 target_cnt
    if len(generated) < need:
        fallback = _uniform_subsample([p for p, _ in orig_frames], target_cnt)
        # 去重保顺序
        seen = set()
        merged = []
        for p in fallback:
            if p not in seen:
                merged.append((p, ts_map.get(p, 0.0)))
                seen.add(p)
        # 如果 fallback 覆盖满足 target_cnt 直接返回
        if len(merged) == target_cnt:
            return merged
        # 否则继续用当前 dedup + generated 填满
    # 合并
    final_pairs = exist_time_pairs + generated
    # 根据时间排序
    final_pairs.sort(key=lambda x: x[1])
    # 若仍不足（极端失败），再用已有最后一帧复制时间向后微调
    if len(final_pairs) < target_cnt and final_pairs:
        last_p, last_t = final_pairs[-1]
        missing = target_cnt - len(final_pairs)
        # 使用更明显的增量，避免被下游“桶化”去重
        # 例如：+0.01, +0.02, ...，确保严格单调递增
        step = 1e-2
        for i in range(missing):
            final_pairs.append((last_p, last_t + step * (i + 1)))
    # 裁切或补齐到精确 target_cnt
    if len(final_pairs) > target_cnt:
        # 均匀再采一次
        final_paths = _uniform_subsample([p for p, _ in final_pairs], target_cnt)
        return [(p, ts_map.get(p, 0.0)) for p in final_paths]
    return final_pairs

__all__ = [
    "_map_score_to_frames",
    "_ir_average_hash",
    "_ir_hamming_int",
    "_ir_frame_hash",
    "_ir_hash_distance",
    "_dedup_frames",
    "_ensure_dir",
    "_uniform_subsample",
    "_extract_frame_opencv",
    "_interpolate_and_fill_frames",
]