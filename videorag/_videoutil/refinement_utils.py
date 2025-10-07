import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from PIL import Image
from tqdm import tqdm
import tempfile
import time
from .text_utils import normalize_text, dedupe_texts_preserve_order_lsh, EASYOCR_CONF_THR, OCR_JACCARD_THR

# EasyOCR
try:
    import easyocr  # type: ignore
    _HAS_EASYOCR = True
except Exception:
    _HAS_EASYOCR = False

# OWL-ViT (open-vocabulary detection)
try:
    # Correct class names in transformers: OwlViT*
    from transformers import OwlViTProcessor as _OwlProcessor, OwlViTForObjectDetection as _OwlModel  # type: ignore
    _HAS_OWLVIT = True
except Exception:
    try:
        # Backward-compat alias just in case some environments expose OWLVit*
        from transformers import OWLVitProcessor as _OwlProcessor, OWLVitForObjectDetection as _OwlModel  # type: ignore
        _HAS_OWLVIT = True
    except Exception:
        _HAS_OWLVIT = False


def _sample_frames(
    video_path: str, 
    start: float, 
    end: float, 
    num_frames: int, 
    exclude_timestamps: Optional[List[float]] = None
) -> List[Image.Image]:
    if num_frames <= 0:
        return []
    
    # Generate potential frame timestamps
    frame_times = np.linspace(start, end, num_frames, endpoint=False)
    
    # If there are timestamps to exclude, filter them out
    if exclude_timestamps and len(exclude_timestamps) > 0:
        # For each new timestamp, check if it's too close to any excluded timestamp
        # A simple threshold of 0.1s should be sufficient to avoid duplicates
        final_times = []
        for t_new in frame_times:
            is_too_close = False
            for t_old in exclude_timestamps:
                if abs(t_new - t_old) < 0.1:
                    is_too_close = True
                    break
            if not is_too_close:
                final_times.append(t_new)
        frame_times = np.array(final_times)

    if len(frame_times) == 0:
        return []

    # Use ffmpeg to extract individual frames at target timestamps. This
    # avoids moviepy's global ffmpeg invocation which can emit AV1 decoder
    # noise. Import run_ffmpeg_with_fallback lazily to avoid top-level import
    # resolution issues in some environments.
    pil_frames: List[Image.Image] = []
    with tempfile.TemporaryDirectory() as td:
        # lazy import to avoid package resolution problems at module import
        try:
            from avr.test_media_utils import run_ffmpeg_with_fallback
        except Exception:
            # best-effort: if helper not importable, fall back to calling ffmpeg directly
            run_ffmpeg_with_fallback = None
        for idx, t in enumerate(frame_times):
            out_path = os.path.join(td, f"frame_{idx:04d}.jpg")
            # Use fast seek (-ss before -i) for per-frame extraction.
            # Add -hide_banner and -loglevel error to reduce ffmpeg warning spam
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(float(t)),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                out_path,
            ]
            try:
                if run_ffmpeg_with_fallback is not None:
                    res = run_ffmpeg_with_fallback(cmd, fallback_hwaccel=True, verbose=False)
                    if res.returncode != 0 or not os.path.exists(out_path):
                        continue
                else:
                    # direct subprocess fallback (suppress warnings)
                    try:
                        from avr.test_media_utils import run_ffmpeg_with_fallback
                        res = run_ffmpeg_with_fallback(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(float(t)), "-i", video_path, "-frames:v", "1", "-q:v", "2", out_path], fallback_hwaccel=True, verbose=False)
                        if res.returncode != 0 or not os.path.exists(out_path):
                            continue
                    except Exception:
                        import subprocess
                        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(float(t)), "-i", video_path, "-frames:v", "1", "-q:v", "2", out_path], check=False)
                        if not os.path.exists(out_path):
                            continue
                pil_frames.append(Image.open(out_path).convert("RGB"))
            except Exception:
                continue
    return pil_frames


def _resolve_local_path(p: str) -> str:
    """Try to resolve a possibly-remote path to a local filesystem path.
    - If path exists, return as-is.
    - If it starts with a known remote prefix, try replacing with cwd.
    - Try searching for the basename under some likely directories.
    Returns the candidate path (original if no alternative found).
    """
    try:
        p_str = str(p)
    except Exception:
        return p
    # already exists
    try:
        if os.path.exists(p_str):
            return p_str
    except Exception:
        pass

    # common remote workspace prefix used in logs
    remote_prefix = "/work/Vimo/VideoRAG-algorithm"
    try:
        cwd = os.getcwd()
    except Exception:
        cwd = None
    if cwd and p_str.startswith(remote_prefix):
        # map remote prefix to current cwd
        candidate = p_str.replace(remote_prefix, cwd.replace('\\', '/'))
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            return candidate

    # try searching by basename in several likely folders to recover moved data
    try:
        import glob
        base = os.path.basename(p_str)
        search_roots = [cwd] if cwd else []
        search_roots += [os.path.join(cwd, 'group') if cwd else None, os.path.join(cwd, 'workdir') if cwd else None, os.path.join(cwd, 'videorag-workdir') if cwd else None]
        try:
            autodl_roots = []
            possible = [' ', ' ', os.path.join(cwd, ' ') if cwd else None]
            for p in possible:
                if p and os.path.exists(p):
                    # prefer the batch_run subfolder if present/used
                    br = os.path.join(p, 'batch_run')
                    if os.path.exists(br):
                        autodl_roots.append(br)
                    else:
                        autodl_roots.append(p)
            search_roots += autodl_roots
        except Exception:
            pass
        for root in [r for r in search_roots if r]:
            pattern = os.path.join(root, '**', base)
            for match in glob.glob(pattern, recursive=True):
                if os.path.exists(match):
                    return match
    except Exception:
        pass

    return p_str


def extract_ocr_text_for_segments(
    segments: List[str],
    video_path_db,
    video_segments,
    num_frames: int,
    languages: List[str] | None = None,
    pre_extracted_frames: Dict[str, List] | None = None,
) -> Dict[str, str]:
    """
    Extract OCR text for each segment by sampling frames and running EasyOCR.
    Returns a map: segment_id -> concatenated_text
    """
    if not _HAS_EASYOCR:
        print("[Refine-OCR] Skipped: easyocr library is not installed.")
        return {}
    
    try:
        if languages is None:
            # English + Simplified Chinese by default
            languages = ["en", "ch_sim"]
        # Prefer GPU if available, but gracefully fall back to CPU on any CUDA/cuDNN/runtime errors
        use_gpu = torch.cuda.is_available() and os.environ.get("EASYOCR_FORCE_CPU", "0").strip() not in {"1", "true", "True"}
        try:
            reader = easyocr.Reader(languages, gpu=use_gpu)
        except Exception as gpu_err:
            # Common causes: cuDNN version mismatch, driver/runtime conflicts, missing CUDA libraries, etc.
            print(f"[Refine-OCR] GPU init failed ({gpu_err}). Falling back to CPU...")
            # Explicitly retry with CPU to avoid CUDA dependencies
            reader = easyocr.Reader(languages, gpu=False)
    except Exception as e:
        print(f"[Refine-OCR] Failed to initialize EasyOCR reader. Error: {e}")
        # This can happen due to model download issues or CUDA/driver incompatibilities.
        return {}

    seg2text: Dict[str, str] = {}
    total_frames = 0
    total_text_tokens = 0
    try:
        for s_id in tqdm(segments, desc="Refine Pass: OCR"):
            # Allow one retry for transient failures when processing a segment.
            max_attempts = 2
            attempt = 0
            # --- Lightweight preflight check: skip segments that are unlikely to yield OCR ---
            try:
                # If pre_extracted_frames provided and contains at least one existing frame file, keep it.
                has_pre_frames = False
                if pre_extracted_frames is not None and s_id in pre_extracted_frames:
                    raw_list = pre_extracted_frames.get(s_id) or []
                    for item in raw_list:
                        path = item[0] if isinstance(item, (list, tuple)) else item
                        try:
                            resolved = _resolve_local_path(str(path))
                            if os.path.exists(resolved):
                                has_pre_frames = True
                                break
                        except Exception:
                            continue

                if not has_pre_frames:
                    # quick video path existence check: try a few common candidate keys
                    try:
                        video_name, _st, _et = _get_segment_time(video_segments, s_id)
                        vpdata = getattr(video_path_db, '_data', None)
                        if not isinstance(vpdata, dict):
                            try:
                                vpdata = dict(vpdata)
                            except Exception:
                                vpdata = {}
                        found_local = False
                        if isinstance(vpdata, dict) and video_name:
                            candidates = [video_name, video_name + '.mp4', video_name + 'FULL', video_name + 'FULL.mp4']
                            # also try stripping trailing underscores parts progressively
                            if '_' in video_name:
                                parts = video_name.split('_')
                                for i in range(1, len(parts)):
                                    prefix = '_'.join(parts[:-i])
                                    if prefix:
                                        candidates.append(prefix)
                                        candidates.append(prefix + '.mp4')
                            seen = set()
                            for c in candidates:
                                if c in seen:
                                    continue
                                seen.add(c)
                                try:
                                    val = vpdata.get(c)
                                except Exception:
                                    val = None
                                if not val:
                                    continue
                                if isinstance(val, str) and os.path.exists(val):
                                    found_local = True
                                    break
                                # if it's a dict mapping to inner clips, check inner values for local paths
                                if isinstance(val, dict):
                                    for kk, vv in val.items():
                                        try:
                                            if isinstance(vv, str) and os.path.exists(vv):
                                                found_local = True
                                                break
                                        except Exception:
                                            continue
                                    if found_local:
                                        break
                        else:
                            found_local = False
                    except Exception:
                        found_local = False

                # If neither pre-extracted frames nor a local video path was found, skip OCR for this segment
                if (not has_pre_frames) and (not found_local):
                    try:
                        if os.environ.get('REFINE_DEBUG', '0').lower() in {'1', 'true', 'yes'}:
                            print(f"[Refine-OCR] Skipping segment (no frames or local video): {s_id}")
                    except Exception:
                        pass
                    # record as skipped (so upper-level can mark failure later if needed)
                    # don't attempt OCR on this segment at all
                    continue
            except Exception:
                # if preflight check fails unexpectedly, fall through to normal processing
                pass
            while attempt < max_attempts:
                try:
                    # Prefer pre-extracted frames if provided
                    frames: List[Image.Image] = []
                    existing_frames: List[Image.Image] = []
                    existing_timestamps: List[float] = []
                    
                    if pre_extracted_frames is not None and s_id in pre_extracted_frames:
                        raw_list = pre_extracted_frames.get(s_id) or []
                        # raw_list may be [(path, ts), ...] or [path, ...]
                        for item in raw_list:
                            path = None
                            try:
                                path, ts = (item[0], item[1]) if isinstance(item, (list, tuple)) and len(item) > 1 else (item, -1.0)
                                path = str(path)
                                # Skip missing files early and optionally emit debug logs
                                resolved_path = _resolve_local_path(path)
                                if not os.path.exists(resolved_path):
                                    if os.environ.get("REFINE_DEBUG", "0").lower() in {"1", "true", "yes"}:
                                        try:
                                            print(f"[Refine-OCR] Warning: pre-extracted frame not found: {path} -> tried: {resolved_path}")
                                        except Exception:
                                            pass
                                    continue
                                existing_frames.append(Image.open(resolved_path).convert("RGB"))
                                if ts >= 0:
                                    existing_timestamps.append(float(ts))
                            except Exception as e:
                                if os.environ.get("REFINE_DEBUG", "0").lower() in {"1", "true", "yes"}:
                                    try:
                                        print(f"[Refine-OCR] Warning: failed to open pre-extracted frame {path}: {e}")
                                    except Exception:
                                        pass
                                continue
                    
                    # Sample additional frames, excluding existing ones
                    video_name, start, end = _get_segment_time(video_segments, s_id)
                    # Safely resolve video_path from video_path_db._data (handle missing entries)
                    vpdata = getattr(video_path_db, '_data', None)
                    if not isinstance(vpdata, dict):
                        try:
                            vpdata = dict(vpdata)
                        except Exception:
                            vpdata = {}

                    # Resolve the most likely video_path with conservative fallbacks.
                    def _resolve_video_path(vp: dict, vname: str):
                        """Try variants of vname to find a matching key in vp dict.

                        Strategy (conservative):
                        - exact match
                        - try with/without .mp4
                        - if vname contains underscores, progressively strip trailing parts
                          (useful for ids like base_3_1 or base_0_1 -> try base_3, base)
                        - finally fall back to the first available entry in vp (best-effort)
                        Returns (path, resolved_key) or (None, None).
                        """
                        if not isinstance(vp, dict) or not vname:
                            return None, None
                        tried = []
                        candidates = [vname]
                        if not vname.endswith('.mp4'):
                            candidates.append(vname + '.mp4')
                        else:
                            candidates.append(vname[:-4])

                        parts = vname.split('_') if '_' in vname else [vname]
                        # progressively strip trailing segments
                        for i in range(1, len(parts)):
                            prefix = '_'.join(parts[:-i])
                            if prefix:
                                candidates.append(prefix)
                                candidates.append(prefix + '.mp4')
                                # also try prefix with FULL marker which is used elsewhere
                                candidates.append(prefix + 'FULL')
                                candidates.append(prefix + 'FULL.mp4')

                        # dedupe while preserving order
                        seen = set(); uniq = []
                        for c in candidates:
                            if c not in seen:
                                seen.add(c); uniq.append(c)

                        for c in uniq:
                            try:
                                if c in vp and vp.get(c):
                                    return vp.get(c), c
                            except Exception:
                                continue

                        # If a top-level entry maps to an inner dict (per-video map),
                        # try to resolve a clip key inside it (e.g., base_0.mp4)
                        try:
                            parts = vname.split('_') if '_' in vname else [vname]
                            inner_candidates = []
                            if len(parts) >= 2 and parts[-1].isdigit():
                                base = '_'.join(parts[:-1])
                                try:
                                    three_idx = int(parts[-1])
                                except Exception:
                                    three_idx = None
                                if base and three_idx is not None:
                                    inner_candidates.extend([
                                        f"{base}_{three_idx}.mp4",
                                        f"{base}_{three_idx}",
                                    ])
                                    inner_candidates.append(base + '.mp4')
                                    inner_candidates.append(base)
                            else:
                                base = vname
                                inner_candidates.extend([base + '.mp4', base])

                            # check if vname or its top_base is a mapping
                            top_base = base.split('_')[0] if base else None
                            inner_map = None
                            if vname in vp and isinstance(vp[vname], dict):
                                inner_map = vp[vname]
                            elif top_base and top_base in vp and isinstance(vp[top_base], dict):
                                inner_map = vp[top_base]

                            if isinstance(inner_map, dict):
                                for ic in inner_candidates:
                                    if ic in inner_map and inner_map.get(ic):
                                        return inner_map.get(ic), ic
                        except Exception:
                            pass

                        # best-effort fallback to any available value
                        try:
                            for k, v in vp.items():
                                if v:
                                    return v, k
                        except Exception:
                            pass
                        return None, None

                    video_path, resolved_key = _resolve_video_path(vpdata, video_name)
                    # If the resolved value is an inner-map (clip_key -> url), try to pick the correct clip
                    # and download it to a local file (reuse test_media_utils.download_file/ensure_valid_video_or_skip).
                    try:
                        # If video_path is a dict, it's likely a mapping of clip_key -> url
                        if isinstance(video_path, dict):
                            inner_map = video_path
                            outer_key = resolved_key
                            # attempt to pick candidate inner keys similar to resolver logic
                            parts = video_name.split('_') if '_' in video_name else [video_name]
                            inner_candidates = []
                            if len(parts) >= 2 and parts[-1].isdigit():
                                base = '_'.join(parts[:-1])
                                try:
                                    three_idx = int(parts[-1])
                                except Exception:
                                    three_idx = None
                                if base and three_idx is not None:
                                    inner_candidates.extend([
                                        f"{base}_{three_idx}.mp4",
                                        f"{base}_{three_idx}",
                                    ])
                                    inner_candidates.append(base + '.mp4')
                                    inner_candidates.append(base)
                            else:
                                base = video_name
                                inner_candidates.extend([base + '.mp4', base])

                            # try to find an available inner clip and download if it's a remote URL
                            dl_path = None
                            dl_key = None
                            for ic in inner_candidates:
                                if ic in inner_map and inner_map.get(ic):
                                    val = inner_map.get(ic)
                                    # if it's already a local path, use it
                                    if isinstance(val, str) and os.path.exists(val):
                                        dl_path = val; dl_key = ic; break
                                    # if it's a remote url, try to download/validate
                                    if isinstance(val, str) and val.startswith(('http://', 'https://')):
                                        try:
                                            # lazy import to avoid hard dependency at module import time
                                            from avr.test_media_utils import download_file, ensure_valid_video_or_skip
                                            downloads_dir = os.path.join(os.getcwd(), 'downloads')
                                            os.makedirs(downloads_dir, exist_ok=True)
                                            cand = download_file(val, downloads_dir)
                                            if cand:
                                                cand2 = ensure_valid_video_or_skip(val, downloads_dir, cand)
                                            else:
                                                cand2 = None
                                            if cand2:
                                                dl_path = cand2; dl_key = ic; break
                                        except Exception:
                                            # if download helpers unavailable or failed, skip
                                            pass
                            # If we obtained a local path, update mapping so future lookups don't re-download
                            if dl_path and dl_key:
                                try:
                                    # write back into vpdata if it's a backing store
                                    if isinstance(vpdata, dict):
                                        # if outer_key refers to a nested dict, replace inner value
                                        if outer_key in vpdata and isinstance(vpdata[outer_key], dict):
                                            vpdata[outer_key][dl_key] = dl_path
                                        else:
                                            vpdata[dl_key] = dl_path
                                    # also attempt to update wrapper video_path_db if present
                                    try:
                                        if hasattr(video_path_db, '_data') and isinstance(video_path_db._data, dict):
                                            if outer_key in video_path_db._data and isinstance(video_path_db._data[outer_key], dict):
                                                video_path_db._data[outer_key][dl_key] = dl_path
                                            else:
                                                video_path_db._data[dl_key] = dl_path
                                    except Exception:
                                        pass
                                    # use the downloaded local path for sampling
                                    video_path = dl_path
                                    resolved_key = dl_key
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    if not video_path:
                        print(f"[Refine-OCR] Warning: video '{video_name}' not found for segment {s_id}. Skipping.")
                        break
                    
                    # Calculate how many *new* frames to sample
                    num_new_frames = max(0, num_frames - len(existing_frames))
                    
                    new_frames = _sample_frames(
                        video_path, 
                        start, 
                        end, 
                        num_new_frames, 
                        exclude_timestamps=existing_timestamps
                    )
                    
                    # Combine existing and new frames
                    frames = existing_frames + new_frames

                    # Debug: print per-segment sampling info when requested
                    try:
                        if os.environ.get("REFINE_DEBUG", "0").lower() in {"1", "true", "yes"}:
                            try:
                                print(f"[Refine-OCR][DEBUG] segment={s_id} video_path={video_path} resolved_key={resolved_key} existing_frames={len(existing_frames)} new_frames={len(new_frames)} total_frames={len(frames)}")
                            except Exception:
                                print(f"[Refine-OCR][DEBUG] segment={s_id} debug info unavailable")
                    except Exception:
                        pass
                    
                    if not frames:
                        break

                    total_frames += len(frames)
                    ocr_texts: List[str] = []
                    for img in frames:
                        # easyocr accepts numpy arrays (RGB)
                        result = reader.readtext(np.array(img))
                        if not result:
                            continue
                        for _, text, conf in result:
                            conf_thr = EASYOCR_CONF_THR
                            if conf is None:
                                conf_val = 0.0
                            else:
                                try:
                                    conf_val = float(conf)
                                except Exception:
                                    conf_val = 0.0
                            if conf_val < conf_thr:
                                continue
                            if isinstance(text, str) and text.strip():
                                ocr_texts.append(text.strip())
                        if len(ocr_texts):
                            normalized = [normalize_text(t) for t in ocr_texts]
                            kept = dedupe_texts_preserve_order_lsh(normalized, thresh=OCR_JACCARD_THR)
                            seg2text[s_id] = " ".join(kept)
                        else:
                            # If no OCR texts were kept, optionally print debug info
                            try:
                                if os.environ.get("REFINE_DEBUG", "0").lower() in {"1", "true", "yes"}:
                                    print(f"[Refine-OCR][DEBUG] segment={s_id} -- no OCR text extracted (frames={len(frames)})")
                            except Exception:
                                pass
                        # count tokens only if seg2text has entry; use get to avoid KeyError
                        total_text_tokens += len(seg2text.get(s_id, "").split())
                    # success, break retry loop
                    break
                except Exception as seg_err:
                    attempt += 1
                    if attempt < max_attempts:
                        try:
                            print(f"[Refine-OCR] Error processing segment {s_id}: {seg_err} -- retrying (attempt {attempt+1}/{max_attempts})")
                        except Exception:
                            pass
                        # small backoff to reduce immediate re-failure
                        try:
                            time.sleep(0.1)
                        except Exception:
                            pass
                        continue
                    else:
                        print(f"[Refine-OCR] Error processing segment {s_id}: {seg_err}")
                        break # Give up after retry
        print(f"[Refine-OCR] segments={len(segments)} frames={total_frames} tokens={total_text_tokens}")
    except Exception as e:
        print(f"[Refine-OCR] A critical error occurred during the OCR process: {e}")
        # Return any data processed so far
        return seg2text
    return seg2text

# Helper: 获取片段时间窗口 (video_name, start, end)
def _get_segment_time(video_segments, segment_id: str):
    """Safely fetch (video_name, start, end) for a segment id.
    Expected structure: video_segments[segment_id] = { 'video': name, 'start': float, 'end': float }
    Fallbacks to (segment_id, 0, 0) if missing.
    """
    try:
        # Support multiple container shapes: plain dict or wrapper with _data (e.g., SimpleStore, SimpleNamespace)
        meta = None
        if isinstance(video_segments, dict):
            meta = video_segments.get(segment_id)
        else:
            # try common wrapper attribute
            try:
                inner = getattr(video_segments, '_data', None)
                if isinstance(inner, dict):
                    meta = inner.get(segment_id)
                else:
                    # last resort: if object exposes .get, use it
                    if hasattr(video_segments, 'get') and callable(getattr(video_segments, 'get')):
                        try:
                            meta = video_segments.get(segment_id)
                        except Exception:
                            meta = None
            except Exception:
                meta = None
        if not meta:
            return segment_id, 0.0, 0.0
        video_name = meta.get('video') or meta.get('video_name') or segment_id
        start = float(meta.get('start', 0.0))
        end = float(meta.get('end', 0.0))
        return video_name, start, end
    except Exception:
        return segment_id, 0.0, 0.0


def extract_keyword_queries_from_query(query: str) -> List[str]:
    # naive keywords extraction: keep words length>=3, deduplicate
    stop = {
        "the","and","with","then","that","have","this","from","into","what","which","when","were","will","like","left","right","top","bottom","front","rear","back","show","showing","shows","look","looks","looking","please","tell","explain","how","why","where","who","whom","whose","does","did","doing","done","make","made","press","pressed","pressing","step","first","second","third","final","begin","end","start","stop","turn","on","off"
    }
    tokens = [w.strip(" ,.;:!?()[]{}\"'\n\t").lower() for w in query.split()]
    candidates = []
    seen = set()
    for w in tokens:
        if len(w) >= 3 and w not in stop and w.isalpha() and w not in seen:
            candidates.append(w)
            seen.add(w)
    # fallback
    if not len(candidates):
        candidates = [query[:64]]
    return candidates[:8]


# ==================================================================================
# == YOLO-World Implementation
# ==================================================================================
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

def detect_objects_for_segments_yolo_world(
    segment_ids: List[str],
    video_path_db: Any,
    video_segments_db: Any,
    pre_extracted_frames_db: Optional[Dict[str, Any]],
    num_frames: int,
    keywords: List[str], # 修改：接收关键词
    yolo_model: Any,
    max_frames_per_segment: int = 8
) -> Dict[str, str]:
    if not _HAS_YOLO:
        print("[Refine-DET] Skipped: ultralytics library is not installed.")
        return {}
    
    if yolo_model is None:
        print("[Refine-DET] YOLO-World model not pre-loaded. Skipping.")
        return {}

    if not keywords:
        print("[Refine-DET] No text queries provided for YOLO-World. Skipping.")
        return {}

    if not segment_ids:
        return {}
        
    # Avoid calling `set_classes` on the model because some ultralytics
    # implementations may attempt tensor operations on different devices
    # (CPU vs CUDA) which raises device-mismatch errors. Instead, we
    # keep the model's classes as-is and perform a post-hoc,
    # case-insensitive label filter against the generated keywords.
    try:
        print(f"[YOLO-World] Using model keywords (post-filter): {keywords}")
    except Exception:
        pass
    keywords_lower = set([str(k).lower() for k in (keywords or [])])

    seg2desc: Dict[str, str] = {}
    total_frames = 0
    total_detections = 0

    try:
        for s_id in tqdm(segment_ids, desc="Refine Pass: DET (YOLO-World)"):
            try:
                frames: List[Image.Image] = []
                # Frame extraction logic remains the same as OWL-ViT version
                # ... (Assuming this part is correct and handles pre_extracted_frames)
                if pre_extracted_frames_db and s_id in pre_extracted_frames_db:
                    # 使用所有目标帧进行检测（不截断）
                    raw_list = pre_extracted_frames_db.get(s_id) or []
                    paths = [item[0] if isinstance(item, (list, tuple)) else item for item in raw_list]
                    frames = []
                    for p in paths:
                        try:
                            p_str = str(p)
                            resolved = _resolve_local_path(p_str)
                            if not os.path.exists(resolved):
                                if os.environ.get("REFINE_DEBUG", "0").lower() in {"1", "true", "yes"}:
                                    try:
                                        print(f"[Refine-DET] Warning: pre-extracted frame not found: {p_str} -> tried: {resolved}")
                                    except Exception:
                                        pass
                                continue
                            frames.append(Image.open(resolved).convert("RGB"))
                        except Exception as e:
                            if os.environ.get("REFINE_DEBUG", "0").lower() in {"1", "true", "yes"}:
                                try:
                                    print(f"[Refine-DET] Warning: failed to open pre-extracted frame {p}: {e}")
                                except Exception:
                                    pass
                            continue
                else:
                    video_name, start, end = _get_segment_time(video_segments_db, s_id)
                    # Safely resolve video_path from video_path_db._data (handle missing entries)
                    vpdata = getattr(video_path_db, '_data', None)
                    if not isinstance(vpdata, dict):
                        try:
                            vpdata = dict(vpdata)
                        except Exception:
                            vpdata = {}

                    # reuse same conservative resolution logic as OCR
                    def _resolve_video_path(vp: dict, vname: str):
                        if not isinstance(vp, dict) or not vname:
                            return None, None
                        tried = []
                        candidates = [vname]
                        if not vname.endswith('.mp4'):
                            candidates.append(vname + '.mp4')
                        else:
                            candidates.append(vname[:-4])
                        parts = vname.split('_') if '_' in vname else [vname]
                        for i in range(1, len(parts)):
                            prefix = '_'.join(parts[:-i])
                            if prefix:
                                candidates.append(prefix)
                                candidates.append(prefix + '.mp4')
                                candidates.append(prefix + 'FULL')
                                candidates.append(prefix + 'FULL.mp4')
                        seen = set(); uniq = []
                        for c in candidates:
                            if c not in seen:
                                seen.add(c); uniq.append(c)
                        for c in uniq:
                            try:
                                if c in vp and vp.get(c):
                                    return vp.get(c), c
                            except Exception:
                                continue
                        try:
                            for k, v in vp.items():
                                if v:
                                    return v, k
                        except Exception:
                            pass
                        return None, None

                    video_path, resolved_key = _resolve_video_path(vpdata, video_name)
                    # handle inner-map -> try to download clip similar to OCR path
                    try:
                        if isinstance(video_path, dict):
                            inner_map = video_path
                            outer_key = resolved_key
                            parts = video_name.split('_') if '_' in video_name else [video_name]
                            inner_candidates = []
                            if len(parts) >= 2 and parts[-1].isdigit():
                                base = '_'.join(parts[:-1])
                                try:
                                    three_idx = int(parts[-1])
                                except Exception:
                                    three_idx = None
                                if base and three_idx is not None:
                                    inner_candidates.extend([
                                        f"{base}_{three_idx}.mp4",
                                        f"{base}_{three_idx}",
                                    ])
                                    inner_candidates.append(base + '.mp4')
                                    inner_candidates.append(base)
                            else:
                                base = video_name
                                inner_candidates.extend([base + '.mp4', base])

                            dl_path = None; dl_key = None
                            for ic in inner_candidates:
                                if ic in inner_map and inner_map.get(ic):
                                    val = inner_map.get(ic)
                                    if isinstance(val, str) and os.path.exists(val):
                                        dl_path = val; dl_key = ic; break
                                    if isinstance(val, str) and val.startswith(('http://', 'https://')):
                                        try:
                                            from avr.test_media_utils import download_file, ensure_valid_video_or_skip
                                            downloads_dir = os.path.join(os.getcwd(), 'downloads')
                                            os.makedirs(downloads_dir, exist_ok=True)
                                            cand = download_file(val, downloads_dir)
                                            if cand:
                                                cand2 = ensure_valid_video_or_skip(val, downloads_dir, cand)
                                            else:
                                                cand2 = None
                                            if cand2:
                                                dl_path = cand2; dl_key = ic; break
                                        except Exception:
                                            pass
                            if dl_path and dl_key:
                                try:
                                    if isinstance(vpdata, dict):
                                        if outer_key in vpdata and isinstance(vpdata[outer_key], dict):
                                            vpdata[outer_key][dl_key] = dl_path
                                        else:
                                            vpdata[dl_key] = dl_path
                                    try:
                                        if hasattr(video_path_db, '_data') and isinstance(video_path_db._data, dict):
                                            if outer_key in video_path_db._data and isinstance(video_path_db._data[outer_key], dict):
                                                video_path_db._data[outer_key][dl_key] = dl_path
                                            else:
                                                video_path_db._data[dl_key] = dl_path
                                    except Exception:
                                        pass
                                    video_path = dl_path; resolved_key = dl_key
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    if not video_path:
                        print(f"[Refine-DET] Warning: video '{video_name}' not found for segment {s_id}. Skipping.")
                        continue
                    frames = _sample_frames(video_path, start, end, max(1, min(num_frames, 16)))
                
                if not frames:
                    continue

                total_frames += len(frames)
                
                # Batch inference with dynamic confidence backoff
                detected_lines: List[str] = []
                try:
                    conf0 = float(os.environ.get("YOLOWORLD_CONF_THR", "0.25") or 0.25)
                except Exception:
                    conf0 = 0.25
                try:
                    iou_thr = float(os.environ.get("YOLOWORLD_IOU_THR", "0.50") or 0.50)
                except Exception:
                    iou_thr = 0.50
                names_map = getattr(yolo_model, 'names', {}) or {}
                backoff = []
                for c in [conf0, 0.25, 0.20, 0.15, 0.10, 0.05]:
                    if c not in backoff and c >= 0.01:
                        backoff.append(c)
                def _predict_with_device_fallback(model, frames, conf_thr, iou_thr):
                    try:
                        return model.predict(source=frames, conf=conf_thr, iou=iou_thr, verbose=False)
                    except RuntimeError as re_err:
                        msg = str(re_err)
                        # Common ultralytics failure: internal `set_classes` or index_select
                        # operating across cuda/cpu tensors. If detected, try a CPU retry.
                        if ('Expected all tensors to be on the same device' in msg) or ('set_classes' in msg) or ('index_select' in msg):
                            try:
                                print('[YOLO-World][WARN] predict failed due to device mismatch; moving model to CPU and retrying...')
                                try:
                                    model.to('cpu')
                                except Exception:
                                    try:
                                        model.cpu()
                                    except Exception:
                                        pass
                                return model.predict(source=frames, conf=conf_thr, iou=iou_thr, verbose=False)
                            except Exception:
                                # fallback: re-raise the original runtime error for outer handler
                                raise
                        raise

                for conf_thr in backoff:
                    results = _predict_with_device_fallback(yolo_model, frames, conf_thr, iou_thr)
                    detected_lines = []
                    for res in results:
                        boxes = getattr(res, 'boxes', None)
                        if boxes is None:
                            continue
                        for box in boxes:
                            try:
                                # ultralytics tensors -> python scalars
                                label_idx = int(getattr(box, 'cls', [0])[0])
                            except Exception:
                                try:
                                    label_idx = int(box.cls.item())
                                except Exception:
                                    label_idx = 0
                            label = names_map.get(label_idx, str(label_idx))
                            # Post-filter by keyword names (case-insensitive).
                            # If keywords were not provided or the set is empty,
                            # keep all detections. This avoids calling
                            # `set_classes` which can trigger device mismatch.
                            if keywords_lower:
                                if str(label).lower() not in keywords_lower:
                                    continue
                            try:
                                score = float(getattr(box, 'conf', [0.0])[0])
                            except Exception:
                                try:
                                    score = float(box.conf.item())
                                except Exception:
                                    score = 0.0
                            if score < conf_thr:
                                continue
                            try:
                                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                            except Exception:
                                try:
                                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                                except Exception:
                                    x1 = y1 = x2 = y2 = 0
                            detected_lines.append(f"{label} ({score:.2f}) @ [{x1},{y1},{x2},{y2}]")
                    if detected_lines:
                        break
                
                if detected_lines:
                    uniq = sorted(list(set(detected_lines)))
                    seg2desc[s_id] = "; ".join(uniq[:20])
                    total_detections += len(uniq)
            
            except Exception as seg_err:
                print(f"[Refine-DET] Error processing segment {s_id} with YOLO-World: {seg_err}")
                continue

        print(f"[Refine-DET] segments={len(segment_ids)} frames={total_frames} detections={total_detections}")
    except Exception as e:
        print(f"[Refine-DET] A critical error occurred during the YOLO-World process: {e}")
        return seg2desc
        
    # If nothing detected at all, try a one-time fallback with generic, highly visible classes
    try:
        do_fallback = (total_detections == 0) and (os.environ.get("YOLOWORLD_FALLBACK_ON_EMPTY", "1").lower() in {"1","true","yes"})
        fallback_keywords = [
            "person", "car", "truck", "bus", "bicycle", "motorcycle",
            "flag", "banner", "microphone", "camera", "phone", "laptop",
            "screen", "table", "chair", "building", "gun", "police",
        ]
        if do_fallback and set([k.lower() for k in keywords]) != set([k.lower() for k in fallback_keywords]):
            print("[Refine-DET] No detections with smart keywords, retrying once with generic visible keywords...")
            return detect_objects_for_segments_yolo_world(
                segment_ids=segment_ids,
                video_path_db=video_path_db,
                video_segments_db=video_segments_db,
                pre_extracted_frames_db=pre_extracted_frames_db,
                num_frames=num_frames,
                keywords=fallback_keywords,
                yolo_model=yolo_model,
                max_frames_per_segment=max_frames_per_segment,
            )
    except Exception:
        pass

    return seg2desc


# ==================================================================================
# == (DEPRECATED) OWL-ViT Implementation
# ==================================================================================
# def detect_objects_for_segments_owlvit(...)
# ... (The entire old function is commented out or can be removed)
# ==================================================================================
