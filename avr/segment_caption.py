import base64
import asyncio
import os
import time
from io import BytesIO
from typing import List
from pathlib import Path

from PIL import Image

from avr.env_utils import image_to_base64
from videorag._config import (
    VISION_CAPTION_FRAME_TIMEOUT_SECONDS_DEFAULT,
    VISION_CAPTION_MAX_PARALLEL_DEFAULT,
    VISION_CAPTION_PARALLEL_EST_MEMORY_GB_DEFAULT,
    VISION_CAPTION_PARALLEL_RESERVE_GB_DEFAULT,
    VISION_CAPTION_RETRY_ATTEMPTS_DEFAULT,
    VISION_CAPTION_SYNTH_RETRY_ATTEMPTS_DEFAULT,
    VISION_CAPTION_SYNTH_TIMEOUT_SECONDS_DEFAULT,
    VISION_CAPTION_WARMUP_TIMEOUT_SECONDS_DEFAULT,
)

_WARMED_CAPTION_MODELS: set[str] = set()


class CaptionOOMError(RuntimeError):
    """Raised when a caption model call fails specifically due to GPU/VRAM exhaustion."""


def _is_caption_oom_error(exc: Exception | str | None) -> bool:
    text = str(exc or "").strip().lower()
    if not text:
        return False
    markers = [
        "cuda out of memory",
        "out of memory",
        "cuda oom",
        "oom",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
        "not enough memory",
        "insufficient memory",
        "failed to allocate memory",
        "allocation on device",
        "hip out of memory",
        "vram",
    ]
    return any(marker in text for marker in markers)


def release_caption_gpu_memory():
    """Best-effort cleanup after a caption OOM to improve recovery odds."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _load_prompt_template(prompt_filename: str, required_placeholders: list[str]) -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / prompt_filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    missing = [placeholder for placeholder in required_placeholders if placeholder not in template]
    if missing:
        raise ValueError(f"Prompt template {prompt_path.name} is missing placeholders: {', '.join(missing)}")
    return template


def _is_likely_vision_capable_model(model_name: str) -> bool:
    name = (model_name or "").strip().lower()
    if not name:
        return False

    try:
        from videorag._config import OLLAMA_VISION_MODEL_HINTS

        hints = [str(item).strip().lower() for item in (OLLAMA_VISION_MODEL_HINTS or []) if str(item).strip()]
        if any(hint and hint in name for hint in hints):
            return True
    except Exception:
        pass

    positive_markers = [
        "llava",
        "minicpm",
        "internvl",
        "gpt-4o",
        "gemini",
        "claude",
        "vision",
        "qwen-vl",
        "qwen2.5-vl",
        "qwen2_5-vl",
        "qwen2vl",
        "qwen2.5vl",
        "vl:",
        "vl-",
    ]
    return any(marker in name for marker in positive_markers)


def is_ollama_caption_backend(llm_cfg) -> bool:
    try:
        raw_func = getattr(llm_cfg, "cheap_model_func_raw", None)
        func_name = getattr(raw_func, "__name__", "") or ""
        return "ollama" in func_name.lower()
    except Exception:
        return False


def _vision_safe_image_base64(image_path: str, min_edge: int = 64, factor: int = 32) -> str:
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size

            scale = max(
                1.0,
                float(min_edge) / float(max(1, width)),
                float(min_edge) / float(max(1, height)),
            )
            new_w = max(min_edge, int(round(width * scale)))
            new_h = max(min_edge, int(round(height * scale)))

            def _align_up(value: int) -> int:
                return max(factor, ((int(value) + factor - 1) // factor) * factor)

            aligned_w = _align_up(new_w)
            aligned_h = _align_up(new_h)

            if (aligned_w, aligned_h) != (width, height):
                img = img.resize((aligned_w, aligned_h))

            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return image_to_base64(image_path)


def _caption_progress_label(frames: List[str]) -> str:
    try:
        if not frames:
            return "segment"
        parent = os.path.basename(os.path.dirname(frames[0])) or "segment"
        return parent[:-7] if parent.endswith("_frames") else parent
    except Exception:
        return "segment"


def _render_caption_progress(label: str, current: int, total: int, last_elapsed: float | None = None):
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    width = 20
    filled = int(round((current / total) * width))
    bar = "#" * filled + "-" * max(0, width - filled)
    suffix = f"{current}/{total}"
    if last_elapsed is not None:
        suffix += f" {last_elapsed:.1f}s"
    end = "\n" if current >= total else "\r"
    print(f"[Caption] {label} [{bar}] {suffix}", end=end, flush=True)


async def _call_caption_model_with_timeout(factory, stage: str, timeout_seconds: float, retry_attempts: int, *, log_timing: bool = True):
    attempts = max(1, int(retry_attempts))
    last_error = ""
    for attempt in range(1, attempts + 1):
        started = time.time()
        try:
            result = await asyncio.wait_for(factory(), timeout=timeout_seconds)
            elapsed = time.time() - started
            if log_timing:
                print(f"[Caption][Timing] {stage} attempt {attempt}/{attempts} done in {elapsed:.2f}s")
            return result
        except asyncio.TimeoutError:
            elapsed = time.time() - started
            last_error = f"timeout after {elapsed:.2f}s"
            if not log_timing:
                print("", flush=True)
            print(f"[Caption][Timeout] {stage} attempt {attempt}/{attempts} exceeded {timeout_seconds}s")
        except Exception as exc:
            if _is_caption_oom_error(exc):
                release_caption_gpu_memory()
                if not log_timing:
                    print("", flush=True)
                print(f"[Caption][OOM] {stage} attempt {attempt}/{attempts} paused due to GPU memory pressure: {exc}")
                raise CaptionOOMError(str(exc)) from exc
            elapsed = time.time() - started
            last_error = str(exc)
            if not log_timing:
                print("", flush=True)
            print(f"[Caption][Error] {stage} attempt {attempt}/{attempts} failed after {elapsed:.2f}s: {exc}")
    print(f"[Caption][GiveUp] {stage} exhausted {attempts} attempt(s). Last error: {last_error}")
    return ""


def estimate_caption_parallel_capacity() -> int:
    explicit_parallel = int(os.environ.get("VISION_CAPTION_MAX_PARALLEL", str(VISION_CAPTION_MAX_PARALLEL_DEFAULT)) or 0)
    fallback_parallelism = explicit_parallel if explicit_parallel > 0 else 1

    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
            return fallback_parallelism

        est_mem_gb = float(os.environ.get("VISION_CAPTION_PARALLEL_EST_MEMORY_GB", str(VISION_CAPTION_PARALLEL_EST_MEMORY_GB_DEFAULT)))
        reserve_gb = float(os.environ.get("VISION_CAPTION_PARALLEL_RESERVE_GB", str(VISION_CAPTION_PARALLEL_RESERVE_GB_DEFAULT)))
        slots = 0
        inspected = 0
        for device_idx in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(device_idx):
                    free_bytes, _ = torch.cuda.mem_get_info()
                free_gb = free_bytes / float(1024 ** 3)
                device_slots = max(0, int((free_gb - reserve_gb) // max(0.5, est_mem_gb)))
                slots += device_slots
                inspected += 1
            except Exception:
                continue

        if inspected <= 0:
            return fallback_parallelism
        if explicit_parallel > 0:
            return max(0, min(explicit_parallel, slots))
        return max(0, slots)
    except Exception:
        return fallback_parallelism


async def warmup_caption_model(llm_cfg, sample_image_path: str | None = None):
    model_name = getattr(llm_cfg, "cheap_model_name", "") or getattr(llm_cfg, "best_model_name", "")
    model_key = (model_name or "").strip()
    if not model_key or model_key in _WARMED_CAPTION_MODELS:
        return
    if not _is_likely_vision_capable_model(model_key):
        _WARMED_CAPTION_MODELS.add(model_key)
        return

    timeout_seconds = float(
        os.environ.get(
            "VISION_CAPTION_WARMUP_TIMEOUT_SECONDS",
            str(VISION_CAPTION_WARMUP_TIMEOUT_SECONDS_DEFAULT),
        )
    )
    images_base64 = None
    if sample_image_path:
        try:
            b64 = _vision_safe_image_base64(sample_image_path)
            if b64:
                images_base64 = [b64]
        except Exception:
            images_base64 = None

    try:
        print(f"[Caption][Warmup] model={model_key} start")
        await asyncio.wait_for(
            llm_cfg.cheap_model_func(
                prompt="Warm up the vision model and reply with OK.",
                system_prompt="Reply with OK only.",
                images_base64=images_base64,
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
            ),
            timeout=timeout_seconds,
        )
        _WARMED_CAPTION_MODELS.add(model_key)
        print(f"[Caption][Warmup] model={model_key} succeeded")
    except Exception as exc:
        print(f"[Caption][Warmup][WARN] model={model_key} failed: {exc}")

# 提供原 question_processing.py 中的段级多帧字幕生成逻辑（逐字搬迁，不改业务逻辑）
async def generate_segment_caption(frames: List[str], llm_cfg) -> str:
    """Generate an English multi-frame grounded caption.

    Pipeline:
      1) Per-frame objective sentence (no speculation) so model attends to every frame.
      2) Synthesize all unique sentences into ONE paragraph that integrates ALL visual evidence.
    """
    if not frames:
        return ""

    model_name = getattr(llm_cfg, "cheap_model_name", "") or getattr(llm_cfg, "best_model_name", "")
    frame_timeout = float(os.environ.get("VISION_CAPTION_FRAME_TIMEOUT_SECONDS", str(VISION_CAPTION_FRAME_TIMEOUT_SECONDS_DEFAULT)))
    synth_timeout = float(os.environ.get("VISION_CAPTION_SYNTH_TIMEOUT_SECONDS", str(VISION_CAPTION_SYNTH_TIMEOUT_SECONDS_DEFAULT)))
    retry_attempts = int(os.environ.get("VISION_CAPTION_RETRY_ATTEMPTS", str(VISION_CAPTION_RETRY_ATTEMPTS_DEFAULT)))
    synth_retry_attempts = int(os.environ.get("VISION_CAPTION_SYNTH_RETRY_ATTEMPTS", str(VISION_CAPTION_SYNTH_RETRY_ATTEMPTS_DEFAULT)))
    progress_label = _caption_progress_label(frames)
    if not _is_likely_vision_capable_model(model_name):
        print(
            f"[Caption][Skip] Model '{model_name or '<unknown>'}' does not look vision-capable; "
            "skipping image caption generation and relying on transcript/OCR text."
        )
        return ""

    # --- Stage 1: per-frame objective description (English) ---
    async def _describe_single_frame(img_path: str, idx: int, total: int) -> str:
        b64 = _vision_safe_image_base64(img_path)
        if not b64:
            return ""
        prompt = _load_prompt_template(
            "FrameDescription.md",
            ["{frame_index}", "{total_frames}"],
        ).format(
            frame_index=idx + 1,
            total_frames=total,
        )
        async def _factory():
            return await llm_cfg.cheap_model_func(
                prompt=prompt,
                system_prompt=(
                    "Return ONLY one concise objective English sentence. Do NOT speculate."
                ),
                images_base64=[b64],
                max_new_tokens=80,
                temperature=0.10,
                top_p=0.9
            )

        resp = await _call_caption_model_with_timeout(
            _factory,
            stage=f"frame {idx + 1}/{total} describe",
            timeout_seconds=frame_timeout,
            retry_attempts=retry_attempts,
            log_timing=False,
        )

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
        started = time.time()
        try:
            d = await _describe_single_frame(fp, i, len(frames))
        except Exception as e:
            print(f"[Caption][ERR] single frame caption error: {e}")
            d = ""
        if d:
            frame_descs.append(d)
        _render_caption_progress(progress_label, i + 1, len(frames), time.time() - started)
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
    synth_prompt = _load_prompt_template("CaptionGeneration.md", ["{joined}"]).format(joined=joined)
    print(f"[Caption] Synthesizing {len(uniq_descs)} unique frame descriptions")
    async def _synth_factory():
        return await llm_cfg.cheap_model_func(
            prompt=synth_prompt,
            system_prompt="Produce ONE objective English paragraph. Do not speculate.",
            images_base64=None,
            max_new_tokens=400,
            temperature=0.2,
            top_p=0.9
        )

    resp2 = await _call_caption_model_with_timeout(
        _synth_factory,
        stage="caption synthesis",
        timeout_seconds=synth_timeout,
        retry_attempts=synth_retry_attempts,
    )
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
