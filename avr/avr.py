import os
import sys
import json
import logging
import warnings
import multiprocessing
from datetime import datetime
import urllib.request
import subprocess
import glob
import argparse
from PIL import Image
import numpy as np
import base64
import traceback
import shutil
import re
import ast
import argparse
import pathlib
import random

# --- Ensure repo root (containing 'videorag/') is on sys.path ---
def _ensure_repo_root_on_syspath():
    try:
        cur = os.path.abspath(os.path.dirname(__file__))
        # Walk up a few levels to find a folder that directly contains 'videorag'
        for _ in range(6):
            if os.path.isdir(os.path.join(cur, "videorag")):
                if cur not in sys.path:
                    sys.path.insert(0, cur)
                break
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    except Exception:
        # Best-effort; don't block execution
        pass

_ensure_repo_root_on_syspath()

try:
    from videorag._config import apply_configured_gpu_environment

    configured_visible = apply_configured_gpu_environment()
    if configured_visible:
        print(f"[GPU] CUDA_VISIBLE_DEVICES={configured_visible}")
except Exception:
    pass

# Early normalization of legacy topk flags so imports that call argparse won't fail
try:
    argv = sys.argv
    if argv:
        normalized = []
        for a in argv:
            if a == '--top1':
                normalized.extend(['--topk', '1'])
            elif a == '--top3':
                normalized.extend(['--topk', '3'])
            else:
                normalized.append(a)
        sys.argv[:] = normalized
except Exception:
    pass

# Early (lightweight) --api detection: if the script was invoked with --api, set SKIP_OLLAMA
# before importing modules that may do model selection/warmup.
try:
    if '--api' in sys.argv:
        os.environ['SKIP_OLLAMA'] = '1'
        try:
            import api_config as _api_cfg
            prov = (_api_cfg.DEFAULT_PROVIDER or '').lower()
            if prov in ('anthropic', 'claude'):
                os.environ['OLLAMA_CHAT_MODEL'] = 'claude'
            elif prov == 'gemini':
                os.environ['OLLAMA_CHAT_MODEL'] = 'gemini'
            else:
                os.environ['OLLAMA_CHAT_MODEL'] = 'chatgpt'
        except Exception:
            pass
except Exception:
    pass

# Defer importing iterative_refinement until after args are parsed and --api handling
refine_context = None
IterativeRefiner = None
get_default_ollama_chat_model = None

from video_urls import video_urls_multi_segment
from avr.media_utils import (
    download_file,
    get_video_resolution,
    ensure_valid_video_or_skip,
    repair_mp4_faststart,
    is_valid_video,
    get_video_duration,
)
from avr.env_utils import (
    sanitize_cuda_libs,
    can_use_faster_whisper_cuda,
    check_dependencies,
    SimpleStore,
    check_models,
    normalize_question_text,
    extract_final_answer,
    choose_llm_config,
    image_to_base64,
    maybe_offline_fallback_answer,
)

# ---------------- Helper utilities moved to test_env_utils.py ----------------
# --- CRITICAL: Sanitize environment BEFORE importing torch-dependent libraries ---
sanitize_cuda_libs()


try:
    import nest_asyncio
    _HAS_NEST_ASYNCIO = True
except Exception:
    _HAS_NEST_ASYNCIO = False

# ---- Notebook parity: keep the exact notebook lines ----
if _HAS_NEST_ASYNCIO:
    nest_asyncio.apply()

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
# utility functions now imported from test_env_utils and test_media_utils
from processing_utils import _frame_cache_dir_for_segment, extract_frames_and_compress, transcribe_segment_audio
from question_processing import process_question

## duplicate choose_llm_config & helpers removed (now sourced from test_env_utils)




async def batch_main():
    # ---- Setup ----
    def _find_repo_root_from_here() -> str:
        cur = os.path.abspath(os.path.dirname(__file__))
        candidates = ("videorag", "faster-distil-whisper-large-v3")
        for _ in range(8):
            has_any = any(os.path.exists(os.path.join(cur, c)) for c in candidates)
            if has_any:
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return os.path.abspath(os.path.dirname(__file__))

    # Normalize legacy topk flags early so any argparse elsewhere won't see unknown args
    try:
        # Map '--top1' -> ['--topk', '1'], '--top3' -> ['--topk', '3'] for robustness
        argv = sys.argv
        normalized = []
        i = 0
        while i < len(argv):
            a = argv[i]
            if a == '--top1':
                normalized.append('--topk')
                normalized.append('1')
            elif a == '--top3':
                normalized.append('--topk')
                normalized.append('3')
            else:
                normalized.append(a)
            i += 1
        # Replace sys.argv in place
        sys.argv[:] = normalized
    except Exception:
        pass

    repo_root = _find_repo_root_from_here()
    work_dir = os.path.join(repo_root, "videorag-workdir")
    os.makedirs(work_dir, exist_ok=True)
    parser = argparse.ArgumentParser(description="Run VideoRAG batch processing.")
    parser.add_argument("file", nargs='?', default=None, help="Path to a specific JSON file with video segments to process.")
    parser.add_argument("--video_path", help="Path to a local benchmark video under Bench/ or a custom local MP4.")
    parser.add_argument("--question", help="Run a single custom question against --video_path.")
    parser.add_argument("--bench_dir", help="Override the benchmark directory. Defaults to config Bench/.")
    parser.add_argument("--workdir", help="Override the runtime work directory. Defaults to config workdir.")
    parser.add_argument("--force", action="store_true", help="Force re-processing of all steps, ignoring caches.")
    parser.add_argument("--base-mode", action="store_true", help="Run in base mode, skipping iterative refinement.")
    parser.add_argument("--ablation", action="store_true", help="Run ablation experiment: disable both OCR and DET during refinement.")
    parser.add_argument("--ablationocr", action="store_true", help="Run ablation experiment: disable only OCR during refinement.")
    parser.add_argument("--ablationdet", action="store_true", help="Run ablation experiment: disable only DET during refinement.")
    parser.add_argument("--api", action="store_true", help="Use API-backed closed-source model as configured in test/api_config.py or environment variables.")
    parser.add_argument("--rerank", action="store_true", help="Save outputs under a rerank folder and randomize segment input order for experimentation.")
    # topk experiment flags: support --top1, --top3 or generic --topk N (default 5)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--top1", action="store_true", help="Use top1 segments (equivalent to --topk 1)")
    group.add_argument("--top3", action="store_true", help="Use top3 segments (equivalent to --topk 3)")
    group.add_argument("--topk", type=int, help="Use top-K segments from the input top5_segments (default 5)")
    args = parser.parse_args()
    if args.question and not args.video_path:
        raise RuntimeError("--question requires --video_path.")
    # Determine topk value for experiments (default 5)
    topk = 5
    if getattr(args, 'topk', None):
        try:
            topk = int(args.topk)
        except Exception:
            topk = 5
    elif getattr(args, 'top1', False):
        topk = 1
    elif getattr(args, 'top3', False):
        topk = 3
    # repo_root and work_dir already set above

    # Helper: produce a derangement (no element remains in its original index) for a list of keys
    def _deranged_order(orig_keys: list[str]) -> list[str]:
        # For small lists, rejection sampling is fine; fallback to a cyclic rotation if unlucky
        if not orig_keys:
            return orig_keys
        n = len(orig_keys)
        if n <= 1:
            return orig_keys
        attempts = 0
        keys = list(orig_keys)
        while attempts < 1000:
            random.shuffle(keys)
            # ensure no key is at its original position
            ok = True
            for i in range(n):
                if keys[i] == orig_keys[i]:
                    ok = False
                    break
            if ok:
                return keys
            attempts += 1
        # fallback deterministic derangement: rotate by 1 (works when n>1)
        return orig_keys[1:] + orig_keys[:1]

    # ---- Preflight checks ----
    # sanitize_cuda_libs() is now called at the top of the script
    check_dependencies()
    check_models(repo_root)

    # ---- Initialize models ----
    print("[ASR] Loading faster-whisper model...")
    from faster_whisper import WhisperModel
    # Allow env override; else use central config default.
    try:
        from videorag._config import (
            WHISPER_MODEL_PATH as CONFIG_WHISPER_MODEL_PATH,
            WHISPER_FALLBACK_MODEL_ID as CONFIG_WHISPER_FALLBACK_MODEL_ID,
        )
    except Exception:
        CONFIG_WHISPER_MODEL_PATH = None
        CONFIG_WHISPER_FALLBACK_MODEL_ID = "distil-large-v3"
    asr_model_path = (
        os.environ.get("FASTER_WHISPER_DIR")
        or os.environ.get("ASR_MODEL_PATH")
        or CONFIG_WHISPER_MODEL_PATH
        or os.path.join(repo_root, "faster-distil-whisper-large-v3")
    )
    if not os.path.exists(asr_model_path):
        print(f"[ASR] Local model path not found: {asr_model_path}")
        print(f"[ASR] Falling back to configured whisper model id '{CONFIG_WHISPER_FALLBACK_MODEL_ID}' (will download if needed)")
        asr_model_path = CONFIG_WHISPER_FALLBACK_MODEL_ID
    
    # Smart device selection: prefer CPU if CUDA_VISIBLE_DEVICES is empty, otherwise try CUDA with fallback
    cuda_available = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
    if cuda_available:
        try:
            import torch
            cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
        except ImportError:
            cuda_available = False

    if cuda_available:
        cuda_available, cuda_reason = can_use_faster_whisper_cuda()
        if cuda_available:
            print(f"[ASR] CUDA runtime check passed: {cuda_reason}")
        else:
            print(f"[ASR] CUDA runtime check failed: {cuda_reason}. Falling back to CPU.")
    
    if cuda_available:
        # Try CUDA with int8 first (more memory efficient), fallback to CPU if OOM
        try:
            asr_model = WhisperModel(asr_model_path, device="cuda", compute_type="int8")
            print("[ASR] Using CUDA with int8 precision")
        except Exception as e:
            print(f"[ASR] CUDA failed ({e}), falling back to CPU...")
            asr_model = WhisperModel(asr_model_path, device="cpu", compute_type="int8")
    else:
        asr_model = WhisperModel(asr_model_path, device="cpu", compute_type="int8")
        print("[ASR] Using CPU with int8 precision")
    
    # Choose LLM config AFTER we process --api so SKIP_OLLAMA / env overrides are respected
    # If user requested --api, try to load thin API config and map to existing LLMConfig factories
    if args.api:
        try:
            # When using external API backends, skip local Ollama detection/usage
            os.environ["SKIP_OLLAMA"] = "1"
            import api_config
            get_api_llm_config = api_config.get_api_llm_config
            api_conf = get_api_llm_config()
            from videorag._llm import create_custom_openai_config, openai_4o_mini_config, azure_openai_config
            provider = api_conf.get("provider")
            # Map provider to friendly model tag for output folders
            try:
                if provider == 'anthropic' or provider == 'claude':
                    os.environ['OLLAMA_CHAT_MODEL'] = 'claude'
                elif provider == 'gemini':
                    os.environ['OLLAMA_CHAT_MODEL'] = 'gemini'
                elif provider == 'openai' or provider == 'custom':
                    os.environ['OLLAMA_CHAT_MODEL'] = 'chatgpt'
            except Exception:
                pass
            if provider == "custom" or provider == "gemini" or (provider == "openai" and api_conf.get("base_url")):
                base_url = api_conf.get("base_url") or api_conf.get("endpoint")
                api_key = api_conf.get("api_key")
                model_name = api_conf.get("model")
                if base_url and api_key:
                    print(f"[LLM] Using custom OpenAI-compatible API at {base_url} model={model_name}")
                    llm_cfg = create_custom_openai_config(base_url, api_key, model_name)
            elif provider == "openai":
                # rely on existing env var OPENAI_API_KEY or map passed key
                if api_conf.get("api_key"):
                    os.environ["OPENAI_API_KEY"] = api_conf.get("api_key")
                print(f"[LLM] Using OpenAI API model={api_conf.get('model')}")
                # fall back to default openai config (openai_4o_mini_config)
                llm_cfg = openai_4o_mini_config
            elif provider == "azure":
                # set Azure envs if provided
                if api_conf.get("api_key"):
                    os.environ["AZURE_OPENAI_API_KEY"] = api_conf.get("api_key")
                if api_conf.get("endpoint"):
                    os.environ["AZURE_OPENAI_ENDPOINT"] = api_conf.get("endpoint")
                print(f"[LLM] Using Azure OpenAI model={api_conf.get('model')}")
                llm_cfg = azure_openai_config
            elif provider == "anthropic":
                # map Anthropic/Claude to a custom OpenAI-compatible wrapper if available
                # For now set env var CUSTOM_OPENAI_BASE_URL to the endpoint and CUSTOM_OPENAI_API_KEY
                if api_conf.get("endpoint"):
                    os.environ["CUSTOM_OPENAI_BASE_URL"] = api_conf.get("endpoint")
                if api_conf.get("api_key"):
                    os.environ["CUSTOM_OPENAI_API_KEY"] = api_conf.get("api_key")
                if api_conf.get("model"):
                    os.environ["CUSTOM_OPENAI_MODEL"] = api_conf.get("model")
                from videorag._llm import create_custom_openai_config
                base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")
                api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
                model_name = os.environ.get("CUSTOM_OPENAI_MODEL")
                if base_url and api_key:
                    print(f"[LLM] Using Anthropic/Claude via custom endpoint {base_url} model={model_name}")
                    llm_cfg = create_custom_openai_config(base_url, api_key, model_name)
            else:
                print(f"[LLM] Unknown api provider from api_config: {provider}, falling back to choose_llm_config()")
        except Exception as e:
            print(f"[LLM] Failed to load API config: {e}. Falling back to default selection.")
    # If API branch didn't set llm_cfg, select the default llm config (honors SKIP_OLLAMA if set above)
    if 'llm_cfg' not in locals() or llm_cfg is None:
        llm_cfg = choose_llm_config()

    # Import iterative_refinement here to avoid module-level warmup before --api handling
    try:
        from videorag.iterative_refinement import refine_context, IterativeRefiner
        from videorag._llm import get_default_ollama_chat_model
    except Exception:
        # fallback: leave as None and allow errors to surface later
        pass
    # ---- Batch Processing ----
    from videorag._config import INPUT_DATA_PATH, WORKDIR_PATH as CONFIG_WORKDIR_PATH
    if not INPUT_DATA_PATH:
        raise RuntimeError("INPUT_DATA_PATH must be set in videorag._config and non-empty")
    input_base_dir = os.path.abspath(args.bench_dir or INPUT_DATA_PATH)
    work_dir = os.path.abspath(args.workdir or CONFIG_WORKDIR_PATH or work_dir)
    os.makedirs(work_dir, exist_ok=True)
    single_json_file: str | None = None
    if args.file:
        user_path = os.path.abspath(args.file)
        if os.path.isfile(user_path):
            single_json_file = user_path
            input_base_dir = os.path.dirname(user_path)
        elif os.path.isdir(user_path):
            input_base_dir = user_path
    force_rerun = args.force or (os.environ.get("FORCE_RERUN", "").strip().lower() in {"1", "true", "yes"})

    def _infer_model_tag(model_name: str) -> str:
        name = (model_name or "").lower()
        
        # Check for API models based on environment variables
        if '--api' in sys.argv:
            # Check for Claude model
            claude_model = os.environ.get("ANTHROPIC_MODEL", "").lower()
            if "claude" in claude_model:
                return "claude"
            
            # Check for Gemini model
            custom_model = os.environ.get("CUSTOM_OPENAI_MODEL", "").lower()
            # If custom model explicitly mentions Claude, prefer that tag
            if "claude" in custom_model:
                return "claude"
            if "gemini" in custom_model:
                return "gemini"
            if "gpt" in custom_model:
                return "gpt"
        
        # Existing logic for other models
        if "internvl" in name:
            return "InternVL"
        if "qwen" in name:
            return "qwen"
        if "llama" in name:
            return "llama"
        if "gemma" in name:
            return "gemma"
        if "llama" in name:
            return "llama"
        if "minicpm" in name or "openbmb/minicpm" in name:
            return "minicpm"
        # fallback: use prefix before ':' or whole name sanitized
        return (name.split(":", 1)[0] or "misc").replace("/", "_")

    # Determine active Ollama chat model and route outputs to a model-specific subfolder
    active_chat_model = os.environ.get("OLLAMA_CHAT_MODEL", "").strip() or get_default_ollama_chat_model()
    model_tag = _infer_model_tag(active_chat_model)
    
    # 根据 base-mode 调整输出目录
    from videorag._config import RESULT_PATH as CONFIG_RESULT_PATH, GT_DATA_PATH as CONFIG_GT_PATH

    if args.base_mode:
        model_tag = f"{model_tag}_base"
        print(f"[LLM] Base mode active. Output will be saved under directory tag: {model_tag}")

    # output root: strictly from config (no fallback)
    if not CONFIG_RESULT_PATH:
        raise RuntimeError("RESULT_PATH must be set in videorag._config and non-empty")
    # If running ablation experiment, write into RESULT_PATH/ablation/<model_tag> or ablationocr/ablationdet
    if getattr(args, 'ablation', False):
        # If topk experiment active, nest under top{k}
        if topk != 5:
            output_base_dir = os.path.join(CONFIG_RESULT_PATH, f"top{topk}", 'ablation', model_tag)
        else:
            output_base_dir = os.path.join(CONFIG_RESULT_PATH, 'ablation', model_tag)
        print(f"[Output] Ablation mode active (OCR+DET disabled). Results will be saved under: {output_base_dir}")
    elif getattr(args, 'ablationocr', False):
        # If topk experiment active, nest under top{k}
        if topk != 5:
            output_base_dir = os.path.join(CONFIG_RESULT_PATH, f"top{topk}", 'ablationocr', model_tag)
        else:
            output_base_dir = os.path.join(CONFIG_RESULT_PATH, 'ablationocr', model_tag)
        print(f"[Output] Ablation OCR mode active (only OCR disabled). Results will be saved under: {output_base_dir}")
    elif getattr(args, 'ablationdet', False):
        # If topk experiment active, nest under top{k}
        if topk != 5:
            output_base_dir = os.path.join(CONFIG_RESULT_PATH, f"top{topk}", 'ablationdet', model_tag)
        else:
            output_base_dir = os.path.join(CONFIG_RESULT_PATH, 'ablationdet', model_tag)
        print(f"[Output] Ablation DET mode active (only DET disabled). Results will be saved under: {output_base_dir}")
    else:
        # If topk experiment active, write under RESULT_PATH/top{k}/<model_tag>
        if topk != 5:
            output_base_dir = os.path.join(CONFIG_RESULT_PATH, f"top{topk}", model_tag)
        else:
            output_base_dir = os.path.join(CONFIG_RESULT_PATH, model_tag)
    # If user requested rerank-run, nest under a 'rerank' subfolder for easy discovery
    if getattr(args, 'rerank', False):
        output_base_dir = os.path.join(CONFIG_RESULT_PATH, 'rerank', os.path.relpath(output_base_dir, CONFIG_RESULT_PATH))
        print(f"[Output] Rerank mode active. Results will be saved under: {output_base_dir}")
    current_mode = "base" if args.base_mode else "refine"

    def _prepare_existing_results(existing_results, output_file_path: str):
        reusable_questions = set()
        cleaned_results = []
        dropped = 0

        for item in existing_results or []:
            if not isinstance(item, dict):
                cleaned_results.append(item)
                continue

            if "question" not in item or item.get("mode") != current_mode:
                cleaned_results.append(item)
                continue

            clean_answer = extract_final_answer(item.get("answer", ""))
            if clean_answer:
                reusable_questions.add(item["question"])
                cleaned_results.append(item)
            else:
                dropped += 1

        if dropped:
            print(f"[Resume] Ignoring {dropped} incomplete cached result(s) in {output_file_path}")

        return cleaned_results, reusable_questions

    def _build_local_segment_urls(video_id: str, local_video_path: str) -> dict[str, str]:
        segment_urls = {
            f"{video_id}_{i}": local_video_path
            for i in range(max(1, topk))
        }
        if getattr(args, 'rerank', False) and segment_urls:
            keys = _deranged_order(list(segment_urls.keys()))
            segment_urls = {k: segment_urls[k] for k in keys}
            print(f"[Rerank] video {video_id} order: {list(segment_urls.keys())}")
        return segment_urls

    def _iter_question_entries(qa_data, source_path: str) -> list[tuple[int, str]]:
        if isinstance(qa_data, list):
            items = qa_data
        elif isinstance(qa_data, dict):
            items = [qa_data]
        else:
            print(f"[Skip] Unsupported JSON shape in {source_path}: {type(qa_data).__name__}")
            return []

        entries: list[tuple[int, str]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                print(f"[Skip] Non-object entry in {source_path} at index {idx}")
                continue
            question = normalize_question_text(item.get("question", ""))
            if not question:
                print(f"[Skip] Missing question in {source_path} at index {idx}")
                continue
            entries.append((idx, question))
        return entries

    def _cleanup_generated_paths(video_id: str, touched_paths: set[str]):
        removed = 0
        for path in list(touched_paths):
            try:
                real_path = os.path.realpath(path)
                real_workdir = os.path.realpath(work_dir)
                if os.path.commonpath([real_path, real_workdir]) != real_workdir:
                    continue
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                    removed += 1
                elif os.path.isfile(path):
                    os.remove(path)
                    removed += 1
            except Exception:
                pass
        print(f"[Cache] Cleaned {video_id} ({removed} items)")

    def _single_video_cache_path(video_id: str) -> str:
        return os.path.join(work_dir, "_single_video_cache", f"{video_id}.json")

    def _load_single_video_cache(video_id: str) -> dict[str, dict]:
        cache_path = _single_video_cache_path(video_id)
        if force_rerun or not os.path.exists(cache_path):
            return {}
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            cache: dict[str, dict] = {}
            for key in ("segment_preprocess", "caption_results", "refined_frames"):
                value = data.get(key, {})
                cache[key] = value if isinstance(value, dict) else {}
            print(f"[Cache] Loaded persistent single-video cache -> {video_id}")
            return cache
        except Exception as err:
            print(f"[Cache][WARN] Failed to load persistent cache for {video_id}: {err}")
            return {}

    def _save_single_video_cache(video_id: str, runtime_cache: dict[str, dict] | None):
        if not runtime_cache:
            return
        cache_path = _single_video_cache_path(video_id)
        payload = {
            "segment_preprocess": runtime_cache.get("segment_preprocess", {}),
            "caption_results": runtime_cache.get("caption_results", {}),
            "refined_frames": runtime_cache.get("refined_frames", {}),
        }
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as err:
            print(f"[Cache][WARN] Failed to save persistent cache for {video_id}: {err}")

    async def _run_single_video_mode(local_video_path: str, custom_question: str | None = None):
        video_id = os.path.splitext(os.path.basename(local_video_path))[0]
        segment_urls = _build_local_segment_urls(video_id, local_video_path)
        if not segment_urls:
            print(f"SKIPPING: No segments found for video ID {video_id}")
            return

        touched_paths_for_video: set = set()
        runtime_cache_for_video: dict[str, dict] | None = {} if force_rerun else _load_single_video_cache(video_id)
        json_targets: list[tuple[str, list[tuple[int, str]]]] = []
        if custom_question is not None:
            question = normalize_question_text(custom_question)
            if not question:
                raise RuntimeError("Custom question is empty after normalization.")
            json_targets.append((os.path.join(video_id, "custom_question.json"), [(0, question)]))
        else:
            query_dir = os.path.join(input_base_dir, video_id)
            json_files = sorted(glob.glob(os.path.join(query_dir, "*.json")))
            if not json_files:
                print(f"[Skip] No JSON files found under: {query_dir}")
                return
            for json_file_path in json_files:
                norm_path = os.path.abspath(json_file_path)
                print(f"\nProcessing file: {norm_path}")
                try:
                    with open(norm_path, 'r', encoding='utf-8') as f:
                        qa_data = json.load(f)
                except Exception as e:
                    print(f"Error reading JSON {norm_path}: {e}")
                    continue
                question_entries = _iter_question_entries(qa_data, norm_path)
                if not question_entries:
                    continue
                json_targets.append((os.path.relpath(norm_path, input_base_dir), question_entries))

        all_done = True
        for relative_path, question_entries in json_targets:
            output_file_path = os.path.join(output_base_dir, relative_path)
            failure_log_path = output_file_path + ".failures.jsonl"
            existing_results = []
            existing_questions = set()
            if not force_rerun:
                try:
                    if os.path.exists(output_file_path):
                        with open(output_file_path, "r", encoding="utf-8") as f:
                            existing_results = json.load(f)
                            if isinstance(existing_results, dict):
                                existing_results = [existing_results]
                    existing_results, existing_questions = _prepare_existing_results(existing_results, output_file_path)
                except Exception:
                    try:
                        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        os.rename(output_file_path, output_file_path + ".bak")
                        print(f"[Resume] Corrupted result file backed up: {output_file_path}.bak")
                    except Exception:
                        pass
                    existing_results = []

            def _append_failure(index: int, question_text: str, err: Exception):
                record = {
                    "video_id": video_id,
                    "result_rel_path": relative_path,
                    "index": index,
                    "question": question_text,
                    "error": str(err),
                    "traceback": traceback.format_exc(),
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
                try:
                    os.makedirs(os.path.dirname(failure_log_path), exist_ok=True)
                    with open(failure_log_path, "a", encoding="utf-8") as flog:
                        flog.write(json.dumps(record, ensure_ascii=False) + "\n")
                    master_fail = os.path.join(output_base_dir, "_failures.jsonl")
                    with open(master_fail, "a", encoding="utf-8") as mlog:
                        mlog.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception:
                    pass

            output_data = [] if force_rerun else existing_results[:]
            expected_questions = {question for _, question in question_entries}

            for index, question in question_entries:
                print(f"\n--- Processing Q{index}: {question} for video {video_id} ---")
                if (not force_rerun) and (question in existing_questions):
                    print(f"[Skip] Q{index}: {question[:50]}... (cached non-empty answer)")
                    continue
                try:
                    answer_obj = await process_question(
                        question,
                        segment_urls,
                        work_dir,
                        asr_model,
                        llm_cfg,
                        touched_paths=touched_paths_for_video,
                        runtime_cache=runtime_cache_for_video,
                        base_mode=args.base_mode,
                        ablation=args.ablation,
                        ablation_ocr=getattr(args, 'ablationocr', False),
                        ablation_det=getattr(args, 'ablationdet', False)
                    )
                    output_data.append(
                        {
                            "question": question,
                            "answer": extract_final_answer(answer_obj.get("answer", "")),
                            "mode": current_mode,
                        }
                    )
                    existing_questions.add(question)
                    _save_single_video_cache(video_id, runtime_cache_for_video)
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                except Exception as err:
                    all_done = False
                    print(f"[Batch Error-Q{index}] {err}")
                    _append_failure(index, question, err)

            try:
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                print(f"[Saved] {output_file_path}")
            except Exception as err:
                all_done = False
                print(f"[Save Error] {err}")

            present_questions = {
                str(item.get("question", ""))
                for item in output_data
                if isinstance(item, dict) and item.get("question")
            }
            if not present_questions.issuperset(expected_questions):
                all_done = False

        _save_single_video_cache(video_id, runtime_cache_for_video)
        if touched_paths_for_video:
            print(f"[Cache] Preserving single-video intermediates for reuse -> {video_id}")

    if args.video_path:
        local_video_path = os.path.abspath(args.video_path)
        if not os.path.isfile(local_video_path):
            raise FileNotFoundError(f"Video not found: {local_video_path}")
        await _run_single_video_mode(local_video_path, custom_question=args.question)
        return
    
    # 如果目录不存在，不要在这里抛错；允许走顶层 JSON 模式/单文件模式
    video_id_dirs: list[str] = []
    if os.path.isdir(input_base_dir):
        # List only real video id directories. Exclude hidden folders (starting with '.')
        # such as '.ipynb_checkpoints' and Python cache directories like '__pycache__'.
        video_id_dirs = [
            d
            for d in os.listdir(input_base_dir)
            if os.path.isdir(os.path.join(input_base_dir, d))
            and not d.startswith('.')
            and d != '__pycache__'
        ]
    else:
        print(f"[Input] Base dir not found or not a directory: {input_base_dir}")

    # Fallback: 顶层JSON模式（当目录下没有子目录时，直接读取 *.json）
    if not video_id_dirs:
        from video_urls import video_urls_multi_segment as _video_map

        def _build_segment_urls_from_top5(top5_names: list[str]) -> dict[str, str]:
            seg_urls: dict[str, str] = {}
            for name in top5_names or []:
                try:
                    if not isinstance(name, str):
                        continue
                    parts = name.split('_')
                    # 形如 U2gvha4CipY_7 （全局第7个30秒片段）
                    if len(parts) == 2 and parts[1].isdigit():
                        base = parts[0]
                        seg_idx = int(parts[1])
                        three_min_idx = seg_idx // 6
                        thirty_idx = seg_idx % 6
                        video_info = _video_map.get(base)
                        if not video_info:
                            print(f"[Top5] Skip: video '{base}' not found in video_urls.py")
                            continue
                        clip_key = f"{base}_{three_min_idx}.mp4"
                        url = video_info.get(clip_key) or next(iter(video_info.values()), None)
                        if not url:
                            continue
                        # 若找不到三分钟分片键而回退到唯一文件（如 base.mp4），标记 FULL 以便下游做全局偏移
                        if clip_key in video_info:
                            seg_id = f"{base}_{three_min_idx}_{thirty_idx}"
                        else:
                            seg_id = f"{base}FULL_{three_min_idx}_{thirty_idx}"
                        seg_urls[seg_id] = url
                    # 已是三元形式：base_三分钟序号_30秒序号
                    elif len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                        base = '_'.join(parts[:-2])
                        three_min_idx = int(parts[-2])
                        thirty_idx = int(parts[-1])
                        video_info = _video_map.get(base)
                        if not video_info:
                            print(f"[Top5] Skip: video '{base}' not found in video_urls.py")
                            continue
                        clip_key = f"{base}_{three_min_idx}.mp4"
                        url = video_info.get(clip_key) or next(iter(video_info.values()), None)
                        if not url:
                            continue
                        if clip_key in video_info:
                            seg_id = f"{base}_{three_min_idx}_{thirty_idx}"
                        else:
                            seg_id = f"{base}FULL_{three_min_idx}_{thirty_idx}"
                        seg_urls[seg_id] = url
                    else:
                        # 最宽松兼容：无法解析时，尝试把前缀当作 video id
                        base = parts[0]
                        video_info = _video_map.get(base)
                        if not video_info:
                            continue
                        url = next(iter(video_info.values()), None)
                        if not url:
                            continue
                        seg_urls[name] = url
                except Exception:
                    continue
            return seg_urls

        processed_json_files: set[str] = set()
        json_files = [single_json_file] if single_json_file else glob.glob(os.path.join(input_base_dir, "*.json"))
        if not json_files:
            print(f"[Top5] No JSON files found under: {input_base_dir}")
            return
        for json_file_path in json_files:
            norm_path = os.path.abspath(json_file_path)
            if norm_path in processed_json_files:
                # 已处理过
                continue
            processed_json_files.add(norm_path)

            print(f"\nProcessing file: {norm_path}")
            try:
                with open(norm_path, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
            except Exception as e:
                print(f"Error reading JSON {norm_path}: {e}")
                continue

            # 仅处理包含 query + top5 segments 的结构（兼容两种命名）
            has_top5 = 'top5_segment_names' in qa_data or 'top5_segments' in qa_data
            if not (isinstance(qa_data, dict) and 'query' in qa_data and has_top5):
                print(f"[Skip] Not a 'query+top5_segments' JSON: {json_file_path}")
                continue

            # 规范化问题文本
            question = normalize_question_text(qa_data.get('query', ''))

            # 根据 top5 段名构建 segment -> url 映射（支持 'top5_segments' 字段）
            segment_names = qa_data.get('top5_segment_names') or qa_data.get('top5_segments') or []
            # Only use top-k of the provided top5 list for topk experiments
            try:
                segment_names = (segment_names or [])[:topk]
            except Exception:
                segment_names = segment_names
            segment_urls = _build_segment_urls_from_top5(segment_names)
            # If rerank requested, randomize the order of the segments before passing downstream
            if getattr(args, 'rerank', False) and segment_urls:
                keys = list(segment_urls.keys())
                keys = _deranged_order(keys)
                segment_urls = {k: segment_urls[k] for k in keys}
                try:
                    print(f"[Rerank] top5 {os.path.basename(json_file_path)} order: {list(segment_urls.keys())}")
                except Exception:
                    # best-effort logging; do not fail processing
                    print(f"[Rerank] top5 order: {list(segment_urls.keys())}")
            if not segment_urls:
                print(f"[Skip] No resolvable segments in {json_file_path}")
                continue

            # 输出位置
            relative_path = os.path.basename(json_file_path) if single_json_file else os.path.relpath(json_file_path, input_base_dir)
            output_file_path = os.path.join(output_base_dir, relative_path)
            failure_log_path = output_file_path + ".failures.jsonl"

            # 读取已存在结果（支持断点续跑）或强制重跑
            existing_results = []
            existing_questions = set()
            if not force_rerun:
                try:
                    if os.path.exists(output_file_path):
                        with open(output_file_path, "r", encoding="utf-8") as f:
                            existing_results = json.load(f)
                            if isinstance(existing_results, dict):
                                existing_results = [existing_results]
                    existing_results, existing_questions = _prepare_existing_results(existing_results, output_file_path)
                except Exception:
                    try:
                        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        os.rename(output_file_path, output_file_path + ".bak")
                        print(f"[Resume] Corrupted result file backed up: {output_file_path}.bak")
                    except Exception:
                        pass
                    existing_results = []

            def append_failure_top(idx: int, question_text: str, err: Exception):
                rec = {
                    "video_id": "_top5_mode_",
                    "result_rel_path": relative_path,
                    "index": idx,
                    "question": question_text,
                    "error": str(err),
                    "traceback": traceback.format_exc(),
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
                try:
                    os.makedirs(os.path.dirname(failure_log_path), exist_ok=True)
                    with open(failure_log_path, "a", encoding="utf-8") as flog:
                        flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    master_fail = os.path.join(output_base_dir, "_failures.jsonl")
                    with open(master_fail, "a", encoding="utf-8") as mlog:
                        mlog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass

            if (not force_rerun) and (question in existing_questions):
                print(f"[Skip] {question[:50]}... (cached non-empty answer)")
                continue

            try:
                touched_paths_for_top: set = set()
                answer_obj = await process_question(
                    question,
                    segment_urls,
                    work_dir,
                    asr_model,
                    llm_cfg,
                    touched_paths=touched_paths_for_top,
                    base_mode=args.base_mode,
                    ablation=args.ablation,
                    ablation_ocr=getattr(args, 'ablationocr', False),
                    ablation_det=getattr(args, 'ablationdet', False)
                )
                clean_ans = extract_final_answer(answer_obj.get("answer", ""))
                # 构建基础输出记录
                rec = {"question": question, "answer": clean_ans, "mode": current_mode}

                # 若 JSON 中包含 video_name 和 source_file，则尝试载入 Bench 中对应的标准答案并评估
                try:
                    video_name = qa_data.get('video_name')
                    source_file = qa_data.get('source_file')
                    if video_name and source_file:
                        # GT path: strictly from config
                        if not CONFIG_GT_PATH:
                            raise RuntimeError("GT_DATA_PATH must be set in videorag._config and non-empty")
                        gt_path = os.path.join(CONFIG_GT_PATH, str(video_name), str(source_file))
                        if os.path.exists(gt_path):
                            try:
                                from videorag.evaluate.evaluate import evaluate_one, normalize_keypoints
                                with open(gt_path, 'r', encoding='utf-8') as gf:
                                    gt_data = json.load(gf)
                                gt_items = gt_data if isinstance(gt_data, list) else [gt_data]
                                matched = None
                                for e in gt_items:
                                    if isinstance(e, dict) and str(e.get('question', '')) == question:
                                        matched = e
                                        break
                                if matched:
                                    raw_kp = matched.get('keypoints', matched.get('keypoint', {}))
                                    kp = normalize_keypoints(raw_kp, gt_path)
                                    gt_answer = str(matched.get('answer', ''))
                                    metrics = evaluate_one(question, kp, clean_ans, gt_answer)
                                    # 将评估结果并入记录
                                    rec.update({
                                        "covered_video_keypoints": int(metrics.get("covered_video_keypoints", 0)),
                                        "covered_text_keypoints": int(metrics.get("covered_text_keypoints", 0)),
                                        "gt_video_n": int(metrics.get("gt_video_n", 0)),
                                        "gt_text_n": int(metrics.get("gt_text_n", 0)),
                                        "total_claimed_keypoints": int(metrics.get("total_claimed_keypoints", 0)),
                                        "likert_score": int(metrics.get("likert_score", 0)),
                                        "likert_subscores": metrics.get("likert_subscores", {}),
                                        "rouge_l_f": (round(metrics.get("rouge_l_f"), 6) if isinstance(metrics.get("rouge_l_f"), float) else None),
                                        "st_cosine_score": (round(metrics.get("st_cosine_score"), 6) if isinstance(metrics.get("st_cosine_score"), float) else None),
                                        "eval_reasoning": metrics.get("eval_reasoning", ""),
                                    })
                                else:
                                    print(f"[Eval] GT question not found in {gt_path}: '{question[:60]}'")
                            except Exception as _e:
                                print(f"[Eval] Failed to evaluate using GT {gt_path}: {_e}")
                        else:
                            print(f"[Eval] GT file not found: {gt_path}")
                except Exception:
                    # 确保评估失败不影响主流程
                    pass

                output_data = ([] if force_rerun else existing_results[:]) + [rec]
                # Ensure parent directory exists before writing (fix: Errno 2 no such file or directory)
                try:
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                except Exception:
                    pass
                with open(output_file_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
            except Exception as err:
                print(f"[Top5 Error] {err}")
                append_failure_top(0, str(question), err)

        return

    for video_id in sorted(video_id_dirs):
        video_dir_path = os.path.join(input_base_dir, video_id)
        json_files = sorted(glob.glob(os.path.join(video_dir_path, "*.json")))
        touched_paths_for_video: set = set()
        local_video_path = os.path.join(input_base_dir, f"{video_id}.mp4")
        if not os.path.isfile(local_video_path):
            print(f"SKIPPING: Local video not found for {video_id}: {local_video_path}")
            continue

        segment_urls = _build_local_segment_urls(video_id, local_video_path)

        if not segment_urls:
            print(f"SKIPPING: No segments found for video ID {video_id}")
            continue

        processed_json_files: set[str] = set()
        for json_file_path in json_files:
            norm_path = os.path.abspath(json_file_path)
            if norm_path in processed_json_files:
                continue
            processed_json_files.add(norm_path)

            print(f"\nProcessing file: {norm_path}")
            try:
                with open(norm_path, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
            except Exception as e:
                print(f"Error reading JSON {norm_path}: {e}")
                continue
            
            # Resolve output path first for resume & logging
            relative_path = os.path.relpath(json_file_path, input_base_dir)
            output_file_path = os.path.join(output_base_dir, relative_path)
            failure_log_path = output_file_path + ".failures.jsonl"

            # Resume: load existing results if any (unless force rerun)
            existing_results = []
            existing_questions = set()
            if not force_rerun:
                try:
                    if os.path.exists(output_file_path):
                        with open(output_file_path, "r", encoding="utf-8") as f:
                            existing_results = json.load(f)
                            if isinstance(existing_results, dict):
                                existing_results = [existing_results]
                    existing_results, existing_questions = _prepare_existing_results(existing_results, output_file_path)
                except Exception:
                    # If corrupted, back it up and start fresh
                    try:
                        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        os.rename(output_file_path, output_file_path + ".bak")
                        print(f"[Resume] Corrupted result file backed up: {output_file_path}.bak")
                    except Exception:
                        pass
                    existing_results = []
                    existing_index_to_pos = {}

            def append_failure(idx: int, question_text: str, err: Exception):
                rec = {
                    "video_id": video_id,
                    "result_rel_path": relative_path,
                    "index": idx,
                    "question": question_text,
                    "error": str(err),
                    "traceback": traceback.format_exc(),
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
                try:
                    os.makedirs(os.path.dirname(failure_log_path), exist_ok=True)
                    with open(failure_log_path, "a", encoding="utf-8") as flog:
                        flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    # Also write to a model-level master failures file
                    master_fail = os.path.join(output_base_dir, "_failures.jsonl")
                    with open(master_fail, "a", encoding="utf-8") as mlog:
                        mlog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass

            output_data = [] if force_rerun else existing_results[:]  # start from existing or reset when forced

            question_entries = _iter_question_entries(qa_data, norm_path)
            
            for i, question in question_entries:

                print(f"\n--- Processing Q{i}: {question} for video {video_id} ---")

                # Resume: skip if already completed (has question in existing results)
                if (not force_rerun) and (question in existing_questions):
                    print(f"[Skip] Q{i}: {question[:50]}... (cached non-empty answer)")
                    continue

                try:
                    # The process_question function encapsulates the logic for a single query
                    answer_obj = await process_question(
                        question,
                        segment_urls,
                        work_dir,
                        asr_model,
                        llm_cfg,
                        touched_paths=touched_paths_for_video,
                        base_mode=args.base_mode,
                        ablation=args.ablation,
                        ablation_ocr=getattr(args, 'ablationocr', False),
                        ablation_det=getattr(args, 'ablationdet', False)
                    )
                    record = {
                        "question": question,
                        "answer": extract_final_answer(answer_obj.get("answer", "")),
                        "mode": current_mode,
                    }
                    output_data.append(record)
                    # Persist progress incrementally after each success
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                except Exception as err:
                    print(f"[Batch Error-Q{i}] {err}")
                    append_failure(i, str(question), err)
                    # Do not write a result record, so that next run retries this question

            # Final save (already incrementally saved after each success)
            try:
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                print(f"[Saved] {output_file_path}")
            except Exception as err:
                print(f"[Save Error] {err}")

        # After all JSONs for this video_id, check completeness; if all done and no pending gaps, clean caches
        try:
            all_done = True
            for jf in json_files:
                # Load input questions
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        qa_data = json.load(f)
                except Exception:
                    all_done = False
                    break
                question_entries = _iter_question_entries(qa_data, jf)
                expected_n = len({question for _, question in question_entries})
                # Corresponding results
                relp = os.path.relpath(jf, input_base_dir)
                outp = os.path.join(output_base_dir, relp)
                if not os.path.exists(outp):
                    all_done = False
                    break
                try:
                    with open(outp, "r", encoding="utf-8") as f:
                        res_list = json.load(f)
                        if isinstance(res_list, dict):
                            res_list = [res_list]
                except Exception:
                    all_done = False
                    break
                present_questions = set()
                for item in res_list:
                    if isinstance(item, dict) and "question" in item:
                        present_questions.add(item["question"])
                # Require all questions present
                if len(present_questions) < expected_n:
                    all_done = False
                    break

            if all_done and touched_paths_for_video:
                removed = 0
                for p in list(touched_paths_for_video):
                    try:
                        real_path = os.path.realpath(p)
                        real_workdir = os.path.realpath(work_dir)
                        if os.path.commonpath([real_path, real_workdir]) != real_workdir:
                            continue
                        if os.path.isdir(p):
                            shutil.rmtree(p, ignore_errors=True)
                            removed += 1
                        elif os.path.isfile(p):
                            os.remove(p)
                            removed += 1
                    except Exception:
                        pass
                print(f"[Cache] Cleaned {video_id} ({removed} items)")
        except Exception as err:
            print(f"[Cache] Skip cleaning for {video_id}: {err}")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    try:
        import asyncio
        asyncio.run(batch_main())
    except Exception as e:
        print(f"[Batch Error] {e}")
