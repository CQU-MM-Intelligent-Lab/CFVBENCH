import ast
import os


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


REPO_ROOT = _repo_root()


def _resolve_value(env_name: str, default_value: str) -> str:
    raw_value = os.environ.get(env_name, "").strip()
    if raw_value:
        return raw_value
    return default_value


def _resolve_repo_path(env_name: str, default_value: str) -> str:
    raw_value = os.environ.get(env_name, "").strip()
    value = raw_value or default_value
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(REPO_ROOT, value))


def _resolve_optional_repo_path(env_name: str, default_value: str | None) -> str | None:
    raw_value = os.environ.get(env_name, "").strip()
    value = raw_value or (default_value or "")
    if not value:
        return None
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(REPO_ROOT, value))


def _resolve_int_list(env_names: list[str], default_value: list[int]) -> list[int]:
    for env_name in env_names:
        raw_value = os.environ.get(env_name, "").strip()
        if not raw_value:
            continue
        try:
            parsed = ast.literal_eval(raw_value) if raw_value.startswith(("[", "(")) else raw_value.split(",")
            if isinstance(parsed, int):
                parsed = [parsed]
            if not isinstance(parsed, (list, tuple)):
                raise ValueError(f"{env_name} must be a list/tuple or comma-separated string")
            return [int(item) for item in parsed]
        except Exception as exc:
            print(f"[GPU][Warn] Invalid {env_name}={raw_value!r}: {exc}. Falling back to default {default_value}.")
            break
    return list(default_value)


def _resolve_str_list(env_names: list[str], default_value: list[str]) -> list[str]:
    for env_name in env_names:
        raw_value = os.environ.get(env_name, "").strip()
        if not raw_value:
            continue
        try:
            parsed = ast.literal_eval(raw_value) if raw_value.startswith(("[", "(")) else raw_value.split(",")
            if isinstance(parsed, str):
                parsed = [parsed]
            if not isinstance(parsed, (list, tuple)):
                raise ValueError(f"{env_name} must be a list/tuple or comma-separated string")
            return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception as exc:
            print(f"[Config][Warn] Invalid {env_name}={raw_value!r}: {exc}. Falling back to default {default_value}.")
            break
    return list(default_value)


def apply_configured_gpu_environment() -> str | None:
    manual_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if manual_visible:
        return manual_visible
    if not GPU_USE:
        return None
    visible = ",".join(str(idx) for idx in GPU_USE)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return visible


def _parse_visible_cuda_devices(raw_value: str) -> list[str]:
    raw = (raw_value or "").strip()
    if not raw:
        return []
    if raw in {"-1", "none", "cpu"}:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def get_effective_visible_gpu_ids() -> list[str]:
    manual_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if manual_visible:
        return _parse_visible_cuda_devices(manual_visible)
    return [str(idx) for idx in GPU_USE]


def get_effective_visible_gpu_count() -> int:
    return len(get_effective_visible_gpu_ids())


def get_transformers_device_map():
    return "auto" if get_effective_visible_gpu_count() > 0 else {"": "cpu"}


# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------
DEFAULT_BENCH_DIR = "Bench"
DEFAULT_WORKDIR = "videorag-workdir"
DEFAULT_RESULT_PATH = os.path.join(DEFAULT_WORKDIR, "results")

BENCH_DIR = _resolve_repo_path("VIDEORAG_BENCH_DIR", DEFAULT_BENCH_DIR)
WORKDIR_PATH = _resolve_repo_path("VIDEORAG_WORKDIR", DEFAULT_WORKDIR)
RESULT_PATH = _resolve_repo_path("VIDEORAG_RESULT_PATH", DEFAULT_RESULT_PATH)

INPUT_DATA_PATH = BENCH_DIR
GT_DATA_PATH = BENCH_DIR


# ------------------------------------------------------------------
# Editable model names
# Environment variables with the same names still take precedence.
# ------------------------------------------------------------------
# Edit this list to choose visible GPU cards, e.g. [0, 1, 2, 3].
# Leave it as [] to keep the current external CUDA_VISIBLE_DEVICES / CPU behavior.
DEFAULT_GPU_USE: list[int] = [2,3]
DEFAULT_OLLAMA_CHAT_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_REFINER_OLLAMA_MODEL = DEFAULT_OLLAMA_CHAT_MODEL
DEFAULT_EVAL_LLM_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_VISION_MODEL_HINTS: list[str] = ["qwen3.5:4b"]
DEFAULT_CUSTOM_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-all"
DEFAULT_WHISPER_FALLBACK_MODEL_ID = "distil-large-v3"

GPU_USE = _resolve_int_list(["VIDEORAG_GPU_USE", "GPU_USE"], DEFAULT_GPU_USE)
OLLAMA_CHAT_MODEL = _resolve_value("OLLAMA_CHAT_MODEL", DEFAULT_OLLAMA_CHAT_MODEL)
OLLAMA_EMBED_MODEL = _resolve_value("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)
REFINER_OLLAMA_MODEL = _resolve_value("REFINER_OLLAMA_MODEL", DEFAULT_REFINER_OLLAMA_MODEL)
EVAL_LLM_MODEL = _resolve_value("EVAL_LLM_MODEL", DEFAULT_EVAL_LLM_MODEL)
OLLAMA_VISION_MODEL_HINTS = _resolve_str_list(
    ["VIDEORAG_OLLAMA_VISION_MODEL_HINTS", "OLLAMA_VISION_MODEL_HINTS"],
    DEFAULT_OLLAMA_VISION_MODEL_HINTS,
)
CUSTOM_OPENAI_MODEL = _resolve_value("CUSTOM_OPENAI_MODEL", DEFAULT_CUSTOM_OPENAI_MODEL)
ANTHROPIC_MODEL = _resolve_value("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
WHISPER_FALLBACK_MODEL_ID = _resolve_value(
    "VIDEORAG_WHISPER_FALLBACK_MODEL_ID",
    DEFAULT_WHISPER_FALLBACK_MODEL_ID,
)


# ------------------------------------------------------------------
# Editable local model paths
# Set DEFAULT_MINICPM_MODEL_PATH to a repo-relative or absolute path if you
# want local MiniCPM captioning without relying on environment variables.
# ------------------------------------------------------------------
DEFAULT_SENT_TRANSFORMER_MODEL_PATH = "/root/models/sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_WHISPER_MODEL_PATH = "/root/models/faster-distil-whisper-large-v3"
DEFAULT_YOLOV8_MODEL_PATH = "/root/models/yolov8m-worldv2.pt"
DEFAULT_MINICPM_MODEL_PATH = "/root/models/MiniCPM-V-4_5-int4"

SENT_TRANSFORMER_MODEL_PATH = _resolve_repo_path(
    "VIDEORAG_SENT_TRANSFORMER_MODEL_PATH",
    DEFAULT_SENT_TRANSFORMER_MODEL_PATH,
)
WHISPER_MODEL_PATH = _resolve_repo_path(
    "VIDEORAG_WHISPER_MODEL_PATH",
    DEFAULT_WHISPER_MODEL_PATH,
)
YOLOV8_MODEL_PATH = _resolve_repo_path(
    "VIDEORAG_YOLOV8_MODEL_PATH",
    DEFAULT_YOLOV8_MODEL_PATH,
)
MINICPM_MODEL_PATH = _resolve_optional_repo_path(
    "MINICPM_MODEL_PATH",
    DEFAULT_MINICPM_MODEL_PATH,
)



# ------------------------------------------------------------------
# Frame refinement / de-dup defaults
# ------------------------------------------------------------------
DEDUP_PHASH_THRESHOLD_DEFAULT = 5
DEDUP_DEBUG = False
FRAME_COUNT_MAPPING_EXTENDED = {
    0: 40,
    1: 36,
    2: 32,
    3: 28,
    4: 24,
    5: 20,
}


# ------------------------------------------------------------------
# Refinement / runtime controls
# ------------------------------------------------------------------
REFINER_TIMEOUT_SECONDS_DEFAULT = 60
REFINER_TIMEOUT_FALLBACK_DEFAULT = "final"
REFINER_MAX_PARALLEL_DEFAULT = 1
REFINER_KEYWORD_TIMEOUT_SECONDS_DEFAULT = 30
REFINER_WARMUP_ON_FIRST_CALL_DEFAULT = True
REFINER_WARMUP_TIMEOUT_SECONDS_DEFAULT = 30
REFINER_RETRY_TIMEOUT_SECONDS_DEFAULT = 90
REFINER_RETRY_ATTEMPTS_DEFAULT = 3
VISION_FRAME_TARGET_HEIGHT_DEFAULT = 96
VISION_CAPTION_MAX_FRAMES_DEFAULT = 5
VISION_CAPTION_WARMUP_TIMEOUT_SECONDS_DEFAULT = 15
VISION_CAPTION_FRAME_TIMEOUT_SECONDS_DEFAULT = 30
VISION_CAPTION_SYNTH_TIMEOUT_SECONDS_DEFAULT = 30
VISION_CAPTION_RETRY_ATTEMPTS_DEFAULT = 3
VISION_CAPTION_SYNTH_RETRY_ATTEMPTS_DEFAULT = 1
VISION_CAPTION_MAX_PARALLEL_DEFAULT = 0
VISION_CAPTION_OLLAMA_MAX_PARALLEL_DEFAULT = 1
VISION_CAPTION_PARALLEL_EST_MEMORY_GB_DEFAULT = 4.0
VISION_CAPTION_PARALLEL_RESERVE_GB_DEFAULT = 2.0
VISION_CAPTION_OOM_POLL_INTERVAL_SECONDS_DEFAULT = 2.0
VISION_CAPTION_OOM_REQUEUE_BACKOFF_SECONDS_DEFAULT = 1.0

REFINE_OCR_FRAMES_DEFAULT = 6
REFINE_DET_FRAMES_DEFAULT = 8

REFINE_DIFF_SUMMARY_MAX_CHARS_DEFAULT = 160
REFINE_DIFF_MAX_FRAMES_DEFAULT = 12
REFINE_DIFF_DECAY_WINDOW_DEFAULT = 3

KEYFRAME_TOKENS_PER_BYTE_DEFAULT = 1.0 / 3.0
REFINE_GLOBAL_MAX_FRAMES_PER_30S_DEFAULT = 40


__all__ = [
    "REPO_ROOT",
    "DEFAULT_BENCH_DIR",
    "DEFAULT_WORKDIR",
    "DEFAULT_RESULT_PATH",
    "BENCH_DIR",
    "WORKDIR_PATH",
    "RESULT_PATH",
    "INPUT_DATA_PATH",
    "GT_DATA_PATH",
    "DEFAULT_GPU_USE",
    "GPU_USE",
    "get_effective_visible_gpu_ids",
    "get_effective_visible_gpu_count",
    "get_transformers_device_map",
    "DEFAULT_OLLAMA_CHAT_MODEL",
    "DEFAULT_OLLAMA_EMBED_MODEL",
    "DEFAULT_REFINER_OLLAMA_MODEL",
    "DEFAULT_EVAL_LLM_MODEL",
    "DEFAULT_OLLAMA_VISION_MODEL_HINTS",
    "DEFAULT_CUSTOM_OPENAI_MODEL",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_WHISPER_FALLBACK_MODEL_ID",
    "OLLAMA_CHAT_MODEL",
    "OLLAMA_EMBED_MODEL",
    "REFINER_OLLAMA_MODEL",
    "EVAL_LLM_MODEL",
    "OLLAMA_VISION_MODEL_HINTS",
    "CUSTOM_OPENAI_MODEL",
    "ANTHROPIC_MODEL",
    "WHISPER_FALLBACK_MODEL_ID",
    "apply_configured_gpu_environment",
    "DEFAULT_SENT_TRANSFORMER_MODEL_PATH",
    "DEFAULT_WHISPER_MODEL_PATH",
    "DEFAULT_YOLOV8_MODEL_PATH",
    "DEFAULT_MINICPM_MODEL_PATH",
    "SENT_TRANSFORMER_MODEL_PATH",
    "WHISPER_MODEL_PATH",
    "YOLOV8_MODEL_PATH",
    "MINICPM_MODEL_PATH",
    "DEDUP_PHASH_THRESHOLD_DEFAULT",
    "DEDUP_DEBUG",
    "FRAME_COUNT_MAPPING_EXTENDED",
    "REFINER_TIMEOUT_SECONDS_DEFAULT",
    "REFINER_TIMEOUT_FALLBACK_DEFAULT",
    "REFINER_MAX_PARALLEL_DEFAULT",
    "REFINER_KEYWORD_TIMEOUT_SECONDS_DEFAULT",
    "REFINER_WARMUP_ON_FIRST_CALL_DEFAULT",
    "REFINER_WARMUP_TIMEOUT_SECONDS_DEFAULT",
    "REFINER_RETRY_TIMEOUT_SECONDS_DEFAULT",
    "REFINER_RETRY_ATTEMPTS_DEFAULT",
    "VISION_FRAME_TARGET_HEIGHT_DEFAULT",
    "VISION_CAPTION_MAX_FRAMES_DEFAULT",
    "VISION_CAPTION_WARMUP_TIMEOUT_SECONDS_DEFAULT",
    "VISION_CAPTION_FRAME_TIMEOUT_SECONDS_DEFAULT",
    "VISION_CAPTION_SYNTH_TIMEOUT_SECONDS_DEFAULT",
    "VISION_CAPTION_RETRY_ATTEMPTS_DEFAULT",
    "VISION_CAPTION_SYNTH_RETRY_ATTEMPTS_DEFAULT",
    "VISION_CAPTION_MAX_PARALLEL_DEFAULT",
    "VISION_CAPTION_OLLAMA_MAX_PARALLEL_DEFAULT",
    "VISION_CAPTION_PARALLEL_EST_MEMORY_GB_DEFAULT",
    "VISION_CAPTION_PARALLEL_RESERVE_GB_DEFAULT",
    "VISION_CAPTION_OOM_POLL_INTERVAL_SECONDS_DEFAULT",
    "VISION_CAPTION_OOM_REQUEUE_BACKOFF_SECONDS_DEFAULT",
    "REFINE_OCR_FRAMES_DEFAULT",
    "REFINE_DET_FRAMES_DEFAULT",
    "REFINE_DIFF_SUMMARY_MAX_CHARS_DEFAULT",
    "REFINE_DIFF_MAX_FRAMES_DEFAULT",
    "REFINE_DIFF_DECAY_WINDOW_DEFAULT",
    "KEYFRAME_TOKENS_PER_BYTE_DEFAULT",
    "REFINE_GLOBAL_MAX_FRAMES_PER_30S_DEFAULT",
]
