
# videorag/_config.py
# This file holds shared, static configuration variables to prevent circular imports.

# 路径常量
INPUT_DATA_PATH = " "  # 输入数据地址
GT_DATA_PATH = " "     # gt地址
RESULT_PATH = " "     # 保存和评估地址

SENT_TRANSFORMER_MODEL_PATH = " "
WHISPER_MODEL_PATH = " "
YOLOV8_MODEL_PATH = " "

# Deprecated placeholder. Do not import this constant; prefer using
# the environment variable `MINICPM_MODEL_PATH` or load MiniCPM via Ollama on demand.
MINICPM_MODEL_PATH = None

# --- Frame refinement / de-dup config defaults ---
# 感知哈希汉明距离阈值（越小越严格）
DEDUP_PHASH_THRESHOLD_DEFAULT = 5
# 是否输出调试统计
DEDUP_DEBUG = False
# 扩展帧数映射：键 = 分数(0~10)，值 = 目标帧数（示例：分数高 → 需要更多帧；你可按需再调）
FRAME_COUNT_MAPPING_EXTENDED = {
    0: 40,
    1: 36,
    2: 32,
    3: 28,
    4: 24,
    5: 20,
}

# ------------------------------------------------------------------
# New refinement / runtime control defaults (centralized tuning)
#   timeout = float(os.environ.get("REFINER_TIMEOUT_SECONDS", REFINER_TIMEOUT_SECONDS_DEFAULT))
# ------------------------------------------------------------------

# 细化阶段上下文评估（计划生成）超时（秒）
REFINER_TIMEOUT_SECONDS_DEFAULT = 60
# 超时后的回退策略: "final" (直接跳过 refine) 或 "refine" (全量 refine)
REFINER_TIMEOUT_FALLBACK_DEFAULT = "final"
# 并发最大并行评估任务
REFINER_MAX_PARALLEL_DEFAULT = 1
# 关键词（DET 用）生成超时（秒）
REFINER_KEYWORD_TIMEOUT_SECONDS_DEFAULT = 30

# OCR / DET 抽帧默认数
REFINE_OCR_FRAMES_DEFAULT = 6
REFINE_DET_FRAMES_DEFAULT = 8

# 差异增量摘要（diff summary）相关
REFINE_DIFF_SUMMARY_MAX_CHARS_DEFAULT = 160
REFINE_DIFF_MAX_FRAMES_DEFAULT = 12
REFINE_DIFF_DECAY_WINDOW_DEFAULT = 3

# Keyframe -> token 估算系数 (approx tokens per raw byte, 用于日志统计)
KEYFRAME_TOKENS_PER_BYTE_DEFAULT = 1.0 / 3.0

# 可选：最大目标帧数上限保护（避免异常配置导致过度抽帧）
REFINE_GLOBAL_MAX_FRAMES_PER_30S_DEFAULT = 40

__all__ = [
    "SENT_TRANSFORMER_MODEL_PATH",
    "WHISPER_MODEL_PATH",
    "YOLOV8_MODEL_PATH",
    "DEDUP_PHASH_THRESHOLD_DEFAULT",
    "DEDUP_DEBUG",
    "FRAME_COUNT_MAPPING_EXTENDED",
    # New exports
    "REFINER_TIMEOUT_SECONDS_DEFAULT",
    "REFINER_TIMEOUT_FALLBACK_DEFAULT",
    "REFINER_MAX_PARALLEL_DEFAULT",
    "REFINER_KEYWORD_TIMEOUT_SECONDS_DEFAULT",
    "REFINE_OCR_FRAMES_DEFAULT",
    "REFINE_DET_FRAMES_DEFAULT",
    "REFINE_DIFF_SUMMARY_MAX_CHARS_DEFAULT",
    "REFINE_DIFF_MAX_FRAMES_DEFAULT",
    "REFINE_DIFF_DECAY_WINDOW_DEFAULT",
    "KEYFRAME_TOKENS_PER_BYTE_DEFAULT",
    "REFINE_GLOBAL_MAX_FRAMES_PER_30S_DEFAULT",
    # Path exports
    "INPUT_DATA_PATH",
    "GT_DATA_PATH",
    "RESULT_PATH",
]
