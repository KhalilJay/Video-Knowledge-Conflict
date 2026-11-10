"""Runtime constants and project paths."""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE_CASE_DIR = PROJECT_ROOT / "dataset" / "source_cases"
DEFAULT_CASE_DIR = PROJECT_ROOT / "dataset" / "cases"
RESULTS_DIR = PROJECT_ROOT / "results"

AUTHOR_JUDGE_MODEL = "openai/gpt-5.6-luna-pro"
GEMINI_QA_MODEL = "google/gemini-3.1-pro-preview"
GEMINI_ARGON_QA_MODEL = "google/gemini-4-argon"
QWEN_QA_MODEL = "qwen3.8-max"
KIMI_QA_MODEL = "kimi-k3"
COSMOS_NANO_QA_MODEL = "nvidia/Cosmos3-Nano"
COSMOS_SUPER_QA_MODEL = "nvidia/Cosmos3-Super"
COSMOS_QA_MODELS = (COSMOS_NANO_QA_MODEL, COSMOS_SUPER_QA_MODEL)
GEMMA_QA_MODEL = "google/gemma-4-31B-it"
GEMMA_12B_QA_MODEL = "google/gemma-4-12B-it"
GEMMA_QA_MODELS = (GEMMA_QA_MODEL, GEMMA_12B_QA_MODEL)
QWEN38_QA_MODEL = "Qwen/Qwen3.8-27B"
QWEN3_VL_INSTRUCT_QA_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
QWEN3_VL_THINKING_QA_MODEL = "Qwen/Qwen3-VL-32B-Thinking"
QWEN3_VL_8B_INSTRUCT_QA_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
QWEN3_VL_8B_THINKING_QA_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
QWEN3_VL_INSTRUCT_QA_MODELS = (QWEN3_VL_INSTRUCT_QA_MODEL, QWEN3_VL_8B_INSTRUCT_QA_MODEL)
QWEN3_VL_THINKING_QA_MODELS = (QWEN3_VL_THINKING_QA_MODEL, QWEN3_VL_8B_THINKING_QA_MODEL)
QWEN3_VL_QA_MODELS = (*QWEN3_VL_INSTRUCT_QA_MODELS, *QWEN3_VL_THINKING_QA_MODELS)
DEFAULT_QA_MODEL = GEMINI_QA_MODEL
CLI_QA_MODELS = (
    "gemini-3.1-pro", "gemini-4-argon", QWEN_QA_MODEL, KIMI_QA_MODEL,
    "cosmos3-nano", "cosmos3-super", "gemma4-31b", "gemma4-12b", "qwen3.8-27b",
    "qwen3-vl-32b-instruct", "qwen3-vl-32b-thinking",
    "qwen3-vl-8b-instruct", "qwen3-vl-8b-thinking",
)
CLI_QA_MODEL_IDS = {
    "gemini-3.1-pro": GEMINI_QA_MODEL,
    "gemini-4-argon": GEMINI_ARGON_QA_MODEL,
    QWEN_QA_MODEL: QWEN_QA_MODEL,
    KIMI_QA_MODEL: KIMI_QA_MODEL,
    "cosmos3-nano": COSMOS_NANO_QA_MODEL,
    "cosmos3-super": COSMOS_SUPER_QA_MODEL,
    "gemma4-31b": GEMMA_QA_MODEL,
    "gemma4-12b": GEMMA_12B_QA_MODEL,
    "gemma4": GEMMA_QA_MODEL,
    "gemma": GEMMA_QA_MODEL,
    "qwen3.8-27b": QWEN38_QA_MODEL,
    "qwen3-vl-32b-instruct": QWEN3_VL_INSTRUCT_QA_MODEL,
    "qwen3-vl-32b-thinking": QWEN3_VL_THINKING_QA_MODEL,
    "qwen3-vl-8b-instruct": QWEN3_VL_8B_INSTRUCT_QA_MODEL,
    "qwen3-vl-8b-thinking": QWEN3_VL_8B_THINKING_QA_MODEL,
}
QA_MODEL_SLUGS = {
    GEMINI_QA_MODEL: "gemini3_1_pro",
    GEMINI_ARGON_QA_MODEL: "gemini4_argon",
    QWEN_QA_MODEL: "qwen",
    KIMI_QA_MODEL: "kimi",
    COSMOS_NANO_QA_MODEL: "cosmos_nano",
    COSMOS_SUPER_QA_MODEL: "cosmos_super",
    GEMMA_QA_MODEL: "gemma",
    GEMMA_12B_QA_MODEL: "gemma4_12b",
    QWEN38_QA_MODEL: "qwen38_27b",
    QWEN3_VL_INSTRUCT_QA_MODEL: "qwen3_vl_32b_instruct",
    QWEN3_VL_THINKING_QA_MODEL: "qwen3_vl_32b_thinking",
    QWEN3_VL_8B_INSTRUCT_QA_MODEL: "qwen3_vl_8b_instruct",
    QWEN3_VL_8B_THINKING_QA_MODEL: "qwen3_vl_8b_thinking",
}
CLI_THINKING_EFFORTS = ("none", "default")
FAIRY_TALE_GROUP = "classic_fairy_tale_film_conflicts"

SEEDANCE_MODEL = "doubao-seedance-2-5-260628"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_CASE_WORKERS = 8
DEFAULT_REQUEST_WORKERS = 24
DEFAULT_REQUEST_RPM = 20.0
DEFAULT_COSMOS_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_COSMOS_WORKERS = 8
DEFAULT_COSMOS_MAX_TOKENS = 8192
DEFAULT_GEMMA_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_GEMMA_WORKERS = 4
DEFAULT_GEMMA_MAX_TOKENS = 8192
DEFAULT_QWEN38_BASE_URL = "http://127.0.0.1:8002/v1"
DEFAULT_QWEN38_WORKERS = 1
DEFAULT_QWEN38_MAX_TOKENS = 8192
DEFAULT_QWEN3_VL_BASE_URL = "http://127.0.0.1:8003/v1"
DEFAULT_QWEN3_VL_WORKERS = 1
DEFAULT_QWEN3_VL_MAX_TOKENS = 8192
DEFAULT_THINKING_TOKEN_BUDGET = 6144
DEFAULT_SEEDANCE_TOTAL_WORKERS = 12

CASE_SCHEMA_VERSION = "4.0"
QUESTION_LIMIT = 3
CONFLICT_VIDEO_LIMIT = 5
TOTAL_VIDEO_LIMIT = CONFLICT_VIDEO_LIMIT + 1
VIDEO_ROLES = {"conflict", "control"}
VIDEO_STATUSES = {"pending", "submitted", "ready", "failed"}
VERDICTS = {"context_grounded", "knowledge_trapped", "ambiguous_or_unjudgeable"}
LEGACY_VERDICTS = {"video_grounded"}
ACCEPTED_VERDICTS = VERDICTS | LEGACY_VERDICTS
INPUT_MODES = ("video", "description", "question_only")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
