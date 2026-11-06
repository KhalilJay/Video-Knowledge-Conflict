#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL-32B：默认 Instruct，单张 A100 80GB，BF16，16K，端口 8003。
# 启动：bash scripts/start_qwen3_vl_vllm.sh
# 换卡：CUDA_VISIBLE_DEVICES=1 bash scripts/start_qwen3_vl_vllm.sh
# Thinking：QWEN_MODEL=Qwen/Qwen3-VL-32B-Thinking bash scripts/start_qwen3_vl_vllm.sh
# 本地目录须指定版本：QWEN_MODEL=/实际模型目录 QWEN_VARIANT=thinking bash scripts/start_qwen3_vl_vllm.sh
# 只打印命令：DRY_RUN=1 bash scripts/start_qwen3_vl_vllm.sh
# 默认沿用视频输入（每条请求最多一个视频，1 fps）；TEXT_ONLY=1 关闭视觉模块。
# 图片输入可追加：--limit-mm-per-prompt '{"image":1,"video":1,"audio":0}'

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
QWEN_VARIANT="${QWEN_VARIANT:-}"
detected_variant=""
case "$QWEN_MODEL" in
  Qwen/Qwen3-VL-32B-Instruct) detected_variant=instruct ;;
  Qwen/Qwen3-VL-32B-Thinking) detected_variant=thinking ;;
  *)
    if [[ -z "$QWEN_VARIANT" ]]; then
      echo "无法识别 QWEN_MODEL 版本；自定义模型目录须设置 QWEN_VARIANT=instruct 或 thinking。" >&2
      exit 1
    fi
    ;;
esac
if [[ -n "$detected_variant" && -n "$QWEN_VARIANT" && "$QWEN_VARIANT" != "$detected_variant" ]]; then
  echo "QWEN_VARIANT=$QWEN_VARIANT 与 QWEN_MODEL=$QWEN_MODEL 冲突。" >&2
  exit 1
fi
QWEN_VARIANT="${QWEN_VARIANT:-$detected_variant}"
case "$QWEN_VARIANT" in
  instruct) served_model_id=Qwen/Qwen3-VL-32B-Instruct ;;
  thinking) served_model_id=Qwen/Qwen3-VL-32B-Thinking ;;
  *)
    echo "QWEN_VARIANT 必须是 instruct 或 thinking。" >&2
    exit 1
    ;;
esac
# 模型名是实验身份，禁止用追加参数覆盖。
for argument in "$@"; do
  case "$argument" in
    --served-model-name|--served-model-name=*)
      echo "--served-model-name 由 QWEN_MODEL/QWEN_VARIANT 固定，不能覆盖。" >&2
      exit 1
      ;;
  esac
done

# dry-run 只构造命令，允许在没有服务器 Conda/vLLM 的机器上检查部署参数。
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "找不到 conda；请先将 Conda 的 bin 目录加入 PATH。" >&2
    exit 1
  fi
  conda_base="$(conda info --base)"
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "${QWEN_CONDA_ENV:-vllm}"
fi

export HF_HOME="${HF_HOME:-/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
# 模型已下载，默认离线加载；缓存缺失时直接报错。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# 沿用参考脚本在缺少匹配 nvcc 时使用的采样回退。
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
# 32B BF16 权重和全注意力 KV cache 占用较大，单卡先用 16K；按显存调整。
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8192}"
if [[ ! "$MAX_OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_OUTPUT_TOKENS 必须是正整数。" >&2
  exit 1
fi
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8003}"

vllm_bin="${CONDA_PREFIX:+$CONDA_PREFIX/bin/}vllm"
if [[ "${DRY_RUN:-0}" != "1" && ! -x "$vllm_bin" ]]; then
  echo "当前 Conda 环境没有 vllm 可执行文件：$vllm_bin" >&2
  exit 1
fi

# vLLM 原生支持 Qwen3VLForConditionalGeneration，无需架构覆盖或 remote code。
# 首次请求均不设置 thinking 开关；Thinking 使用固定的思考模板。
# 官方部署说明：https://github.com/QwenLM/Qwen3-VL#deployment
cmd=(
  "$vllm_bin" serve "$QWEN_MODEL"
  --served-model-name "$served_model_id"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --dtype bfloat16
  --override-generation-config "{\"max_new_tokens\":$MAX_OUTPUT_TOKENS}"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --host "$VLLM_HOST" --port "$VLLM_PORT"
)

if [[ "$QWEN_VARIANT" == "thinking" ]]; then
  cmd+=(
    --reasoning-parser qwen3
    --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
  )
fi

if [[ "${TEXT_ONLY:-0}" == "1" ]]; then
  cmd+=(--language-model-only)
else
  MEDIA_ROOT="${MEDIA_ROOT:-$repo_root/videos}"
  if [[ ! -d "$MEDIA_ROOT" ]]; then
    echo "媒体目录不存在：$MEDIA_ROOT；可设置 MEDIA_ROOT 或 TEXT_ONLY=1。" >&2
    exit 1
  fi
  MEDIA_ROOT="$(cd -- "$MEDIA_ROOT" && pwd -P)"
  cmd+=(
    --allowed-local-media-path "$MEDIA_ROOT"
    --limit-mm-per-prompt '{"image":0,"video":1,"audio":0}'
    # 加载原始帧，交给 Qwen3-VL 处理器按 1 fps 采样一次。
    --media-io-kwargs '{"video":{"video_backend":"opencv","num_frames":-1,"fps":-1}}'
    --mm-processor-kwargs '{"fps":1,"do_sample_frames":true}'
  )
fi
# 首轮关闭 torch.compile / CUDA graph；验收后可设 ENFORCE_EAGER=0。
# eager 模式不保证所有原生算子都无需 JIT 编译。
if [[ "${ENFORCE_EAGER:-1}" == "1" ]]; then
  cmd+=(--enforce-eager)
fi

cmd+=("$@")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q HF_HOME=%q HF_HUB_CACHE=%q HF_HUB_OFFLINE=%q LD_LIBRARY_PATH=%q VLLM_USE_FLASHINFER_SAMPLER=%q ' \
    "$CUDA_VISIBLE_DEVICES" "$HF_HOME" "$HF_HUB_CACHE" "$HF_HUB_OFFLINE" \
    "${LD_LIBRARY_PATH:-}" "$VLLM_USE_FLASHINFER_SAMPLER"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
