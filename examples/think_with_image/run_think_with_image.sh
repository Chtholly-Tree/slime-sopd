#!/bin/bash
#
# Think-with-Image: VLM multi-turn tool-calling RL training
#
# This script trains a Qwen3-VL model using GRPO on the our dataset with multi-turn tool-calling rollout.
# The model learns to use image_zoom_in_tool and python_executor to gather visual and numerical information before providing a final answer.

set -ex

# ========================= Configuration =========================
export HF_ENDPOINT=https://hf-mirror.com 
export https_proxy=10.140.15.68:3128

TRAIN_BACKEND="megatron"
MODEL_PATH="/mnt/tidal-alsh01/dataset/redone/zengyu/video_search_multi_agent/models/Qwen3-VL-2B-Instruct"
TRAIN_DATASET_PATH="/mnt/tidal-alsh01/dataset/redone/zengyu/video_search_multi_agent/slime-tool-opd/examples/think_with_image/data_example.jsonl"
TEST_DATASET_PATH="/mnt/tidal-alsh01/dataset/redone/zengyu/video_search_multi_agent/slime-tool-opd/examples/think_with_image/data_example.jsonl"
NUM_GPUS=8
USE_EXTERNAL_RAY=0
WANDB_API_KEY="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"
WANDB_PROJ="slime-think-with-image"
WANDB_EXP_NAME="Qwen3-VL-2B-Instruct_think_with_image_debug"

# Source megatron model args
source "/mnt/tidal-alsh01/dataset/redone/zengyu/video_search_multi_agent/slime-tool-opd/scripts/models/qwen3-1.7B.sh"

# LLM Judge reward arguments
LLM_JUDGE_API_KEY="QST4b7bfb2f16b7889c6aed4c7a33274c99"
LLM_JUDGE_BASE_URL="https://maas.devops.xiaohongshu.com/v1"
LLM_JUDGE_MODEL="qwen3.5-35b-a3b"

# ========================= Configuration =========================


# Cleanup
pkill -9 sglang 2>/dev/null || true
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force 2>/dev/null || true
   pkill -9 ray 2>/dev/null || true
fi
pkill -9 slime 2>/dev/null || true
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray 2>/dev/null || true
fi
pkill -9 slime 2>/dev/null || true
pkill -9 redis 2>/dev/null || true

export PYTHONBUFFERED=16

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Common args
CKPT_ARGS=(
   --hf-checkpoint ${MODEL_PATH}
   --rotary-base 5000000
)

# Rollout args — key difference from single-turn geo3k: custom generate function
ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATASET_PATH}
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   # --rm-type llm_judge
   --custom-rm-path examples.think_with_image.rewards.llm_judge.async_rm
   --num-rollout 1
   --rollout-batch-size 1
   --n-samples-per-prompt 8
   --rollout-max-response-len 10240
   --rollout-temperature 1
   --global-batch-size 8
   # Custom multi-turn tool-calling rollout
   --custom-generate-function-path examples.think_with_image.rollout.generate
   --custom-config-path examples/think_with_image/config.yaml
)

# Required for VLM datasets
MULTIMODAL_KEYS='{"image": "images"}'

LLM_JUDGE_ARGS=(
   --llm-judge-api-key "${LLM_JUDGE_API_KEY}"
   --llm-judge-base-url "${LLM_JUDGE_BASE_URL}"
   --llm-judge-model "${LLM_JUDGE_MODEL}"
   --llm-judge-max-tokens 256
   --llm-judge-temperature 0.0
   --llm-judge-max-retries 3
   --llm-judge-batch-size 64
)

# Eval args
EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data "test" ${TEST_DATASET_PATH}
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 10240
)

# GRPO args
GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# Optimizer args
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# SGLang args
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256
)

# Wandb args
if [ -n "$WANDB_API_KEY" ]; then
   WANDB_ARGS=(
      # --use-wandb
      --wandb-project ${WANDB_PROJ}
      --wandb-group ${WANDB_EXP_NAME}
      --wandb-key ${WANDB_API_KEY}
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi

MISC_ARGS=(
   --colocate
)

# Backend args (megatron)
BACKEND_ARGS=(
   --train-backend megatron
   --load /${MODEL_PATH}
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
)

# Start Ray if not using external Ray
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# Build runtime env
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${LLM_JUDGE_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]}
