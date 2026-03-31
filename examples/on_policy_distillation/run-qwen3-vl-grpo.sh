#!/bin/bash

# Qwen3-VL OPD training script adapted from geo3k_vlm.
# Usage:
#   bash examples/on_policy_distillation/run-qwen3-vl-opd.sh

set -ex

TRAIN_BACKEND="megatron"
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-8B-Instruct"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
USE_EXTERNAL_RAY=${SLIME_SCRIPT_EXTERNAL_RAY:-0}

TRAIN_DATA=${SLIME_SCRIPT_TRAIN_DATA:-"/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/geo3k_imgurl/train.parquet"}
EVAL_DATA=${SLIME_SCRIPT_EVAL_DATA:-"/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/geo3k_imgurl/test.parquet"}
HF_MODEL_PATH=${SLIME_SCRIPT_HF_MODEL_PATH:-"/mnt/tidal-alsh01/dataset/redone/zengyu/pretrain_model/${MODEL_NAME}"}

# Teacher server (SGLang, for OPD teacher logprobs)
TEACHER_IP=${SLIME_SCRIPT_TEACHER_IP:-"10.144.203.147"}
TEACHER_PORT=${SLIME_SCRIPT_TEACHER_PORT:-13141}

# Judge server (vLLM, for correctness scoring)
JUDGE_IP=${SLIME_SCRIPT_JUDGE_IP:-"10.144.203.157"}
JUDGE_PORT=${SLIME_SCRIPT_JUDGE_PORT:-8006}
JUDGE_MODEL=${SLIME_SCRIPT_JUDGE_MODEL:-"Qwen3-VL-8B-Instruct"}

VALID_MODELS="
  Qwen2.5-VL-3B-Instruct
  Qwen2.5-VL-7B-Instruct
  Qwen2.5-VL-32B-Instruct
  Qwen2.5-VL-72B-Instruct
  Qwen3-VL-2B-Instruct
  Qwen3-VL-4B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-30B-A3B-Instruct
  Qwen3-VL-235B-A22B-Instruct
  Qwen3-VL-2B-Thinking
  Qwen3-VL-4B-Thinking
  Qwen3-VL-8B-Thinking
  Qwen3-VL-30B-A3B-Thinking
  Qwen3-VL-235B-A22B-Thinking
"
if ! echo "$VALID_MODELS" | grep -qw "$MODEL_NAME"; then
   echo "Error: MODEL_NAME must be one of: $VALID_MODELS"
   exit 1
fi

MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# Cleanup
pkill -9 sglang || true
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force || true
   pkill -9 ray || true
fi
pkill -9 slime || true
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray || true
fi
pkill -9 slime || true
pkill -9 redis || true

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Check teacher / judge service
# curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null
# curl -sf http://$JUDGE_IP:$JUDGE_PORT/v1/models > /dev/null

echo "Teacher model server is up at $TEACHER_IP:$TEACHER_PORT"
echo "Judge model server is up at $JUDGE_IP:$JUDGE_PORT"

SAVE_PATH=${SLIME_SCRIPT_SAVE_PATH:-"/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/slime/checkpoints/qwen3-vl-8B_geo3k_grpo_n-8_slime"}

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --rotary-base 5000000
   --save ${SAVE_PATH}
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 3000
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1.0
   --global-batch-size 512
   --balance-data
)

MULTIMODAL_KEYS='{"image": "images"}'

# RM_ARGS=(
#    --custom-rm-path slime.rollout.multimodal_opd.reward_func_math
#    --custom-reward-post-process-path slime.rollout.multimodal_opd.post_process_grpo
# )

LOG_ARGS=(
   --custom-rollout-log-function-path slime.rollout.multimodal_log.custom_rollout_log
   --custom-eval-rollout-log-function-path slime.rollout.multimodal_log.custom_eval_log
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data geo3k ${EVAL_DATA}
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 8192
   --eval-top-p 1
   --eval-input-key problem
   --eval-label-key answer
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # opd + reward as adv
#    --use-opd
#    --opd-kl-coef 1.0

   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
#    --grpo-std-normalization
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-opd
   --wandb-group qwen3-vl-8B_geo3k_grpo_n-8_slime
   --wandb-key "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"
)

MISC_ARGS=(
   --colocate
)

BACKEND_ARGS=(
   --train-backend megatron
   --load ${HF_MODEL_PATH}
   --model-name qwen3vl
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
   --max-tokens-per-gpu 8196
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
)

SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"

if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"JUDGE_URL\": \"http://$JUDGE_IP:$JUDGE_PORT\",
    \"JUDGE_MODEL\": \"$JUDGE_MODEL\",
    \"USE_TEACHER_CONTEXT\": \"0\"
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
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]} \
   ${LOG_ARGS[@]}


