#!/bin/bash

set -ex

# Logits-level GOLD launch template for multimodal cross-vocab distillation.
#
# This script is adapted from the existing token-level GOLD examples, but is
# organized around the planned logits-level pipeline:
# 1. keep `slime.rollout.multimodal_gold` group alignment
# 2. cache alignment metadata / teacher distribution information
# 3. switch training loss from token-level teacher_log_probs to logits-level ULD/GOLD
#
# IMPORTANT:
# - The commented flags in `LOGITS_GOLD_ARGS` are placeholders for the new code
#   path and should be uncommented after the implementation lands.
# - The teacher endpoint must be able to return full-vocab logits/logprobs or a
#   sufficiently large top-k approximation for logits-level distillation.

# ===== experiment =====
EXP_NAME=${SLIME_SCRIPT_EXP_NAME:-"qwen3-vl-2B_teacher_qwen3.5-9B_virl_logits_gold_bs256"}
TRAIN_BACKEND="megatron"
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-2B-Instruct"}

# ===== cluster config =====
NUM_NODES=${SLIME_SCRIPT_NUM_NODES:-8}
GPUS_PER_NODE=${SLIME_SCRIPT_GPUS_PER_NODE:-4}
USE_EXTERNAL_RAY=${SLIME_SCRIPT_EXTERNAL_RAY:-1}
RAY_JOB_ADDR=${SLIME_SCRIPT_RAY_JOB_ADDR:-"http://127.0.0.1:8265"}

# ===== data / model =====
TRAIN_DATA=${SLIME_SCRIPT_TRAIN_DATA:-"/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/ViRL39K/train.parquet"}
EVAL_DATA=${SLIME_SCRIPT_EVAL_DATA:-"/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/ViRL39K/test.parquet"}
HF_MODEL_PATH=${SLIME_SCRIPT_HF_MODEL_PATH:-"/mnt/tidal-alsh01/dataset/redone/zengyu/pretrain_model/${MODEL_NAME}"}

# ===== teacher / judge =====
# For logits-level GOLD we plan to route both async raw-reward computation and
# async teacher forward through a dedicated SGLang API helper class/module.
TEACHER_IP=${SLIME_SCRIPT_TEACHER_IP:-"10.144.204.115"}
TEACHER_PORT=${SLIME_SCRIPT_TEACHER_PORT:-13141}
TEACHER_POOL_CONFIG=${SLIME_SCRIPT_TEACHER_POOL_CONFIG:-"/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/slime-2.4/examples/on_policy_distillation/config/teacher_pool.yaml"}
TEACHER_MODEL_NAME=${SLIME_SCRIPT_TEACHER_MODEL_NAME:-"qwen3_5_397b"}
TEACHER_REQUEST_MAX_CONCURRENCY=${SLIME_SCRIPT_TEACHER_REQUEST_MAX_CONCURRENCY:-256}
JUDGE_IP=${SLIME_SCRIPT_JUDGE_IP:-"10.144.205.231"}
JUDGE_PORT=${SLIME_SCRIPT_JUDGE_PORT:-13141}
GOLD_TEACHER_HF_CHECKPOINT=${SLIME_SCRIPT_GOLD_TEACHER_HF_CHECKPOINT:-"/mnt/tidal-alsh01/dataset/redone/zengyu/pretrain_model/Qwen3.5-9B"}

# ===== rollout / training =====
ROLLOUT_BATCH_SIZE=${SLIME_SCRIPT_ROLLOUT_BATCH_SIZE:-256}
N_SAMPLES_PER_PROMPT=${SLIME_SCRIPT_N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${SLIME_SCRIPT_GLOBAL_BATCH_SIZE:-512}
NUM_STEPS_PER_ROLLOUT=${SLIME_SCRIPT_NUM_STEPS_PER_ROLLOUT:-4}
ROLLOUT_MAX_RESPONSE_LEN=${SLIME_SCRIPT_ROLLOUT_MAX_RESPONSE_LEN:-8192}
EVAL_MAX_RESPONSE_LEN=${SLIME_SCRIPT_EVAL_MAX_RESPONSE_LEN:-8192}

# ===== logits-level GOLD hyperparameters =====
GOLD_STUDENT_TEMPERATURE=${SLIME_SCRIPT_GOLD_STUDENT_TEMPERATURE:-1.0}
GOLD_TEACHER_TEMPERATURE=${SLIME_SCRIPT_GOLD_TEACHER_TEMPERATURE:-1.0}
GOLD_DISTILL_WEIGHT=${SLIME_SCRIPT_GOLD_DISTILL_WEIGHT:-1.0}
GOLD_CE_WEIGHT=${SLIME_SCRIPT_GOLD_CE_WEIGHT:-0.0}
GOLD_TOPK_LOGPROBS=${SLIME_SCRIPT_GOLD_TOPK_LOGPROBS:-0}
GOLD_RESPONSE_PREFIX_RATIO=${SLIME_SCRIPT_GOLD_RESPONSE_PREFIX_RATIO:-1.0}

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

# ===== cleanup =====
pkill -9 sglang || true
sleep 3

if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force || true
   pkill -9 ray || true
fi

pkill -9 slime || true
sleep 3
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

# ===== teacher health check =====
if [ -n "$TEACHER_POOL_CONFIG" ] && [ -n "$TEACHER_MODEL_NAME" ]; then
   echo "Using teacher pool config: $TEACHER_POOL_CONFIG (model: $TEACHER_MODEL_NAME)"
else
   curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" > /dev/null
   echo "Teacher model server is up at ${TEACHER_IP}:${TEACHER_PORT}"
fi

# ===== judge health check =====
curl -sf "http://${JUDGE_IP}:${JUDGE_PORT}/health_generate" > /dev/null
echo "Judge model server is up at ${JUDGE_IP}:${JUDGE_PORT}"

SLIME_DIR=${SLIME_SCRIPT_SLIME_DIR:-"/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/slime-2.4"}
cd "${SLIME_DIR}"

SAVE_PATH=${SLIME_SCRIPT_SAVE_PATH:-"${SLIME_DIR}/checkpoints/${EXP_NAME}"}
MULTIMODAL_KEYS='{"image": "images"}'

CKPT_ARGS=(
   --hf-checkpoint "${HF_MODEL_PATH}"
   --save "${SAVE_PATH}"
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA}"
   --input-key question
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1.0
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --balance-data
   --multimodal-load-workers 32
   --custom-generate-function-path examples.logits_level_gold.rollout.generate
)

RM_ARGS=(
   --custom-rm-path examples.logits_level_gold.reward.reward_func
   --custom-reward-post-process-path examples.logits_level_gold.runtime.post_process_logits_level_gold
   --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
   --judge-url "http://${JUDGE_IP}:${JUDGE_PORT}/generate"
   --teacher-request-max-concurrency "${TEACHER_REQUEST_MAX_CONCURRENCY}"
)

if [ -n "${TEACHER_POOL_CONFIG}" ] && [ -n "${TEACHER_MODEL_NAME}" ]; then
   RM_ARGS+=(
      --teacher-pool-config "${TEACHER_POOL_CONFIG}"
      --teacher-model-name "${TEACHER_MODEL_NAME}"
   )
fi

if [ -n "${GOLD_TEACHER_HF_CHECKPOINT}" ]; then
   RM_ARGS+=(
      --gold-teacher-hf-checkpoint "${GOLD_TEACHER_HF_CHECKPOINT}"
   )
fi

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data virl "${EVAL_DATA}"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
   --eval-top-p 1
   --eval-input-key question
   --eval-label-key answer
)

# Keep the base RL / logging knobs neutral. The actual distillation signal should
# come from the new logits-level GOLD loss path instead of OPD token-level KL.
DISTILLATION_BASE_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
)

# Uncomment / adjust these flags after the logits-level implementation lands.
# This follows the same style as other custom distillation setups: specify the
# custom loss directly, and keep GOLD-specific behavior in dedicated `gold-*`
# hyperparameters.
LOGITS_GOLD_ARGS=(
   --loss-type custom_loss
   --custom-loss-function-path examples.logits_level_gold.topk_distillation_loss.logits_level_gold_loss
   --gold-student-temperature "${GOLD_STUDENT_TEMPERATURE}"
   --gold-teacher-temperature "${GOLD_TEACHER_TEMPERATURE}"
   --gold-distillation-weight "${GOLD_DISTILL_WEIGHT}"
   --gold-cross-entropy-weight "${GOLD_CE_WEIGHT}"
   --gold-train-response-prefix-ratio "${GOLD_RESPONSE_PREFIX_RATIO}"
   --gold-teacher-topk-logprobs "${GOLD_TOPK_LOGPROBS}"
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
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-logits-gold
   --wandb-group "${EXP_NAME}"
)

MISC_ARGS=(
#   --colocate
)

BACKEND_ARGS=(
   --train-backend megatron
   --load "${HF_MODEL_PATH}"
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

MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"

if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head \
      --node-ip-address "${MASTER_ADDR}" \
      --num-gpus "${GPUS_PER_NODE}" \
      --disable-usage-stats \
      --dashboard-host=0.0.0.0 \
      --dashboard-port=8265
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"USE_TEACHER_CONTEXT\": \"0\"
  }
}"

ray job submit --address="${RAY_JOB_ADDR}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes "${NUM_NODES}" \
   --actor-num-gpus-per-node "${GPUS_PER_NODE}" \
   --rollout-num-gpus 32 \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${DISTILLATION_BASE_ARGS[@]} \
   ${LOGITS_GOLD_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]} \
   ${LOG_ARGS[@]}
