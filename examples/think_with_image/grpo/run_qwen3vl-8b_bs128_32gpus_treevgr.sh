#!/bin/bash

# Qwen3-VL GOLD-advantage training script (8 nodes x 8 GPUs)
# Usage example:
#   export SLIME_SCRIPT_EXTERNAL_RAY=1
#   export SLIME_SCRIPT_NUM_NODES=8
#   export SLIME_SCRIPT_GPUS_PER_NODE=8
#   export SLIME_SCRIPT_RAY_JOB_ADDR="http://127.0.0.1:8265"   # if running on Ray head node
#   bash examples/on_policy_distillation/run-qwen3-vl-gold-8x8.sh

set -ex

EXP_NAME=${SLIME_SCRIPT_EXP_NAME:-"qwen3-vl-8b_treevgr_32gpus_bs128_grpo"}

TRAIN_BACKEND="megatron"
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-8B-Instruct"}

# ===== cluster config =====
NUM_NODES=${SLIME_SCRIPT_NUM_NODES:-4}
GPUS_PER_NODE=${SLIME_SCRIPT_GPUS_PER_NODE:-8}
USE_EXTERNAL_RAY=${SLIME_SCRIPT_EXTERNAL_RAY:-1}
RAY_JOB_ADDR=${SLIME_SCRIPT_RAY_JOB_ADDR:-"http://127.0.0.1:8265"}   # run on head node -> 127.0.0.1 is OK

# ===== data / model =====
TRAIN_DATASET_PATH="/mnt/tidal-alsh01/dataset/redone/zengyu/video_search_multi_agent/TreeVGR/train_think_with_image_filter.jsonl@[0:10240]"
TEST_DATASET_PATH="/mnt/tidal-alsh01/dataset/redone/zengyu/video_search_multi_agent/TreeVGR/test_think_with_image.jsonl"
HF_MODEL_PATH=${SLIME_SCRIPT_HF_MODEL_PATH:-"/mnt/tidal-alsh01/dataset/redone/zengyu/pretrain_model/${MODEL_NAME}"}

# ===== teacher server / pool =====
TEACHER_IP=${SLIME_SCRIPT_TEACHER_IP:-"10.144.201.87"}
TEACHER_PORT=${SLIME_SCRIPT_TEACHER_PORT:-13141}
TEACHER_POOL_CONFIG=${SLIME_SCRIPT_TEACHER_POOL_CONFIG:-"/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/slime-2.4/examples/on_policy_distillation/config/teacher_pool.yaml"}
TEACHER_MODEL_NAME=${SLIME_SCRIPT_TEACHER_MODEL_NAME:-"qwen3_5_397b"}
TEACHER_REQUEST_MAX_CONCURRENCY=${SLIME_SCRIPT_TEACHER_REQUEST_MAX_CONCURRENCY:-128}
GOLD_TEACHER_HF_CHECKPOINT=${SLIME_SCRIPT_GOLD_TEACHER_HF_CHECKPOINT:-"/mnt/tidal-alsh01/dataset/redone/zengyu/pretrain_model/Qwen3.5-397B-A17B"}
JUDGE_IP=${SLIME_SCRIPT_JUDGE_IP:-"10.144.206.243"}
JUDGE_PORT=${SLIME_SCRIPT_JUDGE_PORT:-13141}

# LLM Judge reward arguments
JUDGE_URL="http://10.144.205.30:13141"
JUDGE_API_KEY="EMPTY"
JUDGE_MODEL="Qwen3-VL-30B-A3B-Instruct"

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

# ===== cleanup =====
pkill -9 sglang || true
sleep 3

# external ray cluster already exists -> do not stop ray
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

# ===== teacher health check =====
if [ -n "$TEACHER_POOL_CONFIG" ] && [ -n "$TEACHER_MODEL_NAME" ]; then
   echo "Using teacher pool config: $TEACHER_POOL_CONFIG (model: $TEACHER_MODEL_NAME)"
else
   curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null
   echo "Teacher model server is up at $TEACHER_IP:$TEACHER_PORT"
fi

# ===== judge health check =====
curl -sf http://$JUDGE_IP:$JUDGE_PORT/health_generate > /dev/null
echo "Judge model server is up at $JUDGE_IP:$JUDGE_PORT"

cd /mnt/tidal-alsh01/dataset/redone/zengyu/xikun/slime-2.4

SAVE_PATH=${SLIME_SCRIPT_SAVE_PATH:-"/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/slime-2.4/checkpoints/${EXP_NAME}"}

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --save ${SAVE_PATH}
   --save-interval 100
)


ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATASET_PATH}
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-max-response-len 20480
   --rollout-temperature 1.0
   --global-batch-size 1024
   --balance-data
   --custom-generate-function-path examples.think_with_image.rollout.generate
   --custom-config-path examples/think_with_image/config.yaml
)

LLM_JUDGE_ARGS=(
   --llm-judge-api-key "${JUDGE_API_KEY}"
   --llm-judge-base-url "${JUDGE_URL}"
   --llm-judge-model "${JUDGE_MODEL}"
   --llm-judge-max-tokens 256
   --llm-judge-temperature 0.0
   --llm-judge-max-retries 3
   --llm-judge-batch-size 256
)

MULTIMODAL_KEYS='{"image": "images"}'

RM_ARGS=(
   --custom-rm-path examples.think_with_image.rewards.llm_judge.async_rm
   --judge-url http://$JUDGE_IP:$JUDGE_PORT/generate
   --teacher-request-max-concurrency ${TEACHER_REQUEST_MAX_CONCURRENCY}
)

if [ -n "$TEACHER_POOL_CONFIG" ] && [ -n "$TEACHER_MODEL_NAME" ]; then
   RM_ARGS+=(
      --teacher-pool-config ${TEACHER_POOL_CONFIG}
      --teacher-model-name ${TEACHER_MODEL_NAME}
   )
fi

if [ -n "$GOLD_TEACHER_HF_CHECKPOINT" ]; then
   RM_ARGS+=(
      --gold-teacher-hf-checkpoint ${GOLD_TEACHER_HF_CHECKPOINT}
   )
fi

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data treevgr ${TEST_DATASET_PATH}
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 20480
   --eval-top-p 1
   --eval-input-key problem
   --eval-label-key answer
)

GOLD_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
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
   --sglang-server-concurrency 512
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-think-with-image
   --wandb-group ${EXP_NAME}
   --wandb-key "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"
)

MISC_ARGS=(
   --colocate
)

BACKEND_ARGS=(
   --train-backend megatron
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 2
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
   --multimodal-load-workers 8
   --multimodal_task_batch_size 128
   --megatron-to-hf-mode bridge
)

SLIME_DIR="/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/slime-2.4"
MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"

# local ray only for single-node mode
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head \
      --node-ip-address ${MASTER_ADDR} \
      --num-gpus ${GPUS_PER_NODE} \
      --disable-usage-stats \
      --dashboard-host=0.0.0.0 \
      --dashboard-port=8265
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

ray job submit --address="${RAY_JOB_ADDR}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node ${GPUS_PER_NODE} \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${GOLD_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${LLM_JUDGE_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]} \
   ${LOG_ARGS[@]}

