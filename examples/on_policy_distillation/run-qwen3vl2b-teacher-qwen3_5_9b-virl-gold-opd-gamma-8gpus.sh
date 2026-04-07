#!/bin/bash

# Qwen3-VL GOLD training with discounted OPD token rewards.
# This variant uses teacher_log_probs - student_log_probs as token rewards and
# applies a gamma-discounted return before adding them to the base advantages.
# Usage example:
#   export SLIME_SCRIPT_EXTERNAL_RAY=1
#   export SLIME_SCRIPT_NUM_NODES=8
#   export SLIME_SCRIPT_GPUS_PER_NODE=8
#   export SLIME_SCRIPT_RAY_JOB_ADDR="http://127.0.0.1:8265"
#   bash examples/on_policy_distillation/run-qwen3vl2b-teacher-qwen3_5_9b-virl-gold-opd-gamma-8gpus.sh

set -ex

TRAIN_BACKEND="megatron"
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-2B-Instruct"}

# ===== cluster config =====
NUM_NODES=${SLIME_SCRIPT_NUM_NODES:-1}
GPUS_PER_NODE=${SLIME_SCRIPT_GPUS_PER_NODE:-8}
USE_EXTERNAL_RAY=${SLIME_SCRIPT_EXTERNAL_RAY:-0}
RAY_JOB_ADDR=${SLIME_SCRIPT_RAY_JOB_ADDR:-"http://127.0.0.1:8265"}

# ===== data / model =====
TRAIN_DATA=${SLIME_SCRIPT_TRAIN_DATA:-"/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/ViRL39K/train.parquet"}
EVAL_DATA=${SLIME_SCRIPT_EVAL_DATA:-"/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/ViRL39K/test.parquet"}
HF_MODEL_PATH=${SLIME_SCRIPT_HF_MODEL_PATH:-"/mnt/tidal-alsh01/dataset/redone/zengyu/pretrain_model/${MODEL_NAME}"}

# ===== teacher server =====
TEACHER_IP=${SLIME_SCRIPT_TEACHER_IP:-"10.144.205.174"}
TEACHER_PORT=${SLIME_SCRIPT_TEACHER_PORT:-13141}
JUDGE_IP=${SLIME_SCRIPT_JUDGE_IP:-"10.144.206.196"}
JUDGE_PORT=${SLIME_SCRIPT_JUDGE_PORT:-13141}
OPD_GAMMA=${SLIME_SCRIPT_OPD_GAMMA:-0.95}
OPD_KL_COEF=${SLIME_SCRIPT_OPD_KL_COEF:-1.0}
OPD_REWARD_CLAMP_MIN=${SLIME_SCRIPT_OPD_REWARD_CLAMP_MIN:--10.0}

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

curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null
echo "Teacher model server is up at $TEACHER_IP:$TEACHER_PORT"

curl -sf http://$JUDGE_IP:$JUDGE_PORT/health_generate > /dev/null
echo "Judge model server is up at $JUDGE_IP:$JUDGE_PORT"

cd /mnt/tidal-alsh01/dataset/redone/zengyu/xikun/slime-2.4

SAVE_PATH=${SLIME_SCRIPT_SAVE_PATH:-"/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/slime-2.4/checkpoints/qwen3-vl-2B_teacher_qwen3.5-9B_virl_gold_opd_gamma_8gpus_bs64_2.4"}

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --save ${SAVE_PATH}
   --save-interval 200
)

ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
   --input-key question
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --global-batch-size 512
   --balance-data
)

MULTIMODAL_KEYS='{"image": "images"}'

RM_ARGS=(
   --custom-rm-path slime.rollout.multimodal_gold.reward_func_judge
   --custom-reward-post-process-path slime.rollout.multimodal_gold.post_process_rewards
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data virl ${EVAL_DATA}
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 8192
   --eval-top-p 1
   --eval-input-key question
   --eval-label-key answer
)

GOLD_ARGS=(
   --advantage-estimator reinforce_plus_plus
   --gamma ${OPD_GAMMA}
   --use-opd
   --opd-kl-coef ${OPD_KL_COEF}
   --opd-gamma ${OPD_GAMMA}
   --opd-reward-clamp-min ${OPD_REWARD_CLAMP_MIN}
   --opd-type sglang

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
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-opd
   --wandb-group qwen3-vl-2B_teacher_qwen3.5-9B_virl_gold_opd_gamma_8gpus_bs64_2.4
   --wandb-key "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"
)

MISC_ARGS=(
   --colocate
)

BACKEND_ARGS=(
   --train-backend megatron
   --load ${HF_MODEL_PATH}
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
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]} \
   ${LOG_ARGS[@]}
