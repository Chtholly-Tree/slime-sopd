# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
#     --model-path /mnt/tidal-alsh01/dataset/redone/zengyu/pretrain_model/Qwen3.5-9B \
#     --host 0.0.0.0 \
#     --port 13141 \
#     --tp 16 \
#     --nnodes 2 \
#     --node-rank 0 \
#     --dist-init-addr 10.144.201.87:50000 \
#     --chunked-prefill-size 4096 \
#     --mem-fraction-static 0.6 \
#     --context-length 32768 \
#     --watchdog-timeout 1800 \
#     --mm-per-request-timeout 300

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
    --model-path /mnt/tidal-alsh01/dataset/redone/zengyu/pretrain_model/Qwen3.5-9B \
    --host 0.0.0.0 \
    --port 13141 \
    --tp 16 \
    --nnodes 2 \
    --node-rank 1 \
    --dist-init-addr 10.144.201.87:50000 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.6 \
    --context-length 32768 \
    --watchdog-timeout 1800 \
    --mm-per-request-timeout 300
