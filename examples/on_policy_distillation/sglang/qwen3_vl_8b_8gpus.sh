CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
    --model-path /mnt/tidal-alsh01/dataset/redone/zengyu/pretrain_model/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port 13141 \
    --tp-size  8 \
    --mem-fraction-static 0.8 \
    --context-length 262144 \
    --reasoning-parser qwen3
