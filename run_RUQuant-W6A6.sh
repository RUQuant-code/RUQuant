#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py \
    --block_size 128 \
    --max_rotation_step 16 \
    --epochs 0 \
    --wbits 6 \
    --abits 6 \
    --model model_path \
    --alpha 0.6 \
    --smooth \
    --eval_ppl \

