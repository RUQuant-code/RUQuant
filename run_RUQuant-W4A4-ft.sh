#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py \
    --block_size 128 \
    --max_rotation_step 16 \
    --epochs 20 \
    --wbits 4 \
    --abits 4 \
    --model model_path \
    --alpha 0.6 \
    --smooth \
    --lac 0.9 \
    --swc 0.8\
    --eval_ppl \
    --lh \