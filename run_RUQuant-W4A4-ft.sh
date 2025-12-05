#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --epochs 20 \
    --wbits 4 \
    --abits 4 \
    --model /path/to/your/model \
    --alpha 0.6 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --eval_ppl \
    --lh \

