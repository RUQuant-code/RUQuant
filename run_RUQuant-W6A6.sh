#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --epochs 0 \
    --wbits 6 \
    --abits 6 \
    --model /path/to/your/model \
    --alpha 0.6 \
    --smooth \
    --lac 1 \
    --swc 1 \
    --eval_ppl \



