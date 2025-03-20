#!/bin/bash

# Install debugpy if not already installed
# pip install debugpy

# Run with debugpy
# This will wait for a debugger to attach on port 5678
# python -m debugpy --listen 5678 --wait-for-client exp_scripts/video_tokenizer.py

# python exp_scripts/video_tokenizer.py

# Autoencoding videos using `Cosmos-CV` with a compression rate of 8x8x8.
model_name="Cosmos-1.0-Tokenizer-DV8x16x16"
python -m debugpy --listen 5678 --wait-for-client \
    cosmos_tokenizer/video_cli.py \
    --video_pattern 'test_data/video.mp4' \
    --mode=torch \
    --tokenizer_type=DV \
    --temporal_compression=8 \
    --spatial_compression=16 \
    --checkpoint_enc pretrained_ckpts/${model_name}/encoder.jit \
    --checkpoint_dec pretrained_ckpts/${model_name}/decoder.jit