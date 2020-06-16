#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

python run.py \
--data_dir ../dataset \
--w2v_path /workspace/word-vec \
--labels ../dataset/labels.txt \
--epochs 3

