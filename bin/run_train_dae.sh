#!/usr/bin/env bash

set -exu

echo "CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES"

python train_dae.py