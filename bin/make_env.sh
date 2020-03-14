#!/usr/bin/env bash

set -exu

conda create -n cf python=3.6
module add cuda90
conda activate cf
conda install pandas
conda install scipy
pip install tensorflow==1.5
pip install Bottleneck
